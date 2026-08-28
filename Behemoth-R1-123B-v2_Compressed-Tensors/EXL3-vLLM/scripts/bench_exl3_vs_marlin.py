#!/usr/bin/env python3
"""Apples-to-apples EXL3 vs Marlin serving benchmark on TP4.

Prompt lengths 128/1K/4K/8K/16K/32K, 256-token decode, concurrency 1/2/4/16,
cold/warm prefix-cache modes. Writes TTFT/TPOT/throughput/VRAM receipts.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch


def _vram_gib() -> list[float]:
    if not torch.cuda.is_available():
        return []
    return [torch.cuda.max_memory_allocated(i) / (1024**3) for i in range(torch.cuda.device_count())]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return s[idx]


def _run_once(llm, prompt, decode_tokens, temperature=0.0):
    from vllm import SamplingParams

    params = SamplingParams(temperature=temperature, max_tokens=decode_tokens)
    t0 = time.perf_counter()
    outs = llm.generate([prompt] if isinstance(prompt, str) else prompt, params)
    elapsed = time.perf_counter() - t0
    out = outs[0]
    gen = len(out.outputs[0].token_ids)
    prompt_n = len(out.prompt_token_ids)
    metrics = getattr(out, "metrics", None)
    ttft = getattr(metrics, "time_to_first_token", None) if metrics else None
    return {
        "elapsed_s": elapsed,
        "prompt_tokens": prompt_n,
        "decode_tokens": gen,
        "ttft_s": ttft,
        "prefill_tok_s": (prompt_n / ttft) if ttft else None,
        "decode_tok_s": gen / max(elapsed - (ttft or 0), 1e-6),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["exl3", "marlin"], required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--template", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--prompt-lengths", default="128,1024,4096,8192,16384,32768")
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--max-model-len", type=int, default=54272)
    args = parser.parse_args()

    os.environ.setdefault("VLLM_PLUGINS", "vllm_exl3_sm86")
    from vllm import LLM

    quant = "exl3" if args.mode == "exl3" else "compressed-tensors"
    eager = args.mode == "exl3" and os.environ.get("VLLM_EXL3_ALLOW_GRAPHS", "0") != "1"
    llm = LLM(
        model=args.model_dir,
        quantization=quant,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=eager,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.94,
        max_num_batched_tokens=8192,
        kv_cache_dtype="auto",
    )
    lengths = [int(x) for x in args.prompt_lengths.split(",") if x.strip()]
    concs = [int(x) for x in args.concurrency.split(",") if x.strip()]
    tokenizer = llm.get_tokenizer()
    pad_id = tokenizer.bos_token_id or tokenizer.eos_token_id or 1
    results = {"mode": args.mode, "model": args.model_dir, "runs": []}
    for length in lengths:
        prompt_ids = [pad_id] * length
        prompt = tokenizer.decode(prompt_ids)
        for conc in concs:
            prompts = [prompt] * conc
            # cold
            cold = _run_once(llm, prompts, args.decode_tokens)
            # warm prefix
            warm = _run_once(llm, prompts, args.decode_tokens)
            rec = {
                "prompt_len": length,
                "concurrency": conc,
                "cold": cold,
                "warm": warm,
                "peak_vram_gib": _vram_gib(),
            }
            results["runs"].append(rec)
            print(json.dumps(rec))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
