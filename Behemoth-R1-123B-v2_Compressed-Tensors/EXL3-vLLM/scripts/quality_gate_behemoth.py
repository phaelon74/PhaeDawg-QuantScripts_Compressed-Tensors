#!/usr/bin/env python3
"""Behemoth TP4 eager quality gate: load, memory, generation, native parity hooks."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_PLUGINS", "vllm_exl3_sm86")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    import torch

    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        quantization="exl3",
        tensor_parallel_size=args.tp,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.94,
        kv_cache_dtype="auto",
        max_num_batched_tokens=8192,
    )
    startup_s = time.perf_counter() - t0
    vram = []
    if torch.cuda.is_available():
        vram = [
            {
                "rank": i,
                "allocated_gib": torch.cuda.memory_allocated(i) / (1024**3),
                "reserved_gib": torch.cuda.memory_reserved(i) / (1024**3),
                "total_gib": torch.cuda.get_device_properties(i).total_memory / (1024**3),
            }
            for i in range(torch.cuda.device_count())
        ]
    params = SamplingParams(temperature=0.0, max_tokens=32)
    out = llm.generate(
        ["Write a single sentence about tensor parallelism."], params
    )
    text = out[0].outputs[0].text
    receipt = {
        "startup_s": startup_s,
        "max_model_len": args.max_model_len,
        "tp": args.tp,
        "vram": vram,
        "greedy": text,
        "kv_headroom_note": "max_model_len=32768 with kv_cache_dtype=auto is the 32K BF16-KV gate",
    }
    print(json.dumps(receipt, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(receipt, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
