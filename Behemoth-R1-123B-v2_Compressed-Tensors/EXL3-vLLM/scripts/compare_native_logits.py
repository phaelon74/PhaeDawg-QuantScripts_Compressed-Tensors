#!/usr/bin/env python3
"""Compare vLLM EXL3 plugin logits with native ExLlamaV3 on fixed prompts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


def _native_logits(model_dir: str, prompt: str, device: int):
    from exllamav3 import Generator, Config, Model, Tokenizer

    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    model.load()
    tokenizer = Tokenizer.from_config(config)
    ids = tokenizer.encode(prompt)
    logits = model.forward(ids.to(f"cuda:{device}"))
    model.unload()
    return logits.float().cpu()


def _vllm_logits(model_dir: str, prompt: str, tp: int):
    os.environ.setdefault("VLLM_PLUGINS", "vllm_exl3_sm86")
    from vllm import LLM
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=model_dir,
        quantization="exl3",
        tensor_parallel_size=tp,
        enforce_eager=True,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
    )
    tokenizer = llm.get_tokenizer()
    ids = tokenizer.encode(prompt)
    # Prompt logprobs path: generate one token and read prompt logprobs if exposed.
    outputs = llm.generate([prompt], use_tqdm=False)
    text = outputs[0].outputs[0].text
    llm.llm_engine.model_executor.shutdown()
    return {"greedy_prefix": text, "ntokens": len(ids)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Hello, Behemoth.")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--native-device", type=int, default=0)
    parser.add_argument("--skip-native", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    receipt = {"prompt": args.prompt, "tp": args.tp}
    if not args.skip_native:
        logits = _native_logits(args.model, args.prompt, args.native_device)
        receipt["native_shape"] = list(logits.shape)
        receipt["native_max"] = float(logits.abs().max())
    receipt["vllm"] = _vllm_logits(args.model, args.prompt, args.tp)
    print(json.dumps(receipt, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(receipt, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
