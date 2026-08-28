#!/usr/bin/env python3
"""Load a small dense Mistral EXL3 checkpoint under vLLM TP1 and TP4 eager."""

from __future__ import annotations

import argparse
import os
import resource
import sys
from pathlib import Path

os.environ.setdefault("VLLM_PLUGINS", "vllm_exl3_sm86")


def _rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)


def _load(model, tp, eager=True):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        quantization="exl3",
        tensor_parallel_size=tp,
        enforce_eager=eager,
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        disable_log_stats=True,
    )
    params = SamplingParams(temperature=0.0, max_tokens=8)
    out = llm.generate(["The capital of France is"], params)
    text = out[0].outputs[0].text
    logits_tp = None
    llm.llm_engine.model_executor.shutdown()
    del llm
    return text, logits_tp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", default="1,4")
    parser.add_argument("--max-host-gib", type=float, default=None)
    args = parser.parse_args()
    model = str(Path(args.model))
    texts = {}
    for tp in (int(x) for x in args.tp.split(",") if x.strip()):
        print(f"Loading TP{tp} eager RSS={_rss_gib():.2f} GiB")
        text, _ = _load(model, tp, eager=True)
        texts[tp] = text
        rss = _rss_gib()
        print(f"TP{tp} greedy={text!r} RSS={rss:.2f} GiB")
        if args.max_host_gib is not None and rss > args.max_host_gib:
            print(
                f"host RSS {rss:.2f} GiB exceeds {args.max_host_gib} GiB; "
                "refusing four-copy staging",
                file=sys.stderr,
            )
            return 1
    if 1 in texts and 4 in texts and texts[1] != texts[4]:
        print(f"TP1/TP4 greedy mismatch: {texts[1]!r} vs {texts[4]!r}", file=sys.stderr)
        return 1
    print("small-mistral validation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
