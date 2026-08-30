#!/usr/bin/env python3
"""Standalone decode-kernel nsys target: 88 layers of TP4 projections, M=1.

Does not load the 62G checkpoint. Use with:

  VLLM_EXL3_NVTX=1 nsys profile -o results/nsys_decode_kernel \\
      --trace=cuda,nvtx --force-overwrite true \\
      python scripts/profile_decode_nsys.py --device 0 --codebook 3inst
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugin" / "src"))

from vllm_exl3_sm86.constants import (  # noqa: E402
    BEHEMOTH_LAYERS,
    BEHEMOTH_TP4_SHAPES,
    HEAD_BITRATE,
)
from vllm_exl3_sm86.nvtx import nvtx_range  # noqa: E402
from vllm_exl3_sm86.ops import call_exl3_gemm  # noqa: E402


def _payload(device, k: int, n: int, bitrate: int, m: int):
    trellis = torch.zeros((k // 16, n // 16, 16 * bitrate), dtype=torch.int16, device=device)
    suh = torch.ones(k, dtype=torch.float16, device=device)
    svh = torch.ones(n, dtype=torch.float16, device=device)
    x = torch.zeros((m, k), dtype=torch.float16, device=device)
    return x, trellis, suh, svh


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--layers", type=int, default=BEHEMOTH_LAYERS)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--bitrate", type=int, default=4)
    parser.add_argument("--codebook", choices=("3inst", "mul1"), default="3inst")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    os.environ.setdefault("VLLM_EXL3_NVTX", "1")
    os.environ["VLLM_EXL3_FORCE_COMPRESSED"] = "1"
    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 2
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    mcg = False
    mul1 = args.codebook == "mul1"
    payloads = {}
    for name, (k, n) in BEHEMOTH_TP4_SHAPES.items():
        bitrate = HEAD_BITRATE if name == "lm_head" else args.bitrate
        payloads[name] = _payload(device, k, n, bitrate, args.m)

    def one_token():
        with nvtx_range("exl3.decode_token"):
            for _ in range(args.layers):
                for name in (
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ):
                    x, trellis, suh, svh = payloads[name]
                    with nvtx_range(f"exl3.{name}"):
                        call_exl3_gemm(x, trellis, suh, svh, mcg, mul1)
            x, trellis, suh, svh = payloads["lm_head"]
            with nvtx_range("exl3.lm_head"):
                call_exl3_gemm(x, trellis, suh, svh, mcg, mul1)

    for _ in range(args.warmup):
        one_token()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        one_token()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    median = samples[len(samples) // 2]
    print(
        f"codebook={args.codebook} bitrate={args.bitrate} layers={args.layers} "
        f"M={args.m} median_ms={median:.3f} tok_s={1000.0 / median:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
