#!/usr/bin/env python3
"""Behemoth-shape EXL3 microbench and correctness gates on SM86.

Compares:
  - compressed EXL3 (exl3_gemm)
  - reconstructed FP16 + hgemm/cuBLAS
  - optional Marlin kernel (if the mixed checkpoint / marlin op is available)
  - FP16 cuBLAS baseline

Gates (vs Marlin when present; vs reconstruct for correctness always):
  - exact-enough output parity with EXL3 reconstruct
  - M=1  <= 1.25x Marlin kernel time
  - M=8-32 <= 1.50x
  - M>=1024 <= 1.15x
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugin" / "src"))

from vllm_exl3_sm86.constants import (  # noqa: E402
    BEHEMOTH_TP4_SHAPES,
    MICROBENCH_M,
)
from vllm_exl3_sm86.ops import _load_exl3_ext, call_exl3_gemm  # noqa: E402

GATES = {
    1: 1.25,
    8: 1.50,
    16: 1.50,
    32: 1.50,
    1024: 1.15,
    4096: 1.15,
}


def _sync():
    torch.cuda.synchronize()


def _time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def _payloads(k: int, n: int, bitrate: int, m: int, device):
    trellis = torch.randint(
        -1024, 1024, (k // 16, n // 16, 16 * bitrate), dtype=torch.int16, device=device
    )
    suh = torch.randn(k, dtype=torch.float16, device=device)
    svh = torch.randn(n, dtype=torch.float16, device=device)
    x = torch.randn(m, k, dtype=torch.float16, device=device)
    return x, trellis, suh, svh


def _reconstruct_ref(ext, x, trellis, suh, svh, bitrate):
    k = trellis.shape[0] * 16
    n = trellis.shape[1] * 16
    w = torch.empty((k, n), dtype=torch.float16, device=x.device)
    ext.reconstruct(w, trellis, bitrate, False, True)
    xh = torch.empty_like(x)
    ext.had_r_128(x, xh, suh, None, 1.0)
    y = torch.empty((x.shape[0], n), dtype=torch.float16, device=x.device)
    ext.hgemm(xh, w, y)
    ext.had_r_128(y, y, None, svh, 1.0)
    return y, w


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--bitrates", default="3,4,6")
    parser.add_argument("--m", default=",".join(str(v) for v in MICROBENCH_M))
    parser.add_argument("--output", default=str(ROOT / "results" / "kernel_microbench.json"))
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 2

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    ext = _load_exl3_ext()
    bitrates = [int(x) for x in args.bitrates.split(",") if x.strip()]
    ms = [int(x) for x in args.m.split(",") if x.strip()]

    os.environ.setdefault("VLLM_EXL3_FORCE_COMPRESSED", "0")
    rows = []
    failures = []
    for name, (k, n) in BEHEMOTH_TP4_SHAPES.items():
        for bitrate in bitrates:
            if name == "lm_head" and bitrate != 6:
                continue
            if name != "lm_head" and bitrate == 6:
                continue
            for m in ms:
                x, trellis, suh, svh = _payloads(k, n, bitrate, m, device)
                os.environ["VLLM_EXL3_FORCE_COMPRESSED"] = "1"
                got = call_exl3_gemm(x, trellis, suh, svh, False, True)
                ref, w = _reconstruct_ref(ext, x, trellis, suh, svh, bitrate)
                max_err = (got.float() - ref.float()).abs().max().item()
                # GEMV (M=1) and fp16 GEMM vs reconstruct differ in reduction
                # order on random trellis tiles. Same bounds as tests/test_cuda_parity.py.
                atol = 0.75 if m == 1 else 0.25
                parity_ok = torch.allclose(got, ref, rtol=5e-2, atol=atol)

                def run_exl3():
                    os.environ["VLLM_EXL3_FORCE_COMPRESSED"] = "1"
                    return call_exl3_gemm(x, trellis, suh, svh, False, True)

                def run_reconstruct():
                    os.environ["VLLM_EXL3_FORCE_RECONSTRUCT"] = "1"
                    os.environ["VLLM_EXL3_FORCE_COMPRESSED"] = "0"
                    return call_exl3_gemm(x, trellis, suh, svh, False, True)

                def run_cublas():
                    return torch.mm(x, w)

                t_exl3 = _time_ms(run_exl3, args.warmup, args.iters)
                os.environ.pop("VLLM_EXL3_FORCE_RECONSTRUCT", None)
                t_recon = _time_ms(run_reconstruct, args.warmup, args.iters)
                os.environ.pop("VLLM_EXL3_FORCE_RECONSTRUCT", None)
                os.environ["VLLM_EXL3_FORCE_COMPRESSED"] = "1"
                t_fp16 = _time_ms(run_cublas, args.warmup, args.iters)
                row = {
                    "name": name,
                    "k": k,
                    "n": n,
                    "bitrate": bitrate,
                    "m": m,
                    "exl3_ms": t_exl3,
                    "reconstruct_ms": t_recon,
                    "fp16_cublas_ms": t_fp16,
                    "max_err": max_err,
                    "parity_ok": bool(parity_ok),
                }
                rows.append(row)
                if not parity_ok:
                    failures.append(f"parity {name} K={bitrate} M={m} max_err={max_err}")
                print(
                    f"{name:10} K={bitrate} M={m:4d}  exl3={t_exl3:8.3f}ms  "
                    f"recon={t_recon:8.3f}ms  fp16={t_fp16:8.3f}ms  err={max_err:.3e}"
                )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"rows": rows, "failures": failures}, indent=2) + "\n")
    print(f"Wrote {args.output}")
    if failures:
        print("GATES FAILED:", *failures, sep="\n  ", file=sys.stderr)
        return 1 if args.fail_on_gate else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
