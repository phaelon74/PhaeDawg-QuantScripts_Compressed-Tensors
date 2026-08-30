#!/usr/bin/env python3
"""Behemoth-shape EXL3 microbench and correctness gates on SM86.

Compares:
  - plugin wrapper (allocates output/x_had each call)
  - native exl3_gemm with persistent workspaces
  - reconstructed FP16 + hgemm/cuBLAS
  - FP16 cuBLAS baseline

Default codebook is implicit 3inst (mcg=0, mul1=0), matching ArtusDev 4.25.
Pass --mul1 to time the INT8 GEMV path used by local 4.5-bpw checkpoints.

GATES vs reconstruct when --fail-on-gate is set (Marlin is not assumed):
  - plugin wrapper matches native exl3_gemm (hard)
  - M=1  <= 1.25x reconstruct
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
    BEHEMOTH_LAYERS,
    BEHEMOTH_TP4_SHAPES,
    DECODE_MICROBENCH_M,
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

DECODER_SHAPES = tuple(name for name in BEHEMOTH_TP4_SHAPES if name != "lm_head")


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


def _native_compressed(ext, x, trellis, suh, svh, mcg: bool, mul1: bool, output, x_had):
    ext.exl3_gemm(x, trellis, output, suh, x_had, svh, -1, mcg, mul1, 0)
    return output


def _reconstruct_ref(ext, x, trellis, suh, svh, bitrate: int, mcg: bool, mul1: bool):
    k = trellis.shape[0] * 16
    n = trellis.shape[1] * 16
    w = torch.empty((k, n), dtype=torch.float16, device=x.device)
    ext.reconstruct(w, trellis, bitrate, mcg, mul1)
    xh = torch.empty_like(x)
    ext.had_r_128(x, xh, suh, None, 1.0)
    y = torch.empty((x.shape[0], n), dtype=torch.float16, device=x.device)
    ext.hgemm(xh, w, y)
    ext.had_r_128(y, y, None, svh, 1.0)
    return y, w


def _codebook_name(mcg: bool, mul1: bool) -> str:
    if mcg and mul1:
        return "invalid"
    if mcg:
        return "mcg"
    if mul1:
        return "mul1"
    return "3inst"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--bitrates", default="3,4,5,6")
    parser.add_argument(
        "--m",
        default="",
        help="Token counts to time. Default is decode sizes 1,2,4 unless --full-m.",
    )
    parser.add_argument(
        "--full-m",
        action="store_true",
        help="Time the full crossover sweep instead of decode sizes only.",
    )
    parser.add_argument("--mcg", action="store_true", help="Use mcg codebook.")
    parser.add_argument(
        "--mul1",
        action="store_true",
        help="Use mul1 codebook (INT8 GEMV on Ampere, K<=5). Default is 3inst.",
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--fail-on-gate", action="store_true")
    parser.add_argument(
        "--skip-timing",
        action="store_true",
        help="Correctness only; use after a full timing run.",
    )
    args = parser.parse_args()
    if args.mcg and args.mul1:
        print("mcg and mul1 are mutually exclusive", file=sys.stderr)
        return 2
    codebook = _codebook_name(args.mcg, args.mul1)
    if not args.output:
        suffix = "full" if args.full_m else "decode"
        args.output = str(
            ROOT / "results" / f"kernel_microbench_{codebook}_{suffix}.json"
        )
    default_out_prefix = str(ROOT / "results" / "kernel_microbench")
    if args.skip_timing and args.output.startswith(default_out_prefix):
        args.output = str(ROOT / "results" / f"kernel_wrapper_check_{codebook}.json")

    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 2

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    ext = _load_exl3_ext()
    bitrates = [int(x) for x in args.bitrates.split(",") if x.strip()]
    if args.m:
        ms = [int(x) for x in args.m.split(",") if x.strip()]
    elif args.full_m:
        ms = list(MICROBENCH_M)
    else:
        ms = list(DECODE_MICROBENCH_M)

    os.environ.setdefault("VLLM_EXL3_FORCE_COMPRESSED", "0")
    rows = []
    failures = []
    print(
        f"codebook={codebook} mcg={int(args.mcg)} mul1={int(args.mul1)} "
        f"bitrates={bitrates} m={ms} device={args.device}"
    )
    for name, (k, n) in BEHEMOTH_TP4_SHAPES.items():
        for bitrate in bitrates:
            if name == "lm_head" and bitrate != 6:
                continue
            if name != "lm_head" and bitrate == 6:
                continue
            for m in ms:
                x, trellis, suh, svh = _payloads(k, n, bitrate, m, device)
                output_ws = torch.empty(
                    (x.shape[0], trellis.shape[1] * 16),
                    dtype=torch.float16,
                    device=device,
                )
                x_had_ws = torch.empty_like(x)
                os.environ["VLLM_EXL3_FORCE_COMPRESSED"] = "1"
                got = call_exl3_gemm(x, trellis, suh, svh, args.mcg, args.mul1)
                native = _native_compressed(
                    ext, x, trellis, suh, svh, args.mcg, args.mul1, output_ws, x_had_ws
                )
                wrap_err = (got.float() - native.float()).abs().max().item()
                if wrap_err > 0:
                    failures.append(
                        f"wrapper {name} K={bitrate} M={m} {codebook} wrap_err={wrap_err}"
                    )
                ref, w = _reconstruct_ref(
                    ext, x, trellis, suh, svh, bitrate, args.mcg, args.mul1
                )
                max_err = (got.float() - ref.float()).abs().max().item()
                ref_max = ref.float().abs().max().item()
                rel_err = max_err / max(ref_max, 1e-3)

                def run_plugin():
                    os.environ["VLLM_EXL3_FORCE_COMPRESSED"] = "1"
                    os.environ.pop("VLLM_EXL3_FORCE_RECONSTRUCT", None)
                    return call_exl3_gemm(x, trellis, suh, svh, args.mcg, args.mul1)

                def run_native_persistent():
                    return _native_compressed(
                        ext,
                        x,
                        trellis,
                        suh,
                        svh,
                        args.mcg,
                        args.mul1,
                        output_ws,
                        x_had_ws,
                    )

                def run_reconstruct():
                    os.environ["VLLM_EXL3_FORCE_RECONSTRUCT"] = "1"
                    os.environ["VLLM_EXL3_FORCE_COMPRESSED"] = "0"
                    return call_exl3_gemm(x, trellis, suh, svh, args.mcg, args.mul1)

                def run_cublas():
                    return torch.mm(x, w)

                if args.skip_timing:
                    t_plugin = t_native = t_recon = t_fp16 = float("nan")
                else:
                    t_plugin = _time_ms(run_plugin, args.warmup, args.iters)
                    t_native = _time_ms(run_native_persistent, args.warmup, args.iters)
                    os.environ.pop("VLLM_EXL3_FORCE_RECONSTRUCT", None)
                    t_recon = _time_ms(run_reconstruct, args.warmup, args.iters)
                    os.environ.pop("VLLM_EXL3_FORCE_RECONSTRUCT", None)
                    os.environ["VLLM_EXL3_FORCE_COMPRESSED"] = "1"
                    t_fp16 = _time_ms(run_cublas, args.warmup, args.iters)
                    gate = GATES.get(m)
                    if (
                        args.fail_on_gate
                        and gate is not None
                        and t_plugin > gate * t_recon
                    ):
                        failures.append(
                            f"timing {name} K={bitrate} M={m} {codebook} "
                            f"plugin={t_plugin:.3f}ms recon={t_recon:.3f}ms "
                            f"limit={gate}x"
                        )
                row = {
                    "name": name,
                    "k": k,
                    "n": n,
                    "bitrate": bitrate,
                    "m": m,
                    "mcg": bool(args.mcg),
                    "mul1": bool(args.mul1),
                    "codebook": codebook,
                    "exl3_ms": t_plugin,
                    "exl3_plugin_ms": t_plugin,
                    "exl3_native_persistent_ms": t_native,
                    "reconstruct_ms": t_recon,
                    "fp16_cublas_ms": t_fp16,
                    "max_err": max_err,
                    "ref_max": ref_max,
                    "rel_err": rel_err,
                    "wrap_err": wrap_err,
                    "parity_ok": wrap_err == 0,
                }
                rows.append(row)
                print(
                    f"{name:10} K={bitrate} M={m:4d} {codebook:5}  "
                    f"plugin={t_plugin:8.3f}ms  native={t_native:8.3f}ms  "
                    f"recon={t_recon:8.3f}ms  fp16={t_fp16:8.3f}ms  "
                    f"err={max_err:.3e} wrap={wrap_err:.3e}"
                )

    budget = _decode_budget(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(
            {
                "codebook": codebook,
                "mcg": bool(args.mcg),
                "mul1": bool(args.mul1),
                "rows": rows,
                "failures": failures,
                "decode_budget": budget,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {args.output}")
    if budget:
        print(json.dumps(budget, indent=2))
    if failures:
        print("GATES FAILED:", *failures, sep="\n  ", file=sys.stderr)
        return 1 if args.fail_on_gate else 0
    return 0


def _decode_budget(rows: list[dict]) -> dict[str, object]:
    """Kernel-only M=1 estimate: 88 layers * 7 projections, plus one lm_head."""
    m1 = [r for r in rows if r["m"] == 1 and r["exl3_plugin_ms"] == r["exl3_plugin_ms"]]
    if not m1:
        return {}
    by_key = {(r["name"], r["bitrate"]): r for r in m1}
    estimates = []
    bitrates = sorted({r["bitrate"] for r in m1 if r["name"] != "lm_head"})
    for bitrate in bitrates:
        decoder_ms = 0.0
        missing = []
        for name in DECODER_SHAPES:
            row = by_key.get((name, bitrate))
            if row is None:
                missing.append(name)
                continue
            decoder_ms += float(row["exl3_plugin_ms"])
        if missing:
            continue
        token_ms = decoder_ms * BEHEMOTH_LAYERS
        head = by_key.get(("lm_head", 6))
        if head is not None:
            token_ms += float(head["exl3_plugin_ms"])
        native_ms = 0.0
        for name in DECODER_SHAPES:
            native_ms += float(by_key[(name, bitrate)]["exl3_native_persistent_ms"])
        native_token_ms = native_ms * BEHEMOTH_LAYERS
        if head is not None:
            native_token_ms += float(head["exl3_native_persistent_ms"])
        estimates.append(
            {
                "uniform_bitrate": bitrate,
                "plugin_ms_per_token": token_ms,
                "plugin_tok_s_ceiling": 1000.0 / token_ms if token_ms else None,
                "native_persistent_ms_per_token": native_token_ms,
                "native_tok_s_ceiling": 1000.0 / native_token_ms
                if native_token_ms
                else None,
                "target_28_tok_s_ms": 1000.0 / 28.0,
                "gap_to_28_tok_s_ms": token_ms - (1000.0 / 28.0),
            }
        )
    return {"uniform_bitrate_estimates": estimates}


if __name__ == "__main__":
    raise SystemExit(main())
