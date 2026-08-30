#!/usr/bin/env python3
"""M=1/2/4 gate+up: two exl3_gemm vs one exl3_mgemm (ArtusDev 4.25 3inst K=4).

Run on a free GPU after stopping TP4 serve. Physical 0 and 5 stay reserved:
  CUDA_VISIBLE_DEVICES=1 python "$EXL3/scripts/mgemm_microbench.py"
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugin" / "src"))

from vllm_exl3_sm86.constants import (  # noqa: E402
    BEHEMOTH_LAYERS,
    BEHEMOTH_TP4_SHAPES,
    GRAPH_CAPTURE_SIZES,
)
from vllm_exl3_sm86.ops import (  # noqa: E402
    _load_exl3_ext,
    call_exl3_gemm,
    call_exl3_mgemm,
    ext_has_mgemm,
)

K, N = BEHEMOTH_TP4_SHAPES["gate_proj"]
BITRATE = 4


def _sync():
    import torch

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--output",
        default=str(ROOT / "results" / "mgemm_vs_gemm_gateup.json"),
    )
    args = parser.parse_args()
    import torch

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 2
    if not ext_has_mgemm():
        print("exllamav3_ext does not export exl3_mgemm", file=sys.stderr)
        return 2
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    _load_exl3_ext()
    rows = []
    print(f"gate_up K={K} N={N} bitrate={BITRATE} 3inst device={args.device}")
    for m in GRAPH_CAPTURE_SIZES:
        x = torch.randn(m, K, dtype=torch.float16, device=device)
        t0 = torch.randint(
            -1024, 1024, (K // 16, N // 16, 16 * BITRATE), dtype=torch.int16, device=device
        )
        t1 = torch.randint(
            -1024, 1024, (K // 16, N // 16, 16 * BITRATE), dtype=torch.int16, device=device
        )
        suh0 = torch.randn(K, dtype=torch.float16, device=device)
        suh1 = torch.randn(K, dtype=torch.float16, device=device)
        svh0 = torch.randn(N, dtype=torch.float16, device=device)
        svh1 = torch.randn(N, dtype=torch.float16, device=device)
        out0 = torch.empty((m, N), dtype=torch.float16, device=device)
        out1 = torch.empty((m, N), dtype=torch.float16, device=device)
        xhad0 = torch.empty((m, K), dtype=torch.float16, device=device)
        xhad1 = torch.empty((m, K), dtype=torch.float16, device=device)
        gout = torch.empty((2, m, N), dtype=torch.float16, device=device)
        gxhad = torch.empty((2, m, K), dtype=torch.float16, device=device)
        ptrs_t = torch.tensor(
            [int(t0.data_ptr()), int(t1.data_ptr())], dtype=torch.int64, device=device
        )
        ptrs_suh = torch.tensor(
            [int(suh0.data_ptr()), int(suh1.data_ptr())],
            dtype=torch.int64,
            device=device,
        )
        ptrs_svh = torch.tensor(
            [int(svh0.data_ptr()), int(svh1.data_ptr())],
            dtype=torch.int64,
            device=device,
        )

        def two_gemms():
            call_exl3_gemm(x, t0, suh0, svh0, False, False, out=out0, x_had=xhad0)
            call_exl3_gemm(x, t1, suh1, svh1, False, False, out=out1, x_had=xhad1)

        def one_mgemm():
            call_exl3_mgemm(
                x.view(1, m, K),
                ptrs_t,
                ptrs_suh,
                ptrs_svh,
                BITRATE,
                False,
                False,
                gout,
                gxhad,
            )

        two_gemms()
        one_mgemm()
        _sync()
        t_gemm = _time_ms(two_gemms, args.warmup, args.iters)
        t_mgemm = _time_ms(one_mgemm, args.warmup, args.iters)
        layer_ms = {
            "m": m,
            "two_gemm_ms": t_gemm,
            "mgemm_ms": t_mgemm,
            "speedup": t_gemm / t_mgemm if t_mgemm else None,
            "two_gemm_88_ms": t_gemm * BEHEMOTH_LAYERS,
            "mgemm_88_ms": t_mgemm * BEHEMOTH_LAYERS,
        }
        rows.append(layer_ms)
        print(
            f"M={m}  two_gemm={t_gemm:.3f}ms  mgemm={t_mgemm:.3f}ms  "
            f"speedup={layer_ms['speedup']:.3f}x  "
            f"88x two={layer_ms['two_gemm_88_ms']:.1f}ms  "
            f"88x mgemm={layer_ms['mgemm_88_ms']:.1f}ms"
        )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
