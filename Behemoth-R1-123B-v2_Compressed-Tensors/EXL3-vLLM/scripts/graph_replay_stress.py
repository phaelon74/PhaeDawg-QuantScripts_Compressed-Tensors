#!/usr/bin/env python3
"""CUDA-graph capture/replay stress for EXL3 decode sizes 1/2/4.

Default codebook is implicit 3inst. Pass --mul1 for the INT8 diagnostic path.
Use --all-shapes to cover every TP4 projection, not only q_proj.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugin" / "src"))

from vllm_exl3_sm86.constants import (  # noqa: E402
    BEHEMOTH_TP4_SHAPES,
    GRAPH_CAPTURE_SIZES,
)
from vllm_exl3_sm86.ops import call_exl3_gemm, call_exl3_mgemm, ext_has_mgemm  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--bitrate", type=int, default=4)
    parser.add_argument("--mcg", action="store_true")
    parser.add_argument("--mul1", action="store_true")
    parser.add_argument(
        "--all-shapes",
        action="store_true",
        help="Capture q/k/v/o/gate/up/down/lm_head instead of q_proj only.",
    )
    parser.add_argument("--shape", default="q_proj")
    parser.add_argument(
        "--mgemm",
        action="store_true",
        help="Also capture fused gate/up exl3_mgemm graphs (K=4, 3inst).",
    )
    args = parser.parse_args()
    if args.mcg and args.mul1:
        print("mcg and mul1 are mutually exclusive", file=sys.stderr)
        return 2
    import torch

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 2
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    names = list(BEHEMOTH_TP4_SHAPES) if args.all_shapes else [args.shape]
    graphs = []
    # CUDA graphs record device pointers. Keep every captured tensor alive.
    payloads: list[tuple[torch.Tensor, ...]] = []
    for name in names:
        k, n = BEHEMOTH_TP4_SHAPES[name]
        bitrate = 6 if name == "lm_head" else args.bitrate
        for m in GRAPH_CAPTURE_SIZES:
            x = torch.zeros((m, k), dtype=torch.float16, device=device)
            trellis = torch.zeros(
                (k // 16, n // 16, 16 * bitrate), dtype=torch.int16, device=device
            )
            suh = torch.ones(k, dtype=torch.float16, device=device)
            svh = torch.ones(n, dtype=torch.float16, device=device)
            out = torch.empty((m, n), dtype=torch.float16, device=device)
            x_had = torch.empty((m, k), dtype=torch.float16, device=device)
            call_exl3_gemm(
                x, trellis, suh, svh, args.mcg, args.mul1, out=out, x_had=x_had
            )
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                with torch.cuda.graph(g):
                    call_exl3_gemm(
                        x, trellis, suh, svh, args.mcg, args.mul1, out=out, x_had=x_had
                    )
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            graphs.append(g)
            payloads.append((x, trellis, suh, svh, out, x_had))
            print(f"captured {name} K={bitrate} M={m} mcg={int(args.mcg)} mul1={int(args.mul1)}")
    if args.mgemm:
        if not ext_has_mgemm():
            print("exllamav3_ext does not export exl3_mgemm", file=sys.stderr)
            return 2
        k, n = BEHEMOTH_TP4_SHAPES["gate_proj"]
        bitrate = 4
        for m in GRAPH_CAPTURE_SIZES:
            x = torch.zeros((m, k), dtype=torch.float16, device=device)
            trellis0 = torch.zeros(
                (k // 16, n // 16, 16 * bitrate), dtype=torch.int16, device=device
            )
            trellis1 = torch.zeros(
                (k // 16, n // 16, 16 * bitrate), dtype=torch.int16, device=device
            )
            suh0 = torch.ones(k, dtype=torch.float16, device=device)
            suh1 = torch.ones(k, dtype=torch.float16, device=device)
            svh0 = torch.ones(n, dtype=torch.float16, device=device)
            svh1 = torch.ones(n, dtype=torch.float16, device=device)
            ptrs_trellis = torch.tensor(
                [int(trellis0.data_ptr()), int(trellis1.data_ptr())],
                dtype=torch.int64,
                device=device,
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
            out = torch.empty((2, m, n), dtype=torch.float16, device=device)
            x_had = torch.empty((2, m, k), dtype=torch.float16, device=device)
            x3 = x.view(1, m, k)
            call_exl3_mgemm(
                x3,
                ptrs_trellis,
                ptrs_suh,
                ptrs_svh,
                bitrate,
                args.mcg,
                args.mul1,
                out,
                x_had,
            )
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                with torch.cuda.graph(g):
                    call_exl3_mgemm(
                        x3,
                        ptrs_trellis,
                        ptrs_suh,
                        ptrs_svh,
                        bitrate,
                        args.mcg,
                        args.mul1,
                        out,
                        x_had,
                    )
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            graphs.append(g)
            payloads.append(
                (
                    x,
                    trellis0,
                    trellis1,
                    suh0,
                    suh1,
                    svh0,
                    svh1,
                    ptrs_trellis,
                    ptrs_suh,
                    ptrs_svh,
                    out,
                    x_had,
                    x3,
                )
            )
            print(f"captured gate_up mgemm K={bitrate} M={m}")
    for step in range(args.steps):
        graphs[step % len(graphs)].replay()
        if step % 10000 == 0:
            torch.cuda.synchronize()
            print(f"replay {step}/{args.steps}")
    torch.cuda.synchronize()
    print(f"{args.steps} graph replay OK across {len(graphs)} graphs")
    del payloads
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
