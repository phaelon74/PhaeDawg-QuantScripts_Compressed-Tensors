#!/usr/bin/env python3
"""100K CUDA-graph capture/replay stress for EXL3 decode sizes 1/2/4."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugin" / "src"))

from vllm_exl3_sm86.constants import BEHEMOTH_TP4_SHAPES, GRAPH_CAPTURE_SIZES  # noqa: E402
from vllm_exl3_sm86.ops import call_exl3_gemm  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--bitrate", type=int, default=3)
    args = parser.parse_args()
    import torch

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 2
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    k, n = BEHEMOTH_TP4_SHAPES["q_proj"]
    graphs = []
    payloads = []
    for m in GRAPH_CAPTURE_SIZES:
        x = torch.zeros((m, k), dtype=torch.float16, device=device)
        trellis = torch.zeros((k // 16, n // 16, 16 * args.bitrate), dtype=torch.int16, device=device)
        suh = torch.ones(k, dtype=torch.float16, device=device)
        svh = torch.ones(n, dtype=torch.float16, device=device)
        # Warm autotune before capture.
        call_exl3_gemm(x, trellis, suh, svh, False, True)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.cuda.graph(g):
                call_exl3_gemm(x, trellis, suh, svh, False, True)
        torch.cuda.current_stream().wait_stream(s)
        graphs.append(g)
        payloads.append((m, x))
    for step in range(args.steps):
        graphs[step % len(graphs)].replay()
        if step % 10000 == 0:
            torch.cuda.synchronize()
            print(f"replay {step}/{args.steps}")
    torch.cuda.synchronize()
    print("100K graph replay OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
