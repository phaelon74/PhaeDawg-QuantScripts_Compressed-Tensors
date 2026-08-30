#!/usr/bin/env python3
"""Prewarm EXL3 kernels for CUDA-graph capture sizes 1/2/4."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugin" / "src"))

from vllm_exl3_sm86.graph import prewarm_behemoth_tp4  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", default=str(ROOT / "results" / "prewarm.json"))
    parser.add_argument(
        "--codebook",
        choices=("3inst", "mul1", "both"),
        default="both",
        help="3inst is ArtusDev 4.25. mul1 is the 4.5 INT8 diagnostic. both is default.",
    )
    args = parser.parse_args()
    import torch

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 2
    codebooks = {
        "3inst": ((False, False),),
        "mul1": ((False, True),),
        "both": ((False, False), (False, True)),
    }[args.codebook]
    receipts = prewarm_behemoth_tp4(
        torch.device("cuda", args.device),
        codebooks=codebooks,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(receipts, indent=2) + "\n")
    print(f"prewarmed {len(receipts)} shapes -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
