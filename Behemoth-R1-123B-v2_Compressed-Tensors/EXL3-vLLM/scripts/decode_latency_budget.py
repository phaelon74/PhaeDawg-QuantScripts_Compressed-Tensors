#!/usr/bin/env python3
"""Combine 4.25 inventory counts with M=1 microbench times into a token budget."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugin" / "src"))

from vllm_exl3_sm86.constants import BEHEMOTH_LAYERS  # noqa: E402

DECODER_LEAVES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--microbench", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument(
        "--output",
        default=str(ROOT / "results" / "decode_latency_budget.json"),
    )
    parser.add_argument("--target-tok-s", type=float, default=28.0)
    args = parser.parse_args()
    bench = json.loads(Path(args.microbench).read_text(encoding="utf-8"))
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    rows = [r for r in bench.get("rows", []) if r.get("m") == 1]
    by_key = {(r["name"], int(r["bitrate"])): r for r in rows}
    leaf_counts = inventory.get("decoder_leaf_bitrates") or {}
    plugin_ms = 0.0
    native_ms = 0.0
    missing = []
    used = []
    for leaf in DECODER_LEAVES:
        counts = {int(k): int(v) for k, v in (leaf_counts.get(leaf) or {}).items()}
        if not counts:
            missing.append(f"inventory missing {leaf}")
            continue
        for bitrate, n_layers in counts.items():
            row = by_key.get((leaf, bitrate))
            if row is None:
                missing.append(f"microbench missing {leaf} K={bitrate} M=1")
                continue
            plugin = float(row["exl3_plugin_ms"]) * n_layers
            native = float(row["exl3_native_persistent_ms"]) * n_layers
            plugin_ms += plugin
            native_ms += native
            used.append(
                {
                    "name": leaf,
                    "bitrate": bitrate,
                    "layers": n_layers,
                    "plugin_ms": plugin,
                    "native_persistent_ms": native,
                }
            )
    head = by_key.get(("lm_head", 6))
    head_plugin = float(head["exl3_plugin_ms"]) if head else 0.0
    head_native = float(head["exl3_native_persistent_ms"]) if head else 0.0
    plugin_ms += head_plugin
    native_ms += head_native
    target_ms = 1000.0 / args.target_tok_s
    report = {
        "checkpoint": inventory.get("checkpoint"),
        "codebook": bench.get("codebook"),
        "decoder_linears": inventory.get("decoder_linears"),
        "behemoth_layers": BEHEMOTH_LAYERS,
        "plugin_ms_per_token": plugin_ms,
        "plugin_tok_s_ceiling": 1000.0 / plugin_ms if plugin_ms else None,
        "native_persistent_ms_per_token": native_ms,
        "native_tok_s_ceiling": 1000.0 / native_ms if native_ms else None,
        "lm_head_plugin_ms": head_plugin,
        "target_tok_s": args.target_tok_s,
        "target_ms_per_token": target_ms,
        "plugin_gap_ms": plugin_ms - target_ms,
        "measured_serving_decode_tok_s": 18.9,
        "measured_serving_ms": 1000.0 / 18.9,
        "non_gemm_ms_if_plugin_budget": (1000.0 / 18.9) - plugin_ms,
        "rows": used,
        "missing": missing,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
