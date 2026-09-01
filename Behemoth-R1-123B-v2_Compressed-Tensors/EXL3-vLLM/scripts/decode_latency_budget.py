#!/usr/bin/env python3
"""Combine mixed-K inventory counts with M=1 microbench times into a token budget.

K5/K6 decoder leaves fall off the Ampere GEMV path onto the regular kernel.
This script weights measured M=1 times by the checkpoint's per-leaf bitrate
counts and reports weights/second plus that fallback cost.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugin" / "src"))

from vllm_exl3_sm86.constants import (  # noqa: E402
    BEHEMOTH_LAYERS,
    BEHEMOTH_TP4_SHAPES,
    TARGET_DECODE_TOK_S,
)

DECODER_LEAVES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

NARROW_GEMV_LEAVES = frozenset(
    {"o_proj", "gate_proj", "up_proj", "down_proj"}
)


def kernel_path(name: str, bitrate: int) -> str:
    """Current SM86 overlay: K=4 M=1 uses narrow GEMV on o/gate/up/down."""
    if name == "lm_head":
        return "regular"
    if bitrate != 4:
        return "regular"
    if name in NARROW_GEMV_LEAVES:
        return "gemv_narrow"
    return "regular"


def _gwps(weights: float, ms: float) -> float | None:
    if ms <= 0:
        return None
    return (weights / (ms * 1e-3)) / 1e9


def fusion_from_layers(layers: list[dict]) -> dict[str, object]:
    kv = 0
    qkv = 0
    gate_up = 0
    kv_by: dict[int, int] = {}
    qkv_by: dict[int, int] = {}
    for row in layers:
        k_b = row.get("k_proj")
        v_b = row.get("v_proj")
        q_b = row.get("q_proj")
        if k_b is not None and k_b == v_b:
            kv += 1
            kv_by[int(k_b)] = kv_by.get(int(k_b), 0) + 1
        if q_b is not None and k_b == v_b == q_b:
            qkv += 1
            qkv_by[int(q_b)] = qkv_by.get(int(q_b), 0) + 1
        if row.get("gate_proj") == row.get("up_proj") and row.get("gate_proj") is not None:
            gate_up += 1
    return {
        "source": "per_layer",
        "layers": len(layers),
        "kv_fuse_layers": kv,
        "qkv_fuse_layers": qkv,
        "kv_mismatch_layers": len(layers) - kv,
        "gate_up_fuse_layers": gate_up,
        "kv_fuse_by_bitrate": {str(k): v for k, v in sorted(kv_by.items())},
        "qkv_fuse_by_bitrate": {str(k): v for k, v in sorted(qkv_by.items())},
        "note": "exact pairing from checkpoint prefixes; mgemm still needs one scalar bitrate",
    }


def fusion_bounds_from_counts(
    leaf_counts: dict, n_layers: int
) -> dict[str, object]:
    def counts(leaf: str) -> dict[int, int]:
        return {int(k): int(v) for k, v in (leaf_counts.get(leaf) or {}).items()}

    k_c, v_c, q_c = counts("k_proj"), counts("v_proj"), counts("q_proj")
    kv_max = 0
    kv_min = 0
    qkv_max = 0
    kv_by: dict[int, int] = {}
    bitrates = set(k_c) | set(v_c) | set(q_c)
    for br in bitrates:
        ck, cv, cq = k_c.get(br, 0), v_c.get(br, 0), q_c.get(br, 0)
        kv_max += min(ck, cv)
        kv_min += max(0, ck + cv - n_layers)
        qkv_max += min(cq, ck, cv)
        if min(ck, cv):
            kv_by[br] = min(ck, cv)
    gate = counts("gate_proj")
    up = counts("up_proj")
    gate_up = sum(min(gate.get(br, 0), up.get(br, 0)) for br in set(gate) | set(up))
    return {
        "source": "marginal_counts",
        "layers": n_layers,
        "kv_fuse_layers_max": kv_max,
        "kv_fuse_layers_min": kv_min,
        "qkv_fuse_layers_max": qkv_max,
        "gate_up_fuse_layers_max": gate_up,
        "kv_fuse_by_bitrate_max": {str(k): v for k, v in sorted(kv_by.items())},
        "note": "bounds from leaf counts only; re-run inventory for exact per-layer pairing",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--microbench",
        nargs="+",
        required=True,
        help="One or more kernel_microbench JSON files; rows are merged.",
    )
    parser.add_argument("--inventory", required=True)
    parser.add_argument(
        "--output",
        default=str(ROOT / "results" / "decode_latency_budget.json"),
    )
    parser.add_argument("--target-tok-s", type=float, default=TARGET_DECODE_TOK_S)
    parser.add_argument(
        "--serving-tok-s",
        type=float,
        default=0.0,
        help="Measured serving decode rate for non-GEMM budget accounting.",
    )
    args = parser.parse_args()
    rows = []
    codebook = None
    for path in args.microbench:
        bench = json.loads(Path(path).read_text(encoding="utf-8"))
        codebook = codebook or bench.get("codebook")
        rows.extend(r for r in bench.get("rows", []) if r.get("m") == 1)
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    by_key = {(r["name"], int(r["bitrate"])): r for r in rows}
    leaf_counts = inventory.get("decoder_leaf_bitrates") or {}
    plugin_ms = 0.0
    native_ms = 0.0
    weights = 0.0
    missing = []
    used = []
    path_ms: dict[str, float] = {}
    bitrate_ms: dict[int, float] = {}
    k56_native_ms = 0.0
    k56_counterfactual_ms = 0.0
    k56_layers = 0
    for leaf in DECODER_LEAVES:
        counts = {int(k): int(v) for k, v in (leaf_counts.get(leaf) or {}).items()}
        if not counts:
            missing.append(f"inventory missing {leaf}")
            continue
        k, n = BEHEMOTH_TP4_SHAPES[leaf]
        k4_row = by_key.get((leaf, 4))
        for bitrate, n_layers in sorted(counts.items()):
            row = by_key.get((leaf, bitrate))
            if row is None:
                missing.append(f"microbench missing {leaf} K={bitrate} M=1")
                continue
            plugin = float(row["exl3_plugin_ms"]) * n_layers
            native = float(row["exl3_native_persistent_ms"]) * n_layers
            leaf_weights = float(k) * float(n) * n_layers
            plugin_ms += plugin
            native_ms += native
            weights += leaf_weights
            path = kernel_path(leaf, bitrate)
            path_ms[path] = path_ms.get(path, 0.0) + native
            bitrate_ms[bitrate] = bitrate_ms.get(bitrate, 0.0) + native
            extra = None
            if bitrate > 4 and k4_row is not None:
                k56_layers += n_layers
                k56_native_ms += native
                cf = float(k4_row["exl3_native_persistent_ms"]) * n_layers
                k56_counterfactual_ms += cf
                extra = native - cf
            used.append(
                {
                    "name": leaf,
                    "k": k,
                    "n": n,
                    "bitrate": bitrate,
                    "layers": n_layers,
                    "kernel_path": path,
                    "plugin_ms": plugin,
                    "native_persistent_ms": native,
                    "weights": leaf_weights,
                    "native_gwps": _gwps(leaf_weights, native),
                    "k56_extra_vs_k4_ms": extra,
                }
            )
    head_meta = inventory.get("head") or {}
    head_bitrate = int(head_meta.get("bitrate") or 6)
    head = by_key.get(("lm_head", head_bitrate))
    head_plugin = float(head["exl3_plugin_ms"]) if head else 0.0
    head_native = float(head["exl3_native_persistent_ms"]) if head else 0.0
    head_k, head_n = BEHEMOTH_TP4_SHAPES["lm_head"]
    head_weights = float(head_k) * float(head_n) if head else 0.0
    warnings = []
    if head is None:
        warnings.append(f"microbench missing lm_head K={head_bitrate} M=1")
    else:
        plugin_ms += head_plugin
        native_ms += head_native
        weights += head_weights
        used.append(
            {
                "name": "lm_head",
                "k": head_k,
                "n": head_n,
                "bitrate": head_bitrate,
                "layers": 1,
                "kernel_path": kernel_path("lm_head", head_bitrate),
                "plugin_ms": head_plugin,
                "native_persistent_ms": head_native,
                "weights": head_weights,
                "native_gwps": _gwps(head_weights, head_native),
                "k56_extra_vs_k4_ms": None,
            }
        )
    target_ms = 1000.0 / args.target_tok_s
    serving_ms = (
        1000.0 / args.serving_tok_s if args.serving_tok_s > 0 else None
    )
    native_non_gemm_ms = (
        serving_ms - native_ms if serving_ms is not None else None
    )
    required_native_ms = (
        target_ms - native_non_gemm_ms
        if native_non_gemm_ms is not None
        else None
    )
    native_reduction_ms = (
        native_ms - required_native_ms
        if required_native_ms is not None
        else None
    )
    k56_saved = k56_native_ms - k56_counterfactual_ms
    report = {
        "checkpoint": inventory.get("checkpoint"),
        "codebook": codebook,
        "microbench_files": list(args.microbench),
        "decoder_linears": inventory.get("decoder_linears"),
        "behemoth_layers": BEHEMOTH_LAYERS,
        "plugin_ms_per_token": plugin_ms,
        "plugin_tok_s_ceiling": 1000.0 / plugin_ms if plugin_ms else None,
        "plugin_tok_s_note": (
            "includes vLLM custom-op dispatch; CUDA graphs erase this in serving. "
            "Use native_tok_s_ceiling."
        ),
        "native_persistent_ms_per_token": native_ms,
        "native_tok_s_ceiling": 1000.0 / native_ms if native_ms else None,
        "native_gwps": _gwps(weights, native_ms),
        "weights_per_token": weights,
        "lm_head_plugin_ms": head_plugin,
        "lm_head_included": head is not None,
        "target_tok_s": args.target_tok_s,
        "target_ms_per_token": target_ms,
        "plugin_gap_ms": plugin_ms - target_ms,
        "measured_serving_decode_tok_s": args.serving_tok_s or None,
        "measured_serving_ms": serving_ms,
        "non_gemm_ms_if_plugin_budget": (
            serving_ms - plugin_ms if serving_ms is not None else None
        ),
        "non_gemm_ms_if_native_budget": native_non_gemm_ms,
        "required_native_ms_at_target": required_native_ms,
        "required_native_reduction_ms": native_reduction_ms,
        "required_native_reduction_pct": (
            100.0 * native_reduction_ms / native_ms
            if native_reduction_ms is not None and native_ms
            else None
        ),
        "native_ms_by_kernel_path": path_ms,
        "native_ms_by_bitrate": {str(k): v for k, v in sorted(bitrate_ms.items())},
        "k56_regular_fallback": {
            "layers": k56_layers,
            "native_ms": k56_native_ms,
            "if_same_as_k4_ms": k56_counterfactual_ms,
            "extra_ms": k56_saved,
            "tok_s_if_k4_times": (
                1000.0 / (native_ms - k56_saved)
                if k56_saved and native_ms > k56_saved
                else None
            ),
        },
        "mgemm_fusion": (
            fusion_from_layers(inventory["decoder_layer_bitrates"])
            if inventory.get("decoder_layer_bitrates")
            else fusion_bounds_from_counts(
                leaf_counts, int(inventory.get("decoder_linears") or 0) // 7 or BEHEMOTH_LAYERS
            )
        ),
        "rows": used,
        "missing": missing,
        "warnings": warnings,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
