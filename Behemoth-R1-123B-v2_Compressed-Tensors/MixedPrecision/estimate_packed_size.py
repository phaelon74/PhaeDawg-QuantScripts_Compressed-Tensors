"""
Estimate packed on-disk size for a Mistral-Large-class W4A16 / W8A16 mix.

Reads config.json only (does not load 123B weights). Matches `du -sh` units
(GiB = 1024**3). GS32 scale overhead is large; GS128 leaves room for W8.

Usage:
  python estimate_packed_size.py /path/to/bf16_or_quant_dir
  python estimate_packed_size.py /path/to/model --group-size 128 \\
      --promote-down-proj-layers 0,1,87 --max-disk-gib 70
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Iterable


GIB = 1024**3
BYTES_PER_SCALE = 2  # FP16/BF16 group scale
BYTES_PER_BF16 = 2


@dataclass
class LinearModule:
    name: str
    n_params: int
    kind: str  # q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj


@dataclass
class Inventory:
    linears: list[LinearModule]
    embed_params: int
    lm_head_params: int
    tied_embeddings: bool
    config_path: str
    extras: dict = field(default_factory=dict)


def load_config(model_dir: str) -> dict:
    path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"config.json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_inventory(model_dir: str) -> Inventory:
    cfg = load_config(model_dir)
    hidden = int(cfg["hidden_size"])
    intermediate = int(cfg["intermediate_size"])
    n_layers = int(cfg["num_hidden_layers"])
    n_heads = int(cfg["num_attention_heads"])
    n_kv = int(cfg.get("num_key_value_heads", n_heads))
    vocab = int(cfg["vocab_size"])
    head_dim = int(cfg.get("head_dim", hidden // n_heads))
    tied = bool(cfg.get("tie_word_embeddings", False))

    q_out = n_heads * head_dim
    kv_out = n_kv * head_dim
    shapes = {
        "q_proj": (hidden, q_out),
        "k_proj": (hidden, kv_out),
        "v_proj": (hidden, kv_out),
        "o_proj": (q_out, hidden),
        "gate_proj": (hidden, intermediate),
        "up_proj": (hidden, intermediate),
        "down_proj": (intermediate, hidden),
    }

    linears: list[LinearModule] = []
    for layer in range(n_layers):
        for kind, (inn, out) in shapes.items():
            if kind in ("q_proj", "k_proj", "v_proj", "o_proj"):
                name = f"model.layers.{layer}.self_attn.{kind}"
            else:
                name = f"model.layers.{layer}.mlp.{kind}"
            linears.append(LinearModule(name=name, n_params=inn * out, kind=kind))

    embed = vocab * hidden
    lm_head = 0 if tied else embed
    return Inventory(
        linears=linears,
        embed_params=embed,
        lm_head_params=lm_head,
        tied_embeddings=tied,
        config_path=os.path.join(model_dir, "config.json"),
        extras={
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_hidden_layers": n_layers,
            "num_attention_heads": n_heads,
            "num_key_value_heads": n_kv,
            "head_dim": head_dim,
            "vocab_size": vocab,
        },
    )


def _compile_patterns(patterns: Iterable[str]) -> list[re.Pattern]:
    compiled = []
    for raw in patterns:
        text = raw.strip()
        if not text:
            continue
        if text.startswith("re:"):
            text = text[3:]
        compiled.append(re.compile(text))
    return compiled


def assign_bits(
    inv: Inventory,
    w8_regexes: list[str],
    promote_down_proj_layers: list[int],
    bf16_regexes: list[str],
) -> dict[str, int]:
    """Return name -> bit width for every Linear. Default 4. lm_head/embed stay BF16."""
    bits = {m.name: 4 for m in inv.linears}
    w8_re = _compile_patterns(w8_regexes)
    bf16_re = _compile_patterns(bf16_regexes)
    promote = set(promote_down_proj_layers)

    for m in inv.linears:
        if any(p.search(m.name) for p in bf16_re):
            bits[m.name] = 16
            continue
        layer_idx = None
        if ".layers." in m.name:
            try:
                layer_idx = int(m.name.split(".layers.")[1].split(".")[0])
            except (IndexError, ValueError):
                layer_idx = None
        if m.kind == "down_proj" and layer_idx in promote:
            bits[m.name] = 8
            continue
        if any(p.search(m.name) for p in w8_re):
            bits[m.name] = 8
    return bits


def packed_linear_bytes(n_params: int, bits: int, group_size: int, symmetric: bool) -> int:
    if bits == 16:
        return n_params * BYTES_PER_BF16
    weight_bytes = n_params * bits // 8
    n_groups = (n_params + group_size - 1) // group_size
    scale_bytes = n_groups * BYTES_PER_SCALE
    # Packed INT zero-points: 4 bits per group, ignored when symmetric.
    zp_bytes = 0 if symmetric else (n_groups + 1) // 2
    return weight_bytes + scale_bytes + zp_bytes


def estimate(
    inv: Inventory,
    bits_map: dict[str, int],
    group_size: int,
    symmetric: bool,
    sidecar_bytes: int = 64 * 1024 * 1024,
) -> dict:
    by_bits = {4: 0, 8: 0, 16: 0}
    payload = 0
    promoted = []
    for m in inv.linears:
        b = bits_map[m.name]
        by_bits[b] += m.n_params
        payload += packed_linear_bytes(m.n_params, b, group_size, symmetric)
        if b != 4:
            promoted.append((m.name, b, m.n_params))

    unquant = (inv.embed_params + inv.lm_head_params) * BYTES_PER_BF16
    total = payload + unquant + sidecar_bytes
    w4_only = sum(
        packed_linear_bytes(m.n_params, 4, group_size, symmetric) for m in inv.linears
    )
    w4_only += unquant + sidecar_bytes

    rows = []
    for name, b, n_params in promoted:
        extra = packed_linear_bytes(n_params, b, group_size, symmetric) - packed_linear_bytes(
            n_params, 4, group_size, symmetric
        )
        rows.append({"name": name, "bits": b, "params": n_params, "extra_bytes": extra})

    return {
        "linear_params": sum(m.n_params for m in inv.linears),
        "params_w4": by_bits[4],
        "params_w8": by_bits[8],
        "params_bf16": by_bits[16],
        "packed_linear_bytes": payload,
        "unquantized_bytes": unquant,
        "sidecar_bytes": sidecar_bytes,
        "total_bytes": total,
        "uniform_w4_bytes": w4_only,
        "extra_vs_uniform_w4_bytes": total - w4_only,
        "promotions": rows,
        "group_size": group_size,
        "symmetric": symmetric,
    }


def fmt_gib(n_bytes: int) -> str:
    return f"{n_bytes / GIB:.2f} GiB"


def parse_layer_list(raw: str | None) -> list[int]:
    if not raw:
        return []
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate W4/W8 packed checkpoint size.")
    parser.add_argument("model_dir", help="Directory containing config.json.")
    parser.add_argument("--group-size", type=int, default=32, choices=(32, 64, 128))
    parser.add_argument(
        "--asymmetric",
        action="store_true",
        help="Include packed zero-point overhead (INT4 W4A16_ASYM).",
    )
    parser.add_argument(
        "--w8-regex",
        action="append",
        default=[],
        help="Regex (re: optional prefix) for modules to keep at W8A16. Repeatable.",
    )
    parser.add_argument(
        "--bf16-regex",
        action="append",
        default=[],
        help="Regex for Linear modules left in BF16. Repeatable.",
    )
    parser.add_argument(
        "--promote-down-proj-layers",
        type=str,
        default="",
        help="Comma-separated layer indices whose mlp.down_proj is W8.",
    )
    parser.add_argument("--max-disk-gib", type=float, default=70.0)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a table.",
    )
    args = parser.parse_args()

    inv = build_inventory(args.model_dir)
    bits_map = assign_bits(
        inv,
        w8_regexes=args.w8_regex,
        promote_down_proj_layers=parse_layer_list(args.promote_down_proj_layers),
        bf16_regexes=args.bf16_regex,
    )
    result = estimate(
        inv,
        bits_map,
        group_size=args.group_size,
        symmetric=not args.asymmetric,
    )

    if args.json:
        out = {
            **inv.extras,
            **{k: v for k, v in result.items() if k != "promotions"},
            "promotions": result["promotions"],
            "total_gib": result["total_bytes"] / GIB,
            "under_budget": (result["total_bytes"] / GIB) <= args.max_disk_gib,
        }
        json.dump(out, sys.stdout, indent=2)
        print()
        return 0 if out["under_budget"] else 2

    print(f"config: {inv.config_path}")
    for k, v in inv.extras.items():
        print(f"  {k}: {v}")
    print(f"  tied_embeddings: {inv.tied_embeddings}")
    print()
    print(f"group_size={args.group_size}  symmetric={not args.asymmetric}")
    print(
        f"Linear params: {result['linear_params']:,}  "
        f"(W4 {result['params_w4']:,} / W8 {result['params_w8']:,} / BF16 {result['params_bf16']:,})"
    )
    print(f"Packed Linears:     {fmt_gib(result['packed_linear_bytes'])}")
    print(f"embed + lm_head:    {fmt_gib(result['unquantized_bytes'])} (BF16)")
    print(f"sidecar allowance:  {fmt_gib(result['sidecar_bytes'])}")
    print(f"Estimated total:    {fmt_gib(result['total_bytes'])}")
    print(f"Uniform W4 same GS: {fmt_gib(result['uniform_w4_bytes'])}")
    print(f"Extra vs uniform W4:{fmt_gib(result['extra_vs_uniform_w4_bytes'])}")
    print(f"Budget:             {args.max_disk_gib:.1f} GiB")

    if result["promotions"]:
        print()
        print("Promotions (cost vs W4):")
        result["promotions"].sort(key=lambda r: -r["extra_bytes"])
        for row in result["promotions"][:40]:
            print(
                f"  W{row['bits']:<2}  +{row['extra_bytes'] / GIB:.3f} GiB  "
                f"{row['params']:,}  {row['name']}"
            )
        if len(result["promotions"]) > 40:
            print(f"  ... {len(result['promotions']) - 40} more")

    over = result["total_bytes"] / GIB > args.max_disk_gib
    print()
    if over:
        print("REJECT: estimated size exceeds --max-disk-gib")
        return 2
    print("OK: estimated size is under budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
