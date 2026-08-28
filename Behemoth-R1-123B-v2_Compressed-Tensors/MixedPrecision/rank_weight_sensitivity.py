"""
Stream BF16 safetensors and rank W4 -> W8 promotion units.

This is a calibration-free sensitivity proxy for selecting AutoRound mixed-bit
candidates. It measures symmetric grouped-QDQ reconstruction improvement without
loading the 123B model, then emits nested policies under requested size budgets.
Only down_proj is eligible by default; broader fused-safe units are opt-in.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from safetensors import safe_open

from estimate_packed_size import (
    GIB,
    build_inventory,
    estimate,
    packed_linear_bytes,
)

LINEAR_KINDS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def load_weight_map(model_dir: str) -> dict[str, str]:
    path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Safetensors index not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data["weight_map"]


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {requested}")
    return device


def grouped_qdq_errors(
    tensor_slice,
    bit_widths: tuple[int, ...],
    group_size: int,
    chunk_rows: int,
    device: torch.device,
) -> tuple[dict[int, float], float]:
    shape = tuple(tensor_slice.get_shape())
    if len(shape) != 2:
        raise ValueError(f"Expected a 2-D weight tensor, found shape {shape}")
    out_features, in_features = shape
    if in_features % group_size:
        raise ValueError(
            f"in_features={in_features} is not divisible by group_size={group_size}"
        )

    errors = {bits: 0.0 for bits in bit_widths}
    energy = 0.0
    for start in range(0, out_features, chunk_rows):
        stop = min(start + chunk_rows, out_features)
        weight = tensor_slice[start:stop].to(device=device, dtype=torch.float32)
        groups = weight.reshape(stop - start, -1, group_size)
        energy += float(torch.sum(groups.square()).item())
        group_min = groups.amin(dim=-1, keepdim=True).clamp_(max=0)
        group_max = groups.amax(dim=-1, keepdim=True).clamp_(min=0)
        for bits in bit_widths:
            maxq = 1 << (bits - 1)
            min_abs = -group_min
            signed_max = (
                2 * (group_max < min_abs).to(torch.int8) - 1
            ) * torch.maximum(group_max, min_abs)
            scale = (signed_max / maxq).to(torch.float16).to(torch.float32)
            scale = torch.where(
                scale < 0,
                scale.clamp(max=-1e-5),
                scale.clamp(min=1e-5),
            )
            quant = torch.round(groups / scale).clamp_(-maxq, maxq - 1)
            reconstructed = quant * scale
            errors[bits] += float(
                torch.sum((groups - reconstructed).square()).item()
            )
            del scale, quant, reconstructed
        del weight, groups, group_min, group_max
    return errors, energy


def module_layer(name: str) -> int:
    match = re.search(r"\.layers\.(\d+)\.", name)
    if match is None:
        raise ValueError(f"Cannot extract decoder-layer index from {name}")
    return int(match.group(1))


def promotion_unit(name: str, kind: str) -> str:
    layer = module_layer(name)
    if kind in {"q_proj", "k_proj", "v_proj"}:
        suffix = "self_attn.qkv"
    elif kind in {"gate_proj", "up_proj"}:
        suffix = "mlp.gate_up"
    elif kind == "o_proj":
        suffix = "self_attn.o_proj"
    elif kind == "down_proj":
        suffix = "mlp.down_proj"
    else:
        raise ValueError(f"Unsupported Linear kind: {kind}")
    return f"model.layers.{layer}.{suffix}"


def score_modules(
    model_dir: str,
    group_size: int,
    chunk_rows: int,
    device: torch.device,
    target_kinds: set[str],
) -> tuple[object, list[dict]]:
    inventory = build_inventory(model_dir)
    weight_map = load_weight_map(model_dir)
    scores = []
    targets = [module for module in inventory.linears if module.kind in target_kinds]
    total = len(targets)
    for index, module in enumerate(targets, start=1):
        key = f"{module.name}.weight"
        shard = weight_map.get(key)
        if shard is None:
            raise KeyError(f"Weight is missing from safetensors index: {key}")
        shard_path = os.path.join(model_dir, shard)
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            tensor_slice = handle.get_slice(key)
            errors, energy = grouped_qdq_errors(
                tensor_slice,
                bit_widths=(4, 8),
                group_size=group_size,
                chunk_rows=chunk_rows,
                device=device,
            )

        gain = max(errors[4] - errors[8], 0.0)
        relative_gain = gain / max(energy, torch.finfo(torch.float32).tiny)
        extra_bytes = packed_linear_bytes(
            module.n_params, 8, group_size, True
        ) - packed_linear_bytes(module.n_params, 4, group_size, True)
        row = {
            "name": module.name,
            "layer": module_layer(module.name),
            "kind": module.kind,
            "unit": promotion_unit(module.name, module.kind),
            "params": module.n_params,
            "energy": energy,
            "w4_error": errors[4],
            "w8_error": errors[8],
            "relative_gain": relative_gain,
            "extra_bytes": extra_bytes,
        }
        scores.append(row)
        print(
            f"[{index:>3}/{total}] {module.name}: "
            f"relative W4->W8 gain={relative_gain:.8e}"
        )
    return inventory, scores


def aggregate_units(module_scores: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in module_scores:
        grouped[row["unit"]].append(row)

    units = []
    for name, members in grouped.items():
        energy = sum(row["energy"] for row in members)
        gain = sum(max(row["w4_error"] - row["w8_error"], 0.0) for row in members)
        relative_gain = gain / max(energy, torch.finfo(torch.float32).tiny)
        extra_bytes = sum(row["extra_bytes"] for row in members)
        units.append(
            {
                "name": name,
                "members": [row["name"] for row in members],
                "relative_gain": relative_gain,
                "extra_bytes": extra_bytes,
                "gain_per_extra_gib": relative_gain / (extra_bytes / GIB),
            }
        )
    units.sort(
        key=lambda row: (row["gain_per_extra_gib"], row["relative_gain"]),
        reverse=True,
    )
    for rank, row in enumerate(units, start=1):
        row["rank"] = rank
    return units


def load_reusable_scores(
    path: Path,
    inventory,
    group_size: int,
    target_kinds: set[str],
) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("group_size") != group_size:
        raise ValueError(
            f"Score group_size={payload.get('group_size')} does not match "
            f"--group-size {group_size}"
        )
    if set(payload.get("promotion_kinds", [])) != target_kinds:
        raise ValueError(
            "Score promotion_kinds do not match --promotion-kinds: "
            f"{payload.get('promotion_kinds')} vs {sorted(target_kinds)}"
        )

    module_scores = payload.get("modules")
    if not isinstance(module_scores, list):
        raise ValueError(f"Reusable score file has no module list: {path}")
    expected = {
        module.name for module in inventory.linears if module.kind in target_kinds
    }
    actual = {row.get("name") for row in module_scores}
    if actual != expected:
        raise ValueError(
            "Reusable score modules do not match the model inventory: "
            f"missing={len(expected - actual)}, unexpected={len(actual - expected)}"
        )
    return module_scores


def estimate_selection(inventory, selected: set[str], group_size: int) -> dict:
    bits_map = {
        module.name: (8 if module.name in selected else 4)
        for module in inventory.linears
    }
    return estimate(
        inventory,
        bits_map,
        group_size=group_size,
        symmetric=True,
    )


def exact_module_regex(name: str) -> str:
    return f"re:^{re.escape(name)}$"


def write_policy(
    path: Path,
    group_size: int,
    budget_gib: float,
    result: dict,
    selected: set[str],
    selected_units: list[dict],
    added_units: list[dict],
) -> None:
    down_proj_names = {
        name for name in selected if name.endswith(".mlp.down_proj")
    }
    down_proj_layers = sorted(module_layer(name) for name in down_proj_names)
    other_names = selected - down_proj_names
    policy = {
        "algorithm": "autoround",
        "group_size": group_size,
        "asymmetric": False,
        "promote_down_proj_layers": down_proj_layers,
        "w8_regex": [exact_module_regex(name) for name in sorted(other_names)],
        "bf16_regex": [],
        "metadata": {
            "sensitivity_method": "streamed_symmetric_qdq_relative_gain_per_byte",
            "budget_gib": budget_gib,
            "estimated_total_gib": result["total_bytes"] / GIB,
            "w8_module_count": len(selected),
            "promotion_unit_count": len(selected_units),
            "promotion_units": [row["name"] for row in selected_units],
            "selection_objective": "exact_nested_knapsack_sum_relative_gain",
            "cumulative_proxy_utility": sum(
                row["relative_gain"] for row in selected_units
            ),
            "incremental_proxy_utility": sum(
                row["relative_gain"] for row in added_units
            ),
            "added_promotion_units": [row["name"] for row in added_units],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(policy, handle, sort_keys=False)


def budget_slug(budget_gib: float) -> str:
    text = f"{budget_gib:g}".replace(".", "p")
    return f"{text}g"


def select_exact_additions(
    units: list[dict],
    locked_names: set[str],
    capacity_bytes: int,
) -> list[dict]:
    candidates = [row for row in units if row["name"] not in locked_names]
    if not candidates or capacity_bytes <= 0:
        return []

    quantum = candidates[0]["extra_bytes"]
    for row in candidates[1:]:
        quantum = math.gcd(quantum, row["extra_bytes"])
    capacity = capacity_bytes // quantum
    states: list[tuple[float, tuple[int, ...]] | None] = [None] * (capacity + 1)
    states[0] = (0.0, ())

    for index, row in enumerate(candidates):
        cost = row["extra_bytes"] // quantum
        utility = row["relative_gain"]
        for used in range(capacity, cost - 1, -1):
            previous = states[used - cost]
            if previous is None:
                continue
            proposal = (previous[0] + utility, previous[1] + (index,))
            current = states[used]
            if current is None or proposal[0] > current[0]:
                states[used] = proposal

    best_used, best = max(
        ((used, state) for used, state in enumerate(states) if state is not None),
        key=lambda item: (item[1][0], item[0]),
    )
    del best_used
    return [candidates[index] for index in best[1]]


def generate_nested_policies(
    inventory,
    units: list[dict],
    group_size: int,
    budgets: list[float],
    safety_margin_gib: float,
    policy_dir: Path,
) -> list[dict]:
    selected_modules: set[str] = set()
    selected_units: list[dict] = []
    selected_unit_names: set[str] = set()
    outputs = []

    base = estimate_selection(inventory, selected_modules, group_size)
    for budget in sorted(budgets):
        effective_limit = (budget - safety_margin_gib) * GIB
        if base["total_bytes"] > effective_limit:
            raise ValueError(
                f"Uniform W4 estimate {base['total_bytes'] / GIB:.2f} GiB exceeds "
                f"the effective {budget:.2f} GiB budget"
            )

        current = estimate_selection(inventory, selected_modules, group_size)
        additions = select_exact_additions(
            units,
            selected_unit_names,
            int(effective_limit - current["total_bytes"]),
        )
        for unit in sorted(additions, key=lambda row: row["rank"]):
            selected_unit_names.add(unit["name"])
            selected_units.append(unit)
            selected_modules.update(unit["members"])
        current = estimate_selection(inventory, selected_modules, group_size)

        output_path = policy_dir / (
            f"autoround_gs{group_size}_mixed_{budget_slug(budget)}.yaml"
        )
        write_policy(
            output_path,
            group_size,
            budget,
            current,
            selected_modules,
            selected_units,
            additions,
        )
        outputs.append(
            {
                "path": str(output_path),
                "budget_gib": budget,
                "estimated_total_gib": current["total_bytes"] / GIB,
                "w8_modules": len(selected_modules),
                "promotion_units": len(selected_units),
                "added_units": [row["name"] for row in additions],
                "incremental_proxy_utility": sum(
                    row["relative_gain"] for row in additions
                ),
            }
        )
    return outputs


def parse_budgets(raw: str) -> list[float]:
    budgets = sorted({float(item.strip()) for item in raw.split(",") if item.strip()})
    if not budgets or any(not math.isfinite(value) or value <= 0 for value in budgets):
        raise ValueError("--budgets must contain positive finite GiB values")
    return budgets


def parse_kinds(raw: str) -> set[str]:
    if raw.strip().lower() == "all":
        return set(LINEAR_KINDS)
    kinds = {item.strip() for item in raw.split(",") if item.strip()}
    unknown = sorted(kinds - LINEAR_KINDS)
    if not kinds or unknown:
        raise ValueError(
            "--promotion-kinds must be 'all' or a comma-separated subset of "
            f"{sorted(LINEAR_KINDS)}; unknown={unknown}"
        )
    for fused in ({"q_proj", "k_proj", "v_proj"}, {"gate_proj", "up_proj"}):
        if kinds & fused and not fused <= kinds:
            raise ValueError(
                "Fused promotion kinds must be selected together: "
                f"{sorted(fused)}"
            )
    return kinds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank streamed W4/W8 weight sensitivity and emit mixed policies."
    )
    parser.add_argument("model_dir", help="BF16 model with safetensors index.")
    parser.add_argument("--group-size", type=int, choices=(32, 64, 128), default=32)
    parser.add_argument("--budgets", default="69.5,72,74,76")
    parser.add_argument("--safety-margin-gib", type=float, default=0.25)
    parser.add_argument("--chunk-rows", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--promotion-kinds",
        default="down_proj",
        help="Linear kinds eligible for W8; default down_proj. Use 'all' to opt in.",
    )
    parser.add_argument(
        "--score-json",
        default="sensitivity_gs32.json",
        help="Destination for reusable module and promotion-unit scores.",
    )
    parser.add_argument(
        "--reuse-scores",
        action="store_true",
        help="Load --score-json and regenerate policies without rescoring weights.",
    )
    parser.add_argument(
        "--policy-dir",
        default="recipes/generated",
        help="Directory for generated AutoRound policy YAML files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chunk_rows < 1:
        raise ValueError("--chunk-rows must be >= 1")
    if args.safety_margin_gib < 0:
        raise ValueError("--safety-margin-gib must be >= 0")

    budgets = parse_budgets(args.budgets)
    target_kinds = parse_kinds(args.promotion_kinds)
    score_path = Path(args.score_json)
    if args.reuse_scores:
        inventory = build_inventory(args.model_dir)
        module_scores = load_reusable_scores(
            score_path, inventory, args.group_size, target_kinds
        )
        print(f"Reusing sensitivity scores: {score_path}")
    else:
        device = choose_device(args.device)
        print(
            f"Scoring {args.model_dir} on {device} with symmetric "
            f"W4/W8 group-size {args.group_size}; kinds={sorted(target_kinds)}"
        )
        inventory, module_scores = score_modules(
            args.model_dir,
            args.group_size,
            args.chunk_rows,
            device,
            target_kinds,
        )
    units = aggregate_units(module_scores)

    if not args.reuse_scores:
        score_payload = {
            "model_dir": os.path.abspath(args.model_dir),
            "group_size": args.group_size,
            "method": "streamed_symmetric_qdq_relative_gain_per_byte",
            "promotion_kinds": sorted(target_kinds),
            "modules": module_scores,
            "promotion_units": units,
        }
        score_path.parent.mkdir(parents=True, exist_ok=True)
        with score_path.open("w", encoding="utf-8") as handle:
            json.dump(score_payload, handle, indent=2)
        print(f"Saved sensitivity scores: {score_path}")

    outputs = generate_nested_policies(
        inventory,
        units,
        args.group_size,
        budgets,
        args.safety_margin_gib,
        Path(args.policy_dir),
    )
    for output in outputs:
        print(
            f"{output['budget_gib']:.2f} GiB policy: "
            f"{output['estimated_total_gib']:.2f} GiB estimated, "
            f"{output['promotion_units']} units / {output['w8_modules']} modules -> "
            f"{output['path']}"
        )
        print(
            f"  added {len(output['added_units'])} units; "
            f"incremental proxy utility={output['incremental_proxy_utility']:.8e}"
        )
        for name in output["added_units"]:
            print(f"    + {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
