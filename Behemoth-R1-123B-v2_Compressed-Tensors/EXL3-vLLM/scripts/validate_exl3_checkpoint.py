#!/usr/bin/env python3
"""Validate an EXL3 safetensors checkpoint before vLLM load."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugin" / "src"))

from vllm_exl3_sm86.constants import (  # noqa: E402
    BEHEMOTH_DECODER_LINEARS,
    HEAD_BITRATE,
)
from vllm_exl3_sm86.metadata import (  # noqa: E402
    bitrate_from_trellis_shape,
    codebook_from_suffixes,
    stored_suffixes,
    validate_storage_metadata,
)

DECODER_LEAVES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}

LAYER_RE = re.compile(r"layers\.(\d+)\.")


def _bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _profile_errors(
    profile: str,
    bitrates: Counter,
    codebooks: Counter,
    count: int,
    size: int,
) -> list[str]:
    errors: list[str] = []
    gib = size / (1024**3)
    if profile == "inventory":
        return errors
    if profile == "mul1-behemoth":
        if set(bitrates) - {3, 4}:
            errors.append(f"decoder bitrates should be K3/K4, got {dict(bitrates)}")
        if codebooks.get("mul1", 0) != count:
            errors.append(f"expected all mul1 markers, got {dict(codebooks)}")
        if gib < 40 or gib > 70:
            errors.append(f"size {gib:.2f} GiB is outside 40-70 GiB; verify")
        return errors
    if profile == "artusdev-4p25":
        if set(bitrates) - {4, 5, 6}:
            errors.append(
                f"ArtusDev 4.25 decoder bitrates should be K4/K5/K6, got {dict(bitrates)}"
            )
        if {4, 5, 6} - set(bitrates):
            errors.append(
                f"ArtusDev 4.25 must include K4, K5, and K6, got {dict(bitrates)}"
            )
        if codebooks.get("none", 0) != count:
            errors.append(
                f"ArtusDev 4.25 should be implicit 3inst (no mul1/mcg), got {dict(codebooks)}"
            )
        if gib < 55 or gib > 68:
            errors.append(f"size {gib:.2f} GiB is outside the 4.25 ~62 GiB band; verify")
        return errors
    errors.append(f"unknown profile {profile}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--allow-non-behemoth", action="store_true")
    parser.add_argument(
        "--profile",
        choices=("mul1-behemoth", "artusdev-4p25", "inventory"),
        default="mul1-behemoth",
        help="mul1-behemoth is the local 4.5 conversion. artusdev-4p25 is the "
        "serving candidate (implicit 3inst, mixed K3/K4/K5).",
    )
    parser.add_argument("--sha-manifest", default="")
    args = parser.parse_args()
    ckpt = Path(args.checkpoint)
    qcfg_path = ckpt / "quantization_config.json"
    if not qcfg_path.is_file():
        print(f"missing {qcfg_path}", file=sys.stderr)
        return 1
    qcfg = json.loads(qcfg_path.read_text(encoding="utf-8"))
    storage = qcfg.get("tensor_storage") or {}
    count = validate_storage_metadata(storage)
    decoder = []
    head = None
    bitrates = Counter()
    codebooks = Counter()
    leaf_bitrates: dict[str, Counter[int]] = defaultdict(Counter)
    layer_bitrates: dict[int, dict[str, int]] = defaultdict(dict)
    for prefix, entry in storage.items():
        if entry.get("quant_format") != "exl3":
            continue
        suffixes = stored_suffixes(entry)
        codebook = codebook_from_suffixes(suffixes)
        codebooks[codebook or "none"] += 1
        trellis = next(
            info for name, info in entry.get("stored_tensors", {}).items()
            if name.endswith(".trellis")
        )
        bitrate = int(entry.get("bits_per_weight") or bitrate_from_trellis_shape(trellis["shape"]))
        leaf = prefix.rsplit(".", 1)[-1]
        if leaf == "lm_head" or prefix.endswith("lm_head"):
            head = {"prefix": prefix, "bitrate": bitrate, "codebook": codebook}
        elif leaf in DECODER_LEAVES:
            decoder.append(prefix)
            bitrates[bitrate] += 1
            leaf_bitrates[leaf][bitrate] += 1
            layer_match = LAYER_RE.search(prefix)
            if layer_match:
                layer_bitrates[int(layer_match.group(1))][leaf] = bitrate
    size = _bytes(ckpt)
    leaf_bitrate_dict = {
        leaf: dict(sorted(counts.items())) for leaf, counts in sorted(leaf_bitrates.items())
    }
    decoder_layer_bitrates = [
        {"layer": idx, **leaves}
        for idx, leaves in sorted(layer_bitrates.items())
    ]
    print(f"exl3_records={count}")
    print(f"decoder_linears={len(decoder)}")
    print(f"decoder_bitrates={dict(sorted(bitrates.items()))}")
    print(f"decoder_leaf_bitrates={leaf_bitrate_dict}")
    print(f"decoder_layers_with_bitrates={len(decoder_layer_bitrates)}")
    print(f"codebooks={dict(codebooks)}")
    print(f"lm_head={head}")
    print(f"bytes={size} ({size / (1024**3):.2f} GiB)")
    print(f"profile={args.profile}")
    errors = []
    if not args.allow_non_behemoth:
        if len(decoder) != BEHEMOTH_DECODER_LINEARS:
            errors.append(
                f"expected {BEHEMOTH_DECODER_LINEARS} decoder linears, got {len(decoder)}"
            )
        if head is None or head["bitrate"] != HEAD_BITRATE:
            errors.append(f"expected H6 lm_head, got {head}")
        errors.extend(_profile_errors(args.profile, bitrates, codebooks, count, size))
    if args.sha_manifest:
        Path(args.sha_manifest).write_text(
            json.dumps(
                {
                    "checkpoint": str(ckpt),
                    "profile": args.profile,
                    "bytes": size,
                    "decoder_linears": len(decoder),
                    "head": head,
                    "bitrates": dict(sorted(bitrates.items())),
                    "decoder_leaf_bitrates": leaf_bitrate_dict,
                    "decoder_layer_bitrates": decoder_layer_bitrates,
                    "codebooks": dict(codebooks),
                    "errors": errors,
                },
                indent=2,
            )
            + "\n"
        )
    if errors:
        print("INVALID:", *errors, sep="\n  ", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
