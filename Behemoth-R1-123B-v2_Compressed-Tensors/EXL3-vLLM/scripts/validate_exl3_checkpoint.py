#!/usr/bin/env python3
"""Validate an EXL3 safetensors checkpoint before vLLM load."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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


def _bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--allow-non-behemoth", action="store_true")
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
    size = _bytes(ckpt)
    print(f"exl3_records={count}")
    print(f"decoder_linears={len(decoder)}")
    print(f"decoder_bitrates={dict(bitrates)}")
    print(f"codebooks={dict(codebooks)}")
    print(f"lm_head={head}")
    print(f"bytes={size} ({size / (1024**3):.2f} GiB)")
    errors = []
    if not args.allow_non_behemoth:
        if len(decoder) != BEHEMOTH_DECODER_LINEARS:
            errors.append(
                f"expected {BEHEMOTH_DECODER_LINEARS} decoder linears, got {len(decoder)}"
            )
        if head is None or head["bitrate"] != HEAD_BITRATE:
            errors.append(f"expected H6 lm_head, got {head}")
        if set(bitrates) - {3, 4}:
            errors.append(f"decoder bitrates should be K3/K4, got {dict(bitrates)}")
        if codebooks.get("mul1", 0) != count:
            errors.append(f"expected all mul1 markers, got {dict(codebooks)}")
        if size < 40 * 1024**3 or size > 70 * 1024**3:
            errors.append(
                f"size {size / (1024**3):.2f} GiB is outside the expected low-50s range; verify"
            )
    if errors:
        print("INVALID:", *errors, sep="\n  ", file=sys.stderr)
        return 1
    if args.sha_manifest:
        Path(args.sha_manifest).write_text(
            json.dumps(
                {
                    "checkpoint": str(ckpt),
                    "bytes": size,
                    "decoder_linears": len(decoder),
                    "head": head,
                    "bitrates": dict(bitrates),
                    "codebooks": dict(codebooks),
                },
                indent=2,
            )
            + "\n"
        )
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
