"""Checkpoint metadata helpers that do not import vLLM."""

from __future__ import annotations

from typing import Any

from .constants import MAX_BITRATE, MIN_BITRATE, TRELLIS_TILE


def bitrate_from_trellis_shape(shape: tuple[int, ...] | list[int]) -> int:
    """Integer K from trellis packing: last dim is 16 * bitrate."""
    if len(shape) != 3:
        raise ValueError(f"EXL3 trellis must be rank-3, got shape {tuple(shape)}")
    last = int(shape[-1])
    if last % TRELLIS_TILE:
        raise ValueError(f"EXL3 trellis last dim {last} is not a multiple of {TRELLIS_TILE}")
    bitrate = last // TRELLIS_TILE
    if not MIN_BITRATE <= bitrate <= MAX_BITRATE:
        raise ValueError(f"EXL3 bitrate {bitrate} is outside {MIN_BITRATE}..{MAX_BITRATE}")
    return bitrate


def stored_suffixes(entry: dict[str, Any]) -> set[str]:
    stored = entry.get("stored_tensors", {})
    return {name.rsplit(".", 1)[-1] for name in stored}


def codebook_from_suffixes(suffixes: set[str]) -> str | None:
    has_mcg = "mcg" in suffixes
    has_mul1 = "mul1" in suffixes
    if has_mcg and has_mul1:
        raise ValueError("EXL3 record lists both mcg and mul1")
    if has_mcg:
        return "mcg"
    if has_mul1:
        return "mul1"
    return None


def validate_storage_metadata(tensor_storage: dict[str, Any]) -> int:
    """Return the number of EXL3 records. Raise on missing/conflicting markers."""
    bad: list[str] = []
    exl3_count = 0
    for prefix, entry in tensor_storage.items():
        if entry.get("quant_format") != "exl3":
            continue
        exl3_count += 1
        suffixes = stored_suffixes(entry)
        required: set[str] = {"trellis"}
        if not ({"suh", "su"} & suffixes):
            required.add("suh|su")
        if not ({"svh", "sv"} & suffixes):
            required.add("svh|sv")
        missing = [name for name in required if name not in suffixes and name != "trellis"]
        if "trellis" not in suffixes:
            missing.append("trellis")
        if missing:
            bad.append(f"{prefix}: missing {','.join(sorted(missing))}")
        try:
            codebook_from_suffixes(suffixes)
        except ValueError:
            bad.append(f"{prefix}: both mcg and mul1 are present")
        stored = entry.get("stored_tensors", {})
        trellis_meta = next(
            (info for name, info in stored.items() if name.endswith(".trellis")),
            None,
        )
        if trellis_meta and "shape" in trellis_meta:
            try:
                bitrate_from_trellis_shape(trellis_meta["shape"])
            except ValueError as exc:
                bad.append(f"{prefix}: {exc}")
    if not exl3_count:
        raise ValueError("quantization_config.json has no EXL3 tensor records")
    if bad:
        raise ValueError("Invalid EXL3 tensor metadata: " + "; ".join(bad[:16]))
    return exl3_count


def expand_prefix_candidates(prefix: str) -> list[str]:
    candidates = [prefix]
    if prefix.startswith("model."):
        candidates.append(prefix.removeprefix("model."))
    else:
        candidates.append(f"model.{prefix}")
    parts = prefix.split(".")
    for removable in ("model", "language_model"):
        for idx in range(0, len(parts) - 1):
            if parts[idx] != removable:
                continue
            collapsed = ".".join(parts[:idx] + parts[idx + 1 :])
            candidates.extend((collapsed, f"model.{collapsed}"))
            if collapsed.startswith("model."):
                candidates.append(collapsed.removeprefix("model."))
    seen: dict[str, None] = {}
    for candidate in candidates:
        seen.setdefault(candidate, None)
    return list(seen)


def storage_entry(tensor_storage: dict[str, Any], prefix: str) -> dict[str, Any] | None:
    for candidate in expand_prefix_candidates(prefix):
        entry = tensor_storage.get(candidate)
        if entry is not None:
            return entry
    return None
