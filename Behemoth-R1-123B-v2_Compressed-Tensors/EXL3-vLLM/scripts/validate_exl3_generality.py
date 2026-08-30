#!/usr/bin/env python3
"""Prove the plugin loads non-Behemoth EXL3 checkpoints (all codebooks).

Walks a directory of EXL3 model trees, validates metadata, and optionally
loads each under vLLM TP1 eager for a short generate. A 4.25/3inst-only
win is not the deliverable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugin" / "src"))

from vllm_exl3_sm86.metadata import codebook_from_suffixes, stored_suffixes, validate_storage_metadata  # noqa: E402


def _is_exl3(path: Path) -> bool:
    cfg = path / "quantization_config.json"
    if not cfg.is_file():
        return False
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return data.get("quant_method") == "exl3" or "tensor_storage" in data


def _inspect(model: Path) -> dict:
    data = json.loads((model / "quantization_config.json").read_text(encoding="utf-8"))
    storage = data.get("tensor_storage") or {}
    codebooks: dict[str, int] = {}
    for entry in storage.values():
        if not isinstance(entry, dict):
            continue
        try:
            name = codebook_from_suffixes(stored_suffixes(entry)) or "3inst"
        except ValueError:
            name = "invalid"
        codebooks[name] = codebooks.get(name, 0) + 1
    entry = {
        "path": str(model),
        "declared_codebook": data.get("codebook"),
        "bits": data.get("bits") or data.get("bits_per_weight"),
        "head_bits": data.get("head_bits"),
        "storage_entries": len(storage),
        "codebooks": codebooks,
        "config_ok": True,
    }
    try:
        if storage:
            validate_storage_metadata(storage)
    except Exception as exc:
        entry["config_ok"] = False
        entry["config_error"] = str(exc)
    return entry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="Directories that contain EXL3 model folders.",
    )
    parser.add_argument("--load", action="store_true", help="vLLM TP1 eager generate.")
    parser.add_argument(
        "--output",
        default=str(ROOT / "results" / "generality.json"),
    )
    args = parser.parse_args()
    found = []
    for root in args.roots:
        base = Path(root)
        candidates = [base] if _is_exl3(base) else [
            p for p in sorted(base.iterdir()) if p.is_dir() and _is_exl3(p)
        ]
        for model in candidates:
            entry = _inspect(model)
            found.append(entry)
            print(json.dumps(entry))

    if args.load:
        os.environ.setdefault("VLLM_PLUGINS", "vllm_exl3_sm86")
        from vllm import LLM, SamplingParams

        for entry in found:
            if not entry.get("config_ok"):
                continue
            llm = LLM(
                model=entry["path"],
                quantization="exl3",
                tensor_parallel_size=1,
                enforce_eager=True,
                max_model_len=512,
                gpu_memory_utilization=0.5,
            )
            out = llm.generate(["Hello"], SamplingParams(max_tokens=8, temperature=0))
            entry["generate"] = out[0].outputs[0].text
            del llm

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(found, indent=2) + "\n")
    print(f"Wrote {args.output} ({len(found)} checkpoints)")
    if not found:
        return 1
    if any(not e.get("config_ok") for e in found):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
