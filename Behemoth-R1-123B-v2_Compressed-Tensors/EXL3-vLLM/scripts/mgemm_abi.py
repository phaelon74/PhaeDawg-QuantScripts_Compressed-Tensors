#!/usr/bin/env python3
"""Inspect the pinned exl3_mgemm C++ ABI. inspect.signature cannot read pybind.

Pin 0c49587a7c235e6303a6bbedc8b665272ad3a2ea exposes trailing optional
size_n_list (int32) and c_ptrs (int64). Bitrate is a single scalar K for
the whole group. Per-matrix widths require num_tokens == 1, min_index < 0,
and no weights tensor.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINNED_COMMIT = "0c49587a7c235e6303a6bbedc8b665272ad3a2ea"

# Trailing args of int exl3_mgemm(...) in exl3_gemm.cuh at the pin.
PINNED_PARAM_NAMES = [
    "A",
    "B",
    "C",
    "suh",
    "A_had",
    "svh",
    "indices",
    "weights",
    "K",
    "force_shape_idx",
    "mcg_mult",
    "mul1_mult",
    "min_index",
    "max_index",
    "force_num_sms",
    "num_tokens",
    "size_n_list",
    "c_ptrs",
]


def _mgemm_decl_re() -> re.Pattern[str]:
    return re.compile(
        r"int\s+exl3_mgemm\s*\((.*?)\)\s*;",
        re.DOTALL,
    )


def parse_exl3_mgemm_params(text: str) -> list[str]:
    """Return parameter names from a C++ exl3_mgemm declaration."""
    match = _mgemm_decl_re().search(text)
    if not match:
        raise ValueError("no int exl3_mgemm(...) declaration")
    names = []
    for raw in match.group(1).split(","):
        piece = re.sub(r"/\*.*?\*/", "", raw)
        piece = re.sub(r"//.*", "", piece)
        piece = piece.replace("&", " ")
        piece = re.sub(r"\s+", " ", piece).strip()
        if not piece:
            continue
        piece = piece.split("=", 1)[0].strip()
        ident = piece.split()[-1]
        ident = ident.lstrip("*")
        if ident:
            names.append(ident)
    return names


def find_exl3_gemm_cuh() -> Path | None:
    candidates = []
    env = os.environ.get("EXLLAMAV3_SRC", "").strip()
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            Path("/home/phaedawg/exl3vllm/exllamav3"),
            ROOT.parent.parent.parent / "exllamav3",
            Path.home() / "exl3vllm" / "exllamav3",
        ]
    )
    for root in candidates:
        path = root / "exllamav3" / "exllamav3_ext" / "quant" / "exl3_gemm.cuh"
        if path.is_file():
            return path
        nested = root / "exllamav3_ext" / "quant" / "exl3_gemm.cuh"
        if nested.is_file():
            return nested
    return None


def describe_abi(cuh_text: str | None = None) -> dict[str, object]:
    names = list(PINNED_PARAM_NAMES)
    source = "pinned_commit"
    cuh_path = None
    if cuh_text is None:
        found = find_exl3_gemm_cuh()
        if found is not None:
            cuh_path = str(found)
            cuh_text = found.read_text(encoding="utf-8", errors="replace")
    if cuh_text:
        try:
            names = parse_exl3_mgemm_params(cuh_text)
            source = "cuh" if cuh_path else "text"
        except ValueError:
            source = "pinned_commit_fallback"
            names = list(PINNED_PARAM_NAMES)
    return {
        "pinned_commit": PINNED_COMMIT,
        "source": source,
        "cuh_path": cuh_path,
        "param_names": names,
        "param_count": len(names),
        "has_size_n_list": "size_n_list" in names,
        "has_c_ptrs": "c_ptrs" in names,
        "bitrate_is_scalar_K": "K" in names,
        "per_matrix_bitrate": False,
        "size_n_list_constraints": (
            "num_tokens == 1, min_index < 0, weights is None; "
            "size_n_list is int32, c_ptrs is int64"
        ),
    }


def probe_python_binding() -> dict[str, object]:
    sys.path.insert(0, str(ROOT / "plugin" / "src"))
    from vllm_exl3_sm86.ops import _load_exl3_ext, ext_has_mgemm

    if not ext_has_mgemm():
        return {"present": False}
    import inspect

    fn = _load_exl3_ext().exl3_mgemm
    try:
        params = list(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        params = []
    blob = " ".join(
        [
            " ".join(params),
            str(getattr(fn, "__doc__", "") or ""),
            str(getattr(fn, "__text_signature__", "") or ""),
        ]
    )
    return {
        "present": True,
        "inspect_param_names": params,
        "inspect_param_count": len(params),
        "inspect_has_size_n_list": "size_n_list" in blob,
        "note": "empty inspect results are expected for this pybind; trust the C++ decl",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuh", default="", help="Path to exl3_gemm.cuh")
    parser.add_argument(
        "--output",
        default=str(ROOT / "results" / "mgemm_abi.json"),
    )
    parser.add_argument(
        "--skip-python",
        action="store_true",
        help="Do not load exllamav3_ext (CPU-only parse).",
    )
    args = parser.parse_args()
    text = Path(args.cuh).read_text(encoding="utf-8") if args.cuh else None
    report = describe_abi(text)
    if not args.skip_python:
        try:
            report["python_binding"] = probe_python_binding()
        except Exception as exc:
            report["python_binding"] = {"present": False, "error": str(exc)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
