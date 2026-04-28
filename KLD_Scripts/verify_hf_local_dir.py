#!/usr/bin/env python3
"""Verify local snapshot under --local-dir matches Hugging Face hub (sizes + optional sha256 + safetensors headers)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path


def _safetensors_header_ok(path: Path) -> None:
    with path.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated file (length prefix)")
        (header_len,) = struct.unpack("<Q", prefix)
        if header_len < 2 or header_len > 512 * 1024 * 1024:
            raise ValueError(f"invalid header length {header_len}")
        header_bytes = f.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError("truncated file (header JSON incomplete vs length prefix)")
    json.loads(header_bytes.decode("utf-8"))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024 * 64)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--local-dir", required=True, type=Path)
    p.add_argument("--revision", default=None, help="Git revision (default: repo default branch)")
    args = p.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("error: install huggingface_hub (e.g. pip install huggingface_hub)", file=sys.stderr)
        return 2

    root: Path = args.local_dir.resolve()
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 1

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    try:
        try:
            info = api.model_info(args.repo_id, revision=args.revision, files_metadata=True)
        except TypeError:
            info = api.model_info(args.repo_id, revision=args.revision)
    except Exception as e:
        print(f"error: model_info failed: {e}", file=sys.stderr)
        return 1

    errors: list[str] = []
    siblings = getattr(info, "siblings", None) or []

    for s in siblings:
        if isinstance(s, dict):
            rfn = s.get("rfilename") or s.get("path")
            size = s.get("size")
            sha = s.get("sha256")
        else:
            rfn = getattr(s, "rfilename", None) or getattr(s, "path", None)
            size = getattr(s, "size", None)
            sha = getattr(s, "sha256", None)
        if not rfn:
            continue
        if size is None:
            continue
        local = root / rfn
        if not local.is_file():
            errors.append(f"missing file (hub lists it): {rfn}")
            continue
        act = local.stat().st_size
        if act != int(size):
            errors.append(f"size mismatch {rfn}: local={act} hub={size}")
            continue
        if sha:
            got = _sha256_file(local)
            if got.lower() != str(sha).lower():
                errors.append(f"sha256 mismatch {rfn}: local={got} hub={sha}")

    for st in sorted(root.rglob("*.safetensors")):
        if not st.is_file():
            continue
        if st.name.endswith(".safetensors.index.json"):
            continue
        try:
            _safetensors_header_ok(st)
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as e:
            errors.append(f"safetensors header invalid {st.relative_to(root)}: {e}")

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    def _sized(s: object) -> bool:
        if isinstance(s, dict):
            return bool(s.get("rfilename") or s.get("path")) and s.get("size") is not None
        return bool(getattr(s, "rfilename", None) or getattr(s, "path", None)) and getattr(s, "size", None) is not None

    n_sized = sum(1 for s in siblings if _sized(s))
    print(f"ok: {n_sized} hub files (size match); safetensors headers ok under {root}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
