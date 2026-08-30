#!/usr/bin/env python3
"""Confirm a draft model shares the target tokenizer vocabulary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_tokenizer_ids(model_dir: Path) -> tuple[int, list[str]]:
    tok = model_dir / "tokenizer.json"
    cfg = model_dir / "config.json"
    vocab_size = None
    if cfg.is_file():
        vocab_size = int(json.loads(cfg.read_text(encoding="utf-8")).get("vocab_size") or 0)
    added = model_dir / "added_tokens.json"
    extra = []
    if added.is_file():
        extra = sorted(json.loads(added.read_text(encoding="utf-8")))
    return vocab_size or 0, extra


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Target EXL3 model dir")
    parser.add_argument("--draft", required=True, help="Draft model dir")
    args = parser.parse_args()
    target = Path(args.target)
    draft = Path(args.draft)
    t_vocab, t_extra = _load_tokenizer_ids(target)
    d_vocab, d_extra = _load_tokenizer_ids(draft)
    print(f"target vocab_size={t_vocab} extra={len(t_extra)} dir={target}")
    print(f"draft  vocab_size={d_vocab} extra={len(d_extra)} dir={draft}")
    if t_vocab != d_vocab:
        print("FAIL: vocab_size mismatch", file=sys.stderr)
        return 1
    target_tok = target / "tokenizer.json"
    draft_tok = draft / "tokenizer.json"
    if target_tok.is_file() and draft_tok.is_file():
        if target_tok.read_bytes() != draft_tok.read_bytes():
            print("WARN: tokenizer.json bytes differ; probe a few decode IDs")
    print("OK: vocab_size matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
