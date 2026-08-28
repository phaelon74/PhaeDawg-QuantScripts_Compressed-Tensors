#!/usr/bin/env python3
"""Pick per-shape SM86 reconstruct thresholds from kernel_microbench.json.

Chooses the smallest M where reconstruct+hgemm is faster than compressed EXL3.
Does not copy the GLM/SM120 default of 144.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--microbench", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = json.loads(Path(args.microbench).read_text())["rows"]
    grouped: dict[tuple, list] = defaultdict(list)
    for row in rows:
        grouped[(row["k"], row["n"], row["bitrate"], row["name"])].append(row)
    thresholds = []
    for (k, n, bitrate, name), items in sorted(grouped.items()):
        items = sorted(items, key=lambda r: r["m"])
        chosen = None
        for row in items:
            if row["reconstruct_ms"] < row["exl3_ms"]:
                chosen = row["m"]
                break
        thresholds.append(
            {
                "k": k,
                "n": n,
                "bitrate": bitrate,
                "layer": name,
                "m": chosen,
            }
        )
    Path(args.output).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "arch": "sm86",
                "source": args.microbench,
                "thresholds": thresholds,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
