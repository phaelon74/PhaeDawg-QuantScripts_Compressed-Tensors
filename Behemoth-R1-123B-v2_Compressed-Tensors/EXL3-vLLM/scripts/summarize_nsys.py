#!/usr/bin/env python3
"""Bucket nsys cuda_gpu_kern_sum output into GEMM / attention / NCCL / other.

Accepts the CSV from:
  nsys stats --report cuda_gpu_kern_sum --format csv report.nsys-rep
or a directory containing that CSV. Time columns may be ns or ms.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BUCKETS = (
    "exl3_mgemm",
    "exl3_gemm",
    "attention",
    "comm",
    "elementwise",
    "sampling",
    "memcpy",
    "other",
)


def bucket_kernel(name: str) -> str:
    n = name.lower()
    if any(
        token in n
        for token in ("nccl", "allreduce", "all_reduce", "pynccl", "customallreduce")
    ):
        return "comm"
    if "mgemm" in n:
        return "exl3_mgemm"
    if "exl3" in n or "qtip" in n or re.search(r"(^|[_])gemv", n):
        return "exl3_gemm"
    if any(
        token in n
        for token in (
            "flash",
            "fmha",
            "mha",
            "attn",
            "reshape_and_cache",
            "paged_attention",
        )
    ):
        return "attention"
    if any(
        token in n
        for token in ("rmsnorm", "silu", "rotary", "rope", "hadamard", "had_r")
    ):
        return "elementwise"
    if any(token in n for token in ("sample", "softmax", "topk", "topp")):
        return "sampling"
    if "memcpy" in n or "memset" in n:
        return "memcpy"
    return "other"


def _to_ms(value: float, header: str) -> float:
    h = header.lower()
    if "(ns)" in h or h.endswith("_ns") or "nanosecond" in h:
        return value / 1e6
    if "(us)" in h or "micro" in h:
        return value / 1e3
    if "(ms)" in h or "milli" in h:
        return value
    if "(s)" in h and "ns" not in h:
        return value * 1e3
    if value > 1e7:
        return value / 1e6
    return value


def parse_kern_sum(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    header_idx = next(
        (
            i
            for i, line in enumerate(lines)
            if "name" in line.lower() and ("time" in line.lower() or "total" in line.lower())
        ),
        None,
    )
    if header_idx is None:
        raise ValueError(f"no kernel-summary header in {path}")
    reader = csv.DictReader(lines[header_idx:])
    if not reader.fieldnames:
        raise ValueError(f"empty CSV header in {path}")
    name_col = next(c for c in reader.fieldnames if c.lower() == "name")
    time_col = next(
        (
            c
            for c in reader.fieldnames
            if "total time" in c.lower() or c.lower() in {"time", "total"}
        ),
        None,
    )
    if time_col is None:
        time_col = next(c for c in reader.fieldnames if "time" in c.lower())
    inst_col = next(
        (c for c in reader.fieldnames if "instance" in c.lower()),
        None,
    )
    rows = []
    for raw in reader:
        name = (raw.get(name_col) or "").strip()
        if not name:
            continue
        try:
            total = float(str(raw[time_col]).replace(",", ""))
        except (TypeError, ValueError):
            continue
        instances = 0
        if inst_col:
            try:
                instances = int(float(str(raw[inst_col]).replace(",", "")))
            except (TypeError, ValueError):
                instances = 0
        rows.append(
            {
                "name": name,
                "total_ms": _to_ms(total, time_col),
                "instances": instances,
                "bucket": bucket_kernel(name),
            }
        )
    return rows


def summarize(rows: list[dict], decode_tokens: int) -> dict[str, object]:
    totals = {bucket: 0.0 for bucket in BUCKETS}
    instances = {bucket: 0 for bucket in BUCKETS}
    for row in rows:
        totals[row["bucket"]] += float(row["total_ms"])
        instances[row["bucket"]] += int(row["instances"])
    wall_ms = sum(totals.values())
    per_token = None
    if decode_tokens > 0 and wall_ms > 0:
        per_token = {bucket: totals[bucket] / decode_tokens for bucket in BUCKETS}
        per_token["all"] = wall_ms / decode_tokens
    gemm_ms = totals["exl3_gemm"] + totals["exl3_mgemm"]
    top = sorted(rows, key=lambda r: r["total_ms"], reverse=True)[:20]
    return {
        "kernel_rows": len(rows),
        "total_ms": wall_ms,
        "ms_by_bucket": totals,
        "instances_by_bucket": instances,
        "gemm_share": gemm_ms / wall_ms if wall_ms else None,
        "comm_share": totals["comm"] / wall_ms if wall_ms else None,
        "attention_share": totals["attention"] / wall_ms if wall_ms else None,
        "decode_tokens": decode_tokens or None,
        "ms_per_token_by_bucket": per_token,
        "top_kernels": top,
    }


def parse_serve_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "pair_mgemm_enabled": "fused exl3_mgemm decode enabled for packed" in text,
        "kv_mgemm_enabled": "fused exl3_mgemm decode enabled for K/V" in text,
        "mgemm_warmup_failed": "mgemm warmup failed" in text,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="cuda_gpu_kern_sum CSV or a directory")
    parser.add_argument("--serve-log", default="")
    parser.add_argument("--decode-tokens", type=int, default=0)
    parser.add_argument(
        "--output",
        default=str(ROOT / "results" / "serving_attribution.json"),
    )
    args = parser.parse_args()
    path = Path(args.input)
    if path.is_dir():
        csvs = sorted(path.glob("*kern*.csv")) + sorted(path.glob("*.csv"))
        if not csvs:
            print(f"no CSV under {path}", file=sys.stderr)
            return 2
        path = csvs[0]
    rows = parse_kern_sum(path)
    report = summarize(rows, args.decode_tokens)
    report["source"] = str(path)
    if args.serve_log:
        report["serve_log"] = parse_serve_log(Path(args.serve_log))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "top_kernels"}, indent=2))
    print("top kernels:")
    for row in report["top_kernels"][:10]:
        print(f"  {row['total_ms']:.2f}ms  {row['bucket']:12}  {row['name'][:80]}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
