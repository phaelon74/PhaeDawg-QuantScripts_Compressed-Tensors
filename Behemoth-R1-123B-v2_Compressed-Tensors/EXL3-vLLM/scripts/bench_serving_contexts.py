#!/usr/bin/env python3
"""Benchmark a running vLLM server across exact prompt lengths.

Adapted from phaelon74/b12x scripts/bench_serving_tps.py. Uses streaming
completions to measure TTFT, effective prefill throughput, decode throughput,
TPOT, inter-chunk latency, and end-to-end latency. Stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import time
import urllib.error
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTEXTS = "1024,2048,4096,8192,16384,32768"
METRIC_FIELDS = (
    "ttft_ms",
    "effective_prefill_tok_s",
    "decode_tok_s",
    "tpot_ms",
    "itl_p50_ms",
    "itl_p95_ms",
    "itl_p99_ms",
    "e2e_s",
    "e2e_output_tok_s",
)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def normalize_url(value: str, default_port: int) -> str:
    value = value.strip().rstrip("/")
    if "://" not in value:
        value = f"http://{value}"
    parsed = urlsplit(value)
    if not parsed.hostname:
        raise ValueError(f"Invalid server address: {value!r}")
    netloc = parsed.netloc
    if parsed.port is None:
        host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
        netloc = f"{host}:{default_port}"
    path = parsed.path.rstrip("/")
    if path == "/v1":
        path = ""
    elif path:
        raise ValueError("Server address must not contain a path other than /v1")
    return urlunsplit((parsed.scheme, netloc, path, "", "")).rstrip("/")


class VllmClient:
    def __init__(self, base_url: str, api_key: str, timeout: float) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout

    def request(self, endpoint: str, payload: dict[str, Any]):
        data = json.dumps(payload, separators=(",", ":")).encode()
        return urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

    def post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = self.request(endpoint, payload)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {endpoint}: {detail}") from exc

    def tokenize(self, model: str, text: str) -> list[int]:
        body = self.post_json(
            "/tokenize",
            {
                "model": model,
                "prompt": text,
                "add_special_tokens": False,
            },
        )
        tokens = body.get("tokens")
        if not isinstance(tokens, list) or not all(isinstance(x, int) for x in tokens):
            raise RuntimeError(f"Unexpected /tokenize response: {body}")
        return tokens


def make_prompt_tokens(
    client: VllmClient,
    model: str,
    target_tokens: int,
) -> list[int]:
    """Exact-length prompt with no repeated n-grams.

    The old filler sentence was tiled to length. N-gram speculative decoding
    then drafted that loop at 100% accept, which is not a unique-prompt result.
    Each request is a fresh UUID stream so prompt-lookup cannot cheat.
    """
    parts: list[str] = [
        f"{index}:{uuid.uuid4().hex}"
        for index in range(max(16, target_tokens // 4))
    ]
    text = "Unique serving prompt. " + " ".join(parts)
    tokens = client.tokenize(model, text)
    extra = 0
    while len(tokens) < target_tokens:
        extra += 1
        if extra > 64:
            raise RuntimeError(
                f"/tokenize produced {len(tokens)} tokens; needed {target_tokens}"
            )
        text += " " + " ".join(uuid.uuid4().hex for _ in range(32))
        tokens = client.tokenize(model, text)
    return tokens[:target_tokens]


def run_once(
    client: VllmClient,
    model: str,
    prompt_tokens: list[int],
    output_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt_tokens,
        "max_tokens": output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "seed": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = client.request("/v1/completions", payload)
    chunk_times: list[float] = []
    usage: dict[str, Any] = {}
    encoded_at = time.perf_counter()

    try:
        with urllib.request.urlopen(request, timeout=client.timeout) as response:
            started_at = time.perf_counter()
            for raw_line in response:
                line = raw_line.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                event = json.loads(data)
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    if choice.get("text"):
                        chunk_times.append(time.perf_counter())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(
            f"HTTP {exc.code} from /v1/completions: {detail}"
        ) from exc

    finished_at = time.perf_counter()
    if not chunk_times:
        raise RuntimeError("Streaming response contained no completion text")

    prompt_n = int(usage.get("prompt_tokens", len(prompt_tokens)))
    completion_n = int(usage.get("completion_tokens", len(chunk_times)))
    first_at = chunk_times[0]
    last_at = chunk_times[-1]
    ttft_s = first_at - encoded_at
    e2e_s = finished_at - encoded_at
    decode_intervals = [
        right - left for left, right in zip(chunk_times, chunk_times[1:])
    ]
    decode_s = last_at - first_at
    decode_token_intervals = max(completion_n - 1, 0)
    decode_tok_s = (
        decode_token_intervals / decode_s
        if decode_token_intervals and decode_s > 0
        else None
    )
    tpot_ms = (
        decode_s * 1000.0 / decode_token_intervals
        if decode_token_intervals
        else None
    )

    return {
        "prompt_tokens": prompt_n,
        "completion_tokens": completion_n,
        "requested_completion_tokens": output_tokens,
        "ttft_ms": ttft_s * 1000.0,
        # This includes network transfer and server queueing and is therefore
        # an effective request-level prefill rate, not raw kernel throughput.
        "effective_prefill_tok_s": prompt_n / ttft_s,
        "decode_tok_s": decode_tok_s,
        "tpot_ms": tpot_ms,
        "itl_p50_ms": (
            percentile(decode_intervals, 50) * 1000.0
            if decode_intervals
            else None
        ),
        "itl_p95_ms": (
            percentile(decode_intervals, 95) * 1000.0
            if decode_intervals
            else None
        ),
        "itl_p99_ms": (
            percentile(decode_intervals, 99) * 1000.0
            if decode_intervals
            else None
        ),
        "e2e_s": e2e_s,
        "e2e_output_tok_s": completion_n / e2e_s,
        "response_header_ms": (started_at - encoded_at) * 1000.0,
        "stream_chunk_count": len(chunk_times),
        "stream_chunks_match_tokens": len(chunk_times) == completion_n,
    }


def summarize(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    contexts = sorted({int(run["context_tokens"]) for run in runs})
    for context in contexts:
        selected = [run for run in runs if run["context_tokens"] == context]
        row: dict[str, Any] = {"context_tokens": context, "runs": len(selected)}
        for field in METRIC_FIELDS:
            values = [
                float(run[field]) for run in selected if run.get(field) is not None
            ]
            row[field] = {
                "median": statistics.median(values) if values else None,
                "mean": statistics.fmean(values) if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
        summaries.append(row)
    return summaries


def write_receipts(
    output: Path,
    metadata: dict[str, Any],
    runs: list[dict[str, Any]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": 1,
        **metadata,
        "runs": runs,
        "summary": summarize(runs),
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)

    csv_path = output.with_suffix(".csv")
    fields = ["label", "model", "context_tokens", "run", *METRIC_FIELDS]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerow({field: run.get(field) for field in fields})


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "model"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM TTFT, prefill, decode, TPOT and ITL by context."
    )
    parser.add_argument(
        "--host",
        required=True,
        help="vLLM IP/hostname or base URL; bare hosts default to port 8000.",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("VLLM_API_KEY", ""),
        help="vLLM API key; defaults to VLLM_API_KEY.",
    )
    parser.add_argument("--model", required=True, help="Served model name.")
    parser.add_argument("--label", default="", help="Receipt label, e.g. exl3-4p25.")
    parser.add_argument("--contexts", default=DEFAULT_CONTEXTS)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if not args.api_key:
        parser.error("Pass --api-key or set VLLM_API_KEY")
    contexts = [int(value) for value in args.contexts.split(",") if value.strip()]
    if not contexts or any(value <= 0 for value in contexts):
        parser.error("--contexts must contain positive comma-separated integers")
    if args.output_tokens < 2:
        parser.error("--output-tokens must be at least 2 for decode timing")
    if args.runs < 1 or args.warmup_runs < 0:
        parser.error("--runs must be >=1 and --warmup-runs must be >=0")

    base_url = normalize_url(args.host, args.port)
    label = args.label or args.model
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = (
        Path(args.output)
        if args.output
        else ROOT / "results" / f"serving_{safe_name(label)}_{timestamp}.json"
    )
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "label": label,
        "model": args.model,
        "server": base_url,
        "contexts": contexts,
        "output_tokens": args.output_tokens,
        "measured_runs_per_context": args.runs,
        "warmup_runs_per_context": args.warmup_runs,
        "request_concurrency": 1,
        "prompt_format": (
            "exact token IDs from /tokenize; unique UUID stream per request "
            "(no repeated filler, so n-gram spec cannot cheat)"
        ),
        "prefix_cache_policy": "unique random first block for every request",
        "prefill_metric_note": (
            "effective_prefill_tok_s = prompt_tokens / TTFT; includes network "
            "and server queueing"
        ),
        "itl_metric_note": (
            "ITL percentiles use non-empty SSE chunk arrival intervals. "
            "stream_chunks_match_tokens reports whether chunks mapped 1:1."
        ),
    }
    client = VllmClient(base_url, args.api_key, args.timeout)
    runs: list[dict[str, Any]] = []

    print(f"server={base_url} model={args.model!r} label={label!r}")
    print(
        f"contexts={contexts} output_tokens={args.output_tokens} "
        f"runs={args.runs} warmups={args.warmup_runs}"
    )
    for context in contexts:
        for warmup in range(args.warmup_runs):
            print(
                f"context={context:5d} warmup={warmup + 1}/{args.warmup_runs}",
                flush=True,
            )
            prompt = make_prompt_tokens(client, args.model, context)
            result = run_once(client, args.model, prompt, args.output_tokens)
            if result["prompt_tokens"] != context:
                raise RuntimeError(
                    f"Server counted {result['prompt_tokens']} prompt tokens; "
                    f"expected {context}"
                )

        for run_number in range(1, args.runs + 1):
            prompt = make_prompt_tokens(client, args.model, context)
            result = run_once(client, args.model, prompt, args.output_tokens)
            result.update(
                {
                    "label": label,
                    "model": args.model,
                    "context_tokens": context,
                    "run": run_number,
                }
            )
            if result["prompt_tokens"] != context:
                raise RuntimeError(
                    f"Server counted {result['prompt_tokens']} prompt tokens; "
                    f"expected {context}"
                )
            if result["completion_tokens"] != args.output_tokens:
                raise RuntimeError(
                    f"Server returned {result['completion_tokens']} completion "
                    f"tokens; expected {args.output_tokens}"
                )
            runs.append(result)
            write_receipts(output, metadata, runs)
            print(
                f"context={context:5d} run={run_number}/{args.runs} "
                f"TTFT={result['ttft_ms']:8.2f}ms "
                f"prefill={result['effective_prefill_tok_s']:8.2f} tok/s "
                f"decode={result['decode_tok_s']:6.2f} tok/s "
                f"TPOT={result['tpot_ms']:7.2f}ms "
                f"ITL-p95={result['itl_p95_ms']:7.2f}ms",
                flush=True,
            )

    print(f"Wrote {output}")
    print(f"Wrote {output.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
