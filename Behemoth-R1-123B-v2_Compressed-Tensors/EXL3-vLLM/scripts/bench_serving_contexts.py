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


def _unique_uuid_tokens(
    client: VllmClient,
    model: str,
    target_tokens: int,
) -> tuple[list[int], str]:
    parts = [
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
    return tokens[:target_tokens], text


def make_prompt_tokens(
    client: VllmClient,
    model: str,
    target_tokens: int,
    *,
    style: str,
) -> tuple[list[int], str]:
    """Exact-length unique prompt. Returns (token ids, source text).

    uuid: incompressible ID stream. english: unique ID pad plus a unique
    story instruction at the end so the model continues prose, not hex.
    """
    if style == "uuid":
        return _unique_uuid_tokens(client, model, target_tokens)
    if style != "english":
        raise ValueError(f"unknown prompt style: {style}")

    seed = uuid.uuid4().hex
    tail = (
        "\n\nIgnore the identifier noise above. Write a coherent original "
        "scene about a lighthouse keeper in a winter storm. Do not repeat "
        "yourself and do not copy the identifiers. "
        f"Seed {seed}. Continuation:\n"
    )
    tail_tokens = client.tokenize(model, tail)
    if len(tail_tokens) >= target_tokens:
        return tail_tokens[:target_tokens], tail
    pad_tokens, pad_text = _unique_uuid_tokens(
        client, model, target_tokens - len(tail_tokens)
    )
    return pad_tokens + tail_tokens, pad_text + tail


def run_once(
    client: VllmClient,
    model: str,
    prompt_tokens: list[int],
    output_tokens: int,
    *,
    temperature: float,
    ignore_eos: bool,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt_tokens,
        "max_tokens": output_tokens,
        "temperature": temperature,
        "ignore_eos": ignore_eos,
        "seed": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = client.request("/v1/completions", payload)
    chunk_times: list[float] = []
    pieces: list[str] = []
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
                    text = choice.get("text")
                    if text:
                        pieces.append(text)
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
        "completion_text": "".join(pieces),
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


def write_response_file(
    directory: Path,
    *,
    context: int,
    run_number: int,
    prompt_text: str,
    prompt_tail_chars: int,
    save_full_prompt: bool,
    result: dict[str, Any],
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"c{context}_run{run_number}.txt"
    if save_full_prompt:
        prompt_block = prompt_text
        prompt_note = "full prompt"
    else:
        if len(prompt_text) <= prompt_tail_chars:
            prompt_block = prompt_text
            prompt_note = "full prompt (fits in tail window)"
        else:
            prompt_block = prompt_text[-prompt_tail_chars:]
            prompt_note = (
                f"last {prompt_tail_chars} chars of {len(prompt_text)}-char prompt"
            )
    header = (
        f"context_tokens={context}\n"
        f"run={run_number}\n"
        f"prompt_tokens={result.get('prompt_tokens')}\n"
        f"completion_tokens={result.get('completion_tokens')}\n"
        f"decode_tok_s={result.get('decode_tok_s')}\n"
        f"ttft_ms={result.get('ttft_ms')}\n"
        f"tpot_ms={result.get('tpot_ms')}\n"
        f"itl_p95_ms={result.get('itl_p95_ms')}\n"
        f"prompt_section={prompt_note}\n"
    )
    path.write_text(
        header
        + "===== PROMPT =====\n"
        + prompt_block
        + "\n===== COMPLETION =====\n"
        + str(result.get("completion_text") or "")
        + "\n",
        encoding="utf-8",
    )
    return path


def write_receipts(
    output: Path,
    metadata: dict[str, Any],
    runs: list[dict[str, Any]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": 2,
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
    parser.add_argument(
        "--contexts",
        default=DEFAULT_CONTEXTS,
        help="Comma-separated prompt lengths. Coherency sweep: "
        "1024,4096,8192,16384,32768",
    )
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--prompt-style",
        choices=("english", "uuid"),
        default="english",
        help="english: unique ID pad plus a unique story tail so decode is "
        "prose, not hex. uuid: incompressible ID stream only.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0 keeps timing comparable; try 0.7 if "
        "greedy prose looks stuck.",
    )
    parser.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force max_tokens (default). --no-ignore-eos lets the model stop "
        "early; completion length may then be under --output-tokens.",
    )
    parser.add_argument(
        "--save-responses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write one .txt per measured run with prompt tail and full "
        "completion (default on). JSON always stores completion_text.",
    )
    parser.add_argument(
        "--print-responses",
        action="store_true",
        help="Print each measured completion to stdout after the timing line.",
    )
    parser.add_argument(
        "--save-full-prompts",
        action="store_true",
        help="With --save-responses, write the entire prompt text (large at 32k).",
    )
    parser.add_argument(
        "--prompt-tail-chars",
        type=int,
        default=2000,
        help="Chars of prompt tail kept in response .txt files.",
    )
    parser.add_argument(
        "--response-dir",
        default="",
        help="Directory for response .txt files. Default: <receipt>_responses/",
    )
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
    if args.temperature < 0:
        parser.error("--temperature must be >= 0")
    if args.prompt_tail_chars < 1:
        parser.error("--prompt-tail-chars must be >= 1")

    base_url = normalize_url(args.host, args.port)
    label = args.label or args.model
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = (
        Path(args.output)
        if args.output
        else ROOT / "results" / f"serving_{safe_name(label)}_{timestamp}.json"
    )
    response_dir = (
        Path(args.response_dir)
        if args.response_dir
        else output.with_name(output.stem + "_responses")
    )
    prompt_notes = {
        "english": (
            "exact token IDs from /tokenize; unique UUID pad plus a unique "
            "English story tail so decode is prose (n-gram cannot cheat on "
            "tiled filler, and completions are readable)"
        ),
        "uuid": (
            "exact token IDs from /tokenize; unique UUID stream per request "
            "(no repeated filler; greedy decode may still continue hex IDs)"
        ),
    }
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "schema_version": 2,
        "label": label,
        "model": args.model,
        "server": base_url,
        "contexts": contexts,
        "output_tokens": args.output_tokens,
        "measured_runs_per_context": args.runs,
        "warmup_runs_per_context": args.warmup_runs,
        "request_concurrency": 1,
        "prompt_style": args.prompt_style,
        "temperature": args.temperature,
        "ignore_eos": args.ignore_eos,
        "prompt_format": prompt_notes[args.prompt_style],
        "prefix_cache_policy": "unique random first block for every request",
        "responses_dir": str(response_dir) if args.save_responses else None,
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
        f"runs={args.runs} warmups={args.warmup_runs} "
        f"prompt_style={args.prompt_style} temperature={args.temperature} "
        f"ignore_eos={args.ignore_eos}"
    )
    max_needed = max(contexts) + args.output_tokens
    print(
        f"Need max_model_len >= {max_needed} on the server "
        f"(16k/32k fail if the serve is still at 8192).",
        flush=True,
    )
    if args.save_responses:
        print(f"responses={response_dir}", flush=True)

    def one_request(token_count: int) -> tuple[list[int], str, dict[str, Any]]:
        prompt_ids, prompt_text = make_prompt_tokens(
            client,
            args.model,
            token_count,
            style=args.prompt_style,
        )
        result = run_once(
            client,
            args.model,
            prompt_ids,
            args.output_tokens,
            temperature=args.temperature,
            ignore_eos=args.ignore_eos,
        )
        return prompt_ids, prompt_text, result

    for context in contexts:
        for warmup in range(args.warmup_runs):
            print(
                f"context={context:5d} warmup={warmup + 1}/{args.warmup_runs}",
                flush=True,
            )
            _, _, result = one_request(context)
            if result["prompt_tokens"] != context:
                raise RuntimeError(
                    f"Server counted {result['prompt_tokens']} prompt tokens; "
                    f"expected {context}"
                )

        for run_number in range(1, args.runs + 1):
            _, prompt_text, result = one_request(context)
            result.update(
                {
                    "label": label,
                    "model": args.model,
                    "context_tokens": context,
                    "run": run_number,
                    "prompt_style": args.prompt_style,
                    "temperature": args.temperature,
                    "ignore_eos": args.ignore_eos,
                    "prompt_tail": prompt_text[-args.prompt_tail_chars :],
                }
            )
            if result["prompt_tokens"] != context:
                raise RuntimeError(
                    f"Server counted {result['prompt_tokens']} prompt tokens; "
                    f"expected {context}"
                )
            if (
                args.ignore_eos
                and result["completion_tokens"] != args.output_tokens
            ):
                raise RuntimeError(
                    f"Server returned {result['completion_tokens']} completion "
                    f"tokens; expected {args.output_tokens}"
                )
            runs.append(result)
            write_receipts(output, metadata, runs)
            if args.save_responses:
                write_response_file(
                    response_dir,
                    context=context,
                    run_number=run_number,
                    prompt_text=prompt_text,
                    prompt_tail_chars=args.prompt_tail_chars,
                    save_full_prompt=args.save_full_prompts,
                    result=result,
                )
            print(
                f"context={context:5d} run={run_number}/{args.runs} "
                f"TTFT={result['ttft_ms']:8.2f}ms "
                f"prefill={result['effective_prefill_tok_s']:8.2f} tok/s "
                f"decode={result['decode_tok_s']:6.2f} tok/s "
                f"TPOT={result['tpot_ms']:7.2f}ms "
                f"ITL-p95={result['itl_p95_ms']:7.2f}ms "
                f"out_tokens={result['completion_tokens']}",
                flush=True,
            )
            preview = " ".join((result.get("completion_text") or "").split())
            if len(preview) > 120:
                preview = preview[:117] + "..."
            print(f"  preview={preview!r}", flush=True)
            if args.print_responses:
                print("----- COMPLETION -----", flush=True)
                print(result.get("completion_text") or "", flush=True)
                print("----- END -----", flush=True)

    print(f"Wrote {output}")
    print(f"Wrote {output.with_suffix('.csv')}")
    if args.save_responses:
        print(f"Wrote {response_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
