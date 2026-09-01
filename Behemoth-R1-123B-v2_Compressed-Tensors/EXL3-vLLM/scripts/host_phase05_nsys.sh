#!/usr/bin/env bash
# Capture one live TP4 decode under nsys. x86 nsys cannot attach to a running
# process for CUDA traces, so this RESTARTS serve via `nsys launch`.
#
# Stop any existing vLLM on GPUs 1-4 first.
#   source /home/phaedawg/exl3vllm/venv/bin/activate
#   export VLLM_API_KEY=...
#   bash "$EXL3/scripts/host_phase05_nsys.sh"
set -euo pipefail

EXL3="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${EXL3_PHASE05_OUT:-$EXL3/results/phase0_5}"
LAUNCH="${LAUNCH:-$(cd "$EXL3/../../VLLM-Launch_Scripts" && pwd)/behemoth123b-r1-v2-exl3-4p25.sh}"
HOST="${BENCH_HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MODEL="${EXL3_SERVED_MODEL_NAME:-Behemoth-R1-123B-v2-EXL3-4.25-H6}"
DECODE_TOKENS="${PHASE05_DECODE_TOKENS:-32}"
API_KEY="${VLLM_API_KEY:-${1:-}}"

if [[ -z "$API_KEY" ]]; then
  echo "Set VLLM_API_KEY or pass it as \$1" >&2
  exit 2
fi
if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys not on PATH" >&2
  exit 2
fi

mkdir -p "$OUT"
export VLLM_EXL3_NVTX=1
HELP="$(nsys profile --help 2>&1 || true)"
LAUNCH_HELP="$(nsys launch --help 2>&1 || true)"

TRACE="cuda,nvtx"
if echo "$HELP$LAUNCH_HELP" | grep -qi nccl; then
  TRACE="cuda,nvtx,nccl"
fi

echo "nsys launch serve (load will take several minutes)..."
# Space form: this nsys rejects --target-processes=all
TARGET=()
if echo "$LAUNCH_HELP" | grep -q -- '--target-processes'; then
  TARGET=(--target-processes all)
elif echo "$LAUNCH_HELP" | grep -qE -- '-y,|--target-processes'; then
  TARGET=(-y all)
fi

nsys launch -t "$TRACE" "${TARGET[@]}" bash "$LAUNCH" "$API_KEY" \
  >"$OUT/nsys_serve.log" 2>&1 &
SERVE_PID=$!

echo "waiting for http://$HOST:$PORT/v1/models ..."
ready=0
for _ in $(seq 1 120); do
  if curl -sf "http://$HOST:$PORT/v1/models" -H "Authorization: Bearer $API_KEY" >/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then
    echo "serve exited early; see $OUT/nsys_serve.log" >&2
    tail -n 40 "$OUT/nsys_serve.log" >&2 || true
    exit 1
  fi
  sleep 5
done
if [[ "$ready" != 1 ]]; then
  echo "server did not become ready in 10 minutes" >&2
  exit 1
fi

echo "nsys start + 32-token english decode"
nsys start || nsys start --session=default || true
python "$EXL3/scripts/bench_serving_contexts.py" \
  --host "$HOST" --port "$PORT" --api-key "$API_KEY" \
  --model "$MODEL" \
  --contexts 512 --output-tokens "$DECODE_TOKENS" --runs 1 \
  --warmup-runs 0 --prompt-style english \
  --output "$OUT/nsys_probe_bench.json"
nsys stop -o "$OUT/nsys_serve_decode" || nsys stop --export="$OUT/nsys_serve_decode" || true

kill "$SERVE_PID" || true
wait "$SERVE_PID" 2>/dev/null || true

REP="$(ls -1 "$OUT"/nsys_serve_decode*.nsys-rep "$OUT"/report*.nsys-rep 2>/dev/null | head -n 1 || true)"
if [[ -z "${REP:-}" ]]; then
  echo "no .nsys-rep under $OUT; check nsys stop output and $OUT/nsys_serve.log" >&2
  exit 1
fi
nsys stats --report cuda_gpu_kern_sum --format csv -o "$OUT/kern_sum" "$REP" || true
CSV="$(ls -1 "$OUT"/kern_sum*.csv 2>/dev/null | head -n 1 || true)"
if [[ -n "${CSV:-}" ]]; then
  python "$EXL3/scripts/summarize_nsys.py" "$CSV" \
    --decode-tokens "$DECODE_TOKENS" \
    --serve-log "$OUT/nsys_serve.log" \
    --output "$OUT/serving_attribution.json"
fi
echo "Wrote $OUT/serving_attribution.json"
