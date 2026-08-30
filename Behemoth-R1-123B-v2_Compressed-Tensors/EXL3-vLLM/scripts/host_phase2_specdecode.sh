#!/usr/bin/env bash
# Phase 2 spec-decode A/B: n-gram first, then optional draft model.
# Stop any existing serve. Physical GPUs 1-4 only.
#
#   export EXL3=...
#   export VLLM_API_KEY=...
#   bash "$EXL3/scripts/host_phase2_specdecode.sh"
set -euo pipefail
EXL3="${EXL3:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LAUNCH="${LAUNCH:-$(cd "$EXL3/../../VLLM-Launch_Scripts" && pwd)/behemoth123b-r1-v2-exl3-4p25.sh}"
OUT="${EXL3_PHASE2_OUT:-$EXL3/results/phase2}"
HOST="${BENCH_HOST:-10.9.99.22}"
PORT="${PORT:-8000}"
MODEL="${EXL3_SERVED_MODEL_NAME:-Behemoth-R1-123B-v2-EXL3-4.25-H6}"
API_KEY="${VLLM_API_KEY:?set VLLM_API_KEY}"
mkdir -p "$OUT"

bench() {
  local label="$1"
  python "$EXL3/scripts/bench_serving_contexts.py" \
    --host "$HOST" --port "$PORT" --api-key "$API_KEY" \
    --model "$MODEL" --contexts 1024,4096 --output-tokens 256 --runs 3 \
    --label "$label" --output "$OUT/${label}.json"
}

serve_and_bench() {
  local label="$1"
  shift
  env "$@" bash "$LAUNCH" "$API_KEY" >"$OUT/${label}.serve.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 90); do
    if curl -sf "http://$HOST:$PORT/v1/models" -H "Authorization: Bearer $API_KEY" >/dev/null; then
      break
    fi
    sleep 5
  done
  bench "$label" || true
  kill "$pid" || true
  wait "$pid" 2>/dev/null || true
  sleep 8
}

serve_and_bench "phase2-no-spec"
serve_and_bench "phase2-ngram3" EXL3_NGRAM_SPEC=1 EXL3_NGRAM_TOKENS=3
serve_and_bench "phase2-ngram5" EXL3_NGRAM_SPEC=1 EXL3_NGRAM_TOKENS=5 EXL3_NGRAM_LOOKUP_MAX=8

if [[ -n "${EXL3_DRAFT_MODEL:-}" ]]; then
  python "$EXL3/scripts/verify_draft_tokenizer.py" \
    --target "${EXL3_MODEL_DIR:-/media/fmodels/ArtusDev/TheDrummer_Behemoth-R1-123B-v2-EXL3/4.25bpw_H6}" \
    --draft "$EXL3_DRAFT_MODEL"
  SPEC=$(python -c "import json,os; print(json.dumps({'model': os.environ['EXL3_DRAFT_MODEL'], 'num_speculative_tokens': int(os.environ.get('EXL3_DRAFT_TOKENS','4'))}))")
  serve_and_bench "phase2-draft" EXL3_SPECULATIVE_CONFIG="$SPEC"
fi

echo "Phase 2 receipts under $OUT"
echo "Gate G2: 1K/256 decode >= 30 tok/s with spec decode; log accepted tokens/step from vLLM."
