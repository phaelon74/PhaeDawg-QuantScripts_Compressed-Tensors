#!/usr/bin/env bash
# Clean worker restart: start, generate, kill, start again, generate.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the EXL3 checkpoint}"
PORT="${PORT:-8017}"
KEY="${VLLM_API_KEY:-restart-test-key}"

export VLLM_PLUGINS="${VLLM_PLUGINS:-vllm_exl3_sm86}"
export VLLM_EXL3_EXT_PATH="${VLLM_EXL3_EXT_PATH:-$ROOT/build/exllamav3_ext}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"

start_server() {
  ENFORCE_EAGER=1 PORT="$PORT" MODEL_DIR="$MODEL_DIR" \
    "$ROOT/scripts/serve_exl3_sm86.sh" "$KEY" >"$ROOT/results/restart_serve.log" 2>&1 &
  echo $!
}

wait_ready() {
  local i
  for i in $(seq 1 180); do
    if curl -sf -H "Authorization: Bearer $KEY" "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
      return 0
    fi
    sleep 2
  done
  return 1
}

generate() {
  curl -sf "http://127.0.0.1:$PORT/v1/completions" \
    -H "Authorization: Bearer $KEY" \
    -H "Content-Type: application/json" \
    -d '{"model":"Behemoth-R1-123B-v2-EXL3-3.5-H6","prompt":"Hello","max_tokens":8,"temperature":0}'
}

mkdir -p "$ROOT/results"
PID=$(start_server)
trap 'kill "$PID" 2>/dev/null || true' EXIT
wait_ready
generate | tee "$ROOT/results/restart_pass1.json"
kill "$PID"
wait "$PID" || true
sleep 3
PID=$(start_server)
wait_ready
generate | tee "$ROOT/results/restart_pass2.json"
echo "restart test OK"
