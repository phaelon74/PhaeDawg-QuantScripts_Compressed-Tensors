#!/usr/bin/env bash
# Phase 0: zero-code env sweep, power A/B, GB/s microbench, ncu of M=1 gate.
# Physical GPUs 0 and 5 are reserved. Stop serve before this script.
#
# Usage (from the Linux host, after git pull):
#   export EXL3=/home/phaedawg/kld-exl3-vllm/PhaeDawg-QuantScripts_Compressed-Tensors/Behemoth-R1-123B-v2_Compressed-Tensors/EXL3-vLLM
#   source /home/phaedawg/kld-exl3-vllm/kld-exl3-vllm/bin/activate
#   bash "$EXL3/scripts/host_phase0.sh"
set -euo pipefail

EXL3="${EXL3:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LAUNCH="${LAUNCH:-$(cd "$EXL3/../../VLLM-Launch_Scripts" && pwd)/behemoth123b-r1-v2-exl3-4p25.sh}"
OUT="${EXL3_PHASE0_OUT:-$EXL3/results/phase0}"
HOST="${BENCH_HOST:-10.9.99.22}"
PORT="${PORT:-8000}"
MODEL="${EXL3_SERVED_MODEL_NAME:-Behemoth-R1-123B-v2-EXL3-4.25-H6}"
API_KEY="${VLLM_API_KEY:-}"
GPUS="${PHASE0_GPUS:-1,2,3,4}"
LOGICAL_DEV="${PHASE0_LOGICAL_DEVICE:-0}"

mkdir -p "$OUT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export VLLM_EXL3_EXT_PATH="${VLLM_EXL3_EXT_PATH:-$EXL3/build/exllamav3_ext}"
export VLLM_EXL3_SKIP_VERSION_GUARD=1
export PYTHONPATH="$VLLM_EXL3_EXT_PATH:$EXL3/plugin/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")'):${LD_LIBRARY_PATH:-}"

echo "EXL3=$EXL3"
echo "OUT=$OUT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo "== microbench M=1,2,4,8 3inst GB/s =="
python "$EXL3/scripts/kernel_microbench.py" \
  --device "$LOGICAL_DEV" \
  --bitrates 4,5,6 \
  --m 1,2,4,8 \
  --output "$OUT/kernel_microbench_3inst_m1248.json"

echo "== ncu M=1 gate (INT pipe + DRAM) =="
if command -v ncu >/dev/null 2>&1; then
  ncu --set full --target-processes all \
    --metrics sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct \
    --kernel-name-base demangled \
    --launch-count 20 \
    -o "$OUT/ncu_gate_m1" --force-overwrite \
    python "$EXL3/scripts/profile_decode_nsys.py" \
      --device "$LOGICAL_DEV" --m 1 --bitrate 4 --codebook 3inst --warmup 2 --iters 8 \
    || echo "ncu failed; continuing"
else
  echo "ncu not on PATH; skip. Install Nsight Compute or run scripts/profile_ncu_gate.sh"
fi

if [[ -z "$API_KEY" ]]; then
  echo "VLLM_API_KEY unset: skipping serving env sweep and power A/B."
  echo "Microbench JSON is at $OUT/kernel_microbench_3inst_m1248.json"
  exit 0
fi

bench() {
  local label="$1"
  python "$EXL3/scripts/bench_serving_contexts.py" \
    --host "$HOST" --port "$PORT" --api-key "$API_KEY" \
    --model "$MODEL" \
    --contexts 1024 --output-tokens 256 --runs 3 \
    --label "$label" \
    --output "$OUT/${label}.json"
}

serve_and_bench() {
  local label="$1"
  shift
  echo "== serve $label =="
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

echo "== env sweep (stop any existing serve first) =="
serve_and_bench "phase0-baseline"
serve_and_bench "phase0-gemv2" EXL3_GEMV=2
serve_and_bench "phase0-gemv0" EXL3_GEMV=0
serve_and_bench "phase0-gemv-smem1" EXL3_GEMV=2 EXL3_GEMV_SMEM=1
serve_and_bench "phase0-int8-2" EXL3_INT8_GEMV=2
serve_and_bench "phase0-int8-maxk6" EXL3_INT8_GEMV=2 EXL3_INT8_GEMV_MAX_K=6
serve_and_bench "phase0-int8-maxk8" EXL3_INT8_GEMV=2 EXL3_INT8_GEMV_MAX_K=8

echo "== power A/B on physical $GPUS (never touch 0 or 5) =="
IFS=',' read -r -a GPU_ARR <<<"$GPUS"
set_pl() {
  local watts="$1"
  for g in "${GPU_ARR[@]}"; do
    nvidia-smi -i "$g" -pl "$watts" || true
  done
}
set_pl 350
serve_and_bench "phase0-pl350"
set_pl 270
serve_and_bench "phase0-pl270"

echo "Phase 0 receipts under $OUT"
echo "Gate G0: any 1K/256 decode >= 21 tok/s, else documented no with ncu + GB/s table."
