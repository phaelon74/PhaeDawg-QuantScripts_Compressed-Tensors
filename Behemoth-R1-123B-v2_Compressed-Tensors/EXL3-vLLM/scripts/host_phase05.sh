#!/usr/bin/env bash
# Phase 0.5 of exl3-int4-speed-decode:
#   1. Parse pinned exl3_mgemm ABI (size_n_list / c_ptrs / scalar K)
#   2. Re-inventory with per-layer bitrates
#   3. Re-budget with exact k/v pairing
#   4. Native (ext.*) mgemm vs N gemm for gate/up, k/v, true-width qkv
#   5. Optional wrapper A/B on k/v K6 to quantify dispatch tax
#   6. Optional nsys attach to a live TP4 server
#
# Stop vLLM on GPUs 1-4 before steps 4-5. Step 6 needs the server running.
# Physical 0 and 5 stay reserved.
#
#   source /home/phaedawg/exl3vllm/venv/bin/activate
#   bash "$EXL3/scripts/host_phase05.sh"
#   SKIP_GPU=1 bash "$EXL3/scripts/host_phase05.sh"   # ABI + inventory only
#   SKIP_NSYS=1 bash "$EXL3/scripts/host_phase05.sh"  # skip serving attach
set -euo pipefail

EXL3="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_4P25="${EXL3_MODEL_DIR:-/media/fmodels/ArtusDev/TheDrummer_Behemoth-R1-123B-v2-EXL3/4.25bpw_H6}"
OUT="${EXL3_PHASE05_OUT:-$EXL3/results/phase0_5}"
MICRO="${EXL3_MICROBENCH:-$EXL3/results/phase0_mixedk/roofline_baseline_m1.json}"
LAUNCH="${LAUNCH:-$(cd "$EXL3/../../VLLM-Launch_Scripts" && pwd)/behemoth123b-r1-v2-exl3-4p25.sh}"
HOST="${BENCH_HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MODEL="${EXL3_SERVED_MODEL_NAME:-Behemoth-R1-123B-v2-EXL3-4.25-H6}"
DECODE_TOKENS="${PHASE05_DECODE_TOKENS:-32}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export VLLM_EXL3_EXT_PATH="${VLLM_EXL3_EXT_PATH:-$EXL3/build/exllamav3_ext}"
export VLLM_EXL3_SKIP_VERSION_GUARD=1
export PYTHONPATH="$VLLM_EXL3_EXT_PATH:$EXL3/plugin/src${PYTHONPATH:+:$PYTHONPATH}"
if command -v python >/dev/null 2>&1; then
  TORCH_LIB="$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")' 2>/dev/null || true)"
  if [[ -n "${TORCH_LIB:-}" ]]; then
    export LD_LIBRARY_PATH="$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi

mkdir -p "$OUT"

echo "EXL3=$EXL3"
echo "OUT=$OUT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo "=== 1. pinned exl3_mgemm ABI ==="
python "$EXL3/scripts/mgemm_abi.py" --skip-python --output "$OUT/mgemm_abi.json" || true
python "$EXL3/scripts/mgemm_abi.py" --output "$OUT/mgemm_abi.json"

echo "=== 2. ArtusDev 4.25 inventory with per-layer bitrates ==="
python "$EXL3/scripts/validate_exl3_checkpoint.py" "$MODEL_4P25" \
  --profile artusdev-4p25 \
  --sha-manifest "$OUT/artusdev_4p25_inventory.json"

echo "=== 3. mixed-K budget with fusion pairing ==="
if [[ -f "$MICRO" ]]; then
  python "$EXL3/scripts/decode_latency_budget.py" \
    --microbench "$MICRO" \
    --inventory "$OUT/artusdev_4p25_inventory.json" \
    --target-tok-s 30 \
    --output "$OUT/decode_latency_budget.json"
else
  echo "missing $MICRO; skip budget (copy roofline_baseline_m1.json or re-run kernel_microbench)"
fi

if [[ "${SKIP_GPU:-0}" != "1" ]]; then
  echo "=== 4. native mgemm (stop vLLM on 1-4 first) ==="
  python "$EXL3/scripts/mgemm_microbench.py" \
    --device 0 --path native --shapes gate_up --bitrate 4 --m 1,2,4 \
    --output "$OUT/mgemm_native_gate_up_k4.json"
  python "$EXL3/scripts/mgemm_microbench.py" \
    --device 0 --path native --shapes kv --bitrate 4 --m 1,2,4 \
    --output "$OUT/mgemm_native_kv_k4.json"
  python "$EXL3/scripts/mgemm_microbench.py" \
    --device 0 --path native --shapes kv --bitrate 6 --m 1,2,4 \
    --output "$OUT/mgemm_native_kv_k6.json"
  python "$EXL3/scripts/mgemm_microbench.py" \
    --device 0 --path native --shapes qkv --bitrate 4 --m 1,2,4 \
    --output "$OUT/mgemm_native_qkv_widths_k4.json" \
    || echo "true-width qkv mgemm failed; ABI or launch error is in the log"
  echo "=== 5. wrapper A/B on k/v K6 (dispatch tax) ==="
  python "$EXL3/scripts/mgemm_microbench.py" \
    --device 0 --path wrapper --shapes kv --bitrate 6 --m 1,2,4 \
    --output "$OUT/mgemm_wrapper_kv_k6.json"
fi

if [[ "${SKIP_NSYS:-0}" == "1" ]]; then
  echo "SKIP_NSYS=1: not attaching nsys."
  echo "Phase 0.5 receipts under $OUT"
  exit 0
fi

echo "=== 6. nsys attach to live TP4 serve ==="
if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys not on PATH. After a capture, run:"
  echo "  nsys stats --report cuda_gpu_kern_sum --format csv -o $OUT/kern_sum <rep>"
  echo "  python $EXL3/scripts/summarize_nsys.py $OUT/kern_sum.csv --decode-tokens $DECODE_TOKENS --output $OUT/serving_attribution.json"
  exit 0
fi

PARENT_PID="${VLLM_PID:-}"
if [[ -z "$PARENT_PID" ]]; then
  PARENT_PID="$(pgrep -f 'vllm.entrypoints.openai.api_server|VLLM::EngineCore' | head -n 1 || true)"
fi
if [[ -z "$PARENT_PID" ]]; then
  echo "No vLLM PID. Start serve with VLLM_EXL3_NVTX=1, then rerun:"
  echo "  SKIP_GPU=1 VLLM_PID=<pid> bash $EXL3/scripts/host_phase05.sh"
  echo "Or profile the launcher:"
  echo "  VLLM_EXL3_NVTX=1 nsys profile -t cuda,nvtx,nccl --duration=25 -o $OUT/nsys_serve_decode --target-processes=all bash $LAUNCH \"\$VLLM_API_KEY\""
  exit 0
fi

echo "attaching nsys to pid $PARENT_PID for 20s; sending a short decode..."
nsys profile \
  --duration=20 \
  --delay=2 \
  --trace=cuda,nvtx,nccl \
  --gpu-metrics-device=none \
  --force-overwrite=true \
  --target-processes=all \
  -p "$PARENT_PID" \
  -o "$OUT/nsys_serve_decode" &
NSYS_PID=$!
sleep 4
if [[ -n "${VLLM_API_KEY:-}" ]]; then
  python "$EXL3/scripts/bench_serving_contexts.py" \
    --host "$HOST" --port "$PORT" --api-key "$VLLM_API_KEY" \
    --model "$MODEL" \
    --contexts 512 --output-tokens "$DECODE_TOKENS" --runs 1 \
    --warmup-runs 0 --prompt-style english \
    --output "$OUT/nsys_probe_bench.json" || true
else
  echo "VLLM_API_KEY unset: generate tokens from another shell during the 20s window."
fi
wait "$NSYS_PID" || true

if [[ -f "$OUT/nsys_serve_decode.nsys-rep" ]]; then
  nsys stats --report cuda_gpu_kern_sum --format csv \
    -o "$OUT/kern_sum" "$OUT/nsys_serve_decode.nsys-rep" || true
  CSV="$(ls -1 "$OUT"/kern_sum*.csv 2>/dev/null | head -n 1 || true)"
  if [[ -n "${CSV:-}" ]]; then
    python "$EXL3/scripts/summarize_nsys.py" "$CSV" \
      --decode-tokens "$DECODE_TOKENS" \
      --output "$OUT/serving_attribution.json"
  fi
fi

echo "Phase 0.5 receipts under $OUT"
echo "Paste mgemm_abi.json, decode_latency_budget.json, native mgemm JSON, and serving_attribution.json."
