#!/usr/bin/env bash
# Serve Behemoth EXL3 on 4x RTX 3090 via the out-of-tree vLLM plugin.
# Derived from VLLM-Launch_Scripts/behemoth123b-r1-v2.sh with Marlin-only
# environment variables removed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_SLEEP_WHEN_IDLE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export SAFETENSORS_FAST_GPU=1
export OMP_NUM_THREADS=8

# Plugin + extension pins
export VLLM_PLUGINS="${VLLM_PLUGINS:-vllm_exl3_sm86}"
export VLLM_EXL3_EXT_PATH="${VLLM_EXL3_EXT_PATH:-$ROOT/build/exllamav3_ext}"
unset VLLM_MARLIN_USE_ATOMIC_ADD

API_KEY="${1:-${VLLM_API_KEY:-}}"
if [[ -z "$API_KEY" ]]; then
  echo "Usage: $0 API-KEY-HERE" >&2
  exit 2
fi

MODEL_DIR="${MODEL_DIR:-/media/fmodels/ArtusDev/TheDrummer_Behemoth-R1-123B-v2-EXL3/3.5bpw_H6}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Behemoth-R1-123B-v2-EXL3-3.5-H6}"
TOKENIZER="${TOKENIZER:-$MODEL_DIR}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-54272}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
TOKENIZER_MODE="${TOKENIZER_MODE:-hf}"
QUANTIZATION="${QUANTIZATION:-exl3}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"

VLLM_ARGS=(
  "$MODEL_DIR"
  --served-model-name "$SERVED_MODEL_NAME"
  --api-key "$API_KEY"
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size "$TP_SIZE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --tokenizer "$TOKENIZER"
  --tokenizer-mode "$TOKENIZER_MODE"
  --dtype auto
  --quantization "$QUANTIZATION"
  --disable-custom-all-reduce
)

if [[ "$ENFORCE_EAGER" == "1" ]]; then
  VLLM_ARGS+=(--enforce-eager)
else
  CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-full_decode_only}"
  CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-[1,2,3,4,5,6,8]}"
  COMPILATION_CONFIG="${COMPILATION_CONFIG:-{\"mode\":3,\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\",\"cudagraph_capture_sizes\":${CUDAGRAPH_CAPTURE_SIZES}}}"
  export VLLM_EXL3_ALLOW_GRAPHS=1
  VLLM_ARGS+=(--compilation-config "$COMPILATION_CONFIG")
fi

echo "Launching EXL3 vLLM"
echo "  MODEL_DIR=$MODEL_DIR"
echo "  VLLM_PLUGINS=$VLLM_PLUGINS"
echo "  VLLM_EXL3_EXT_PATH=$VLLM_EXL3_EXT_PATH"
echo "  ENFORCE_EAGER=$ENFORCE_EAGER"
echo "  TP_SIZE=$TP_SIZE"
echo "  MAX_MODEL_LEN=$MAX_MODEL_LEN"

vllm serve "${VLLM_ARGS[@]}"
