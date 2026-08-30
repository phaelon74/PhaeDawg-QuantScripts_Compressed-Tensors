#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Behemoth-R1-123B-v2 AutoRound GS32 mixed W4A16/W8A16
# compressed-tensors/Marlin on physical GPUs 1,2,3,4 (TP4)
# ============================================================

export PYTORCH_ALLOC_CONF=expandable_segments:True

# Physical GPUs 0 and 5 are reserved by other services.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2,3,4

export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_SLEEP_WHEN_IDLE=1
export VLLM_MARLIN_USE_ATOMIC_ADD=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export SAFETENSORS_FAST_GPU=1
export OMP_NUM_THREADS=8

# Do not inherit EXL3 plugin settings into the Marlin process. An empty
# allowlist loads no third-party general plugins; an unset value loads all.
export VLLM_PLUGINS=""
unset VLLM_EXL3_EXT_PATH
unset VLLM_EXL3_QUANT_CONFIG
unset VLLM_EXL3_CROSSOVER_JSON
unset VLLM_EXL3_ALLOW_GRAPHS
unset VLLM_EXL3_ALLOW_VLLM_DRIFT
unset VLLM_EXL3_SKIP_VERSION_GUARD
unset VLLM_EXL3_FORCE_COMPRESSED
unset VLLM_EXL3_FORCE_RECONSTRUCT

API_KEY="${1:-${VLLM_API_KEY:-}}"
if [[ -z "$API_KEY" ]]; then
  echo "Usage: $0 API-KEY-HERE" >&2
  echo "       or: VLLM_API_KEY=API-KEY-HERE $0" >&2
  exit 2
fi

# Backend-specific names prevent stale MODEL_DIR/TOKENIZER exports from an
# EXL3 or conversion session from silently selecting the wrong checkpoint.
MODEL_DIR="${AUTOROUND_MODEL_DIR:-/media/fmodels/TheHouseOfTheDude/Behemoth-R1-123B-v2/AutoRound_GS32_Mixed_69p5G}"
SERVED_MODEL_NAME="${AUTOROUND_SERVED_MODEL_NAME:-Behemoth-R1-123B-v2-AutoRound-GS32-Mixed-69p5G}"
TOKENIZER="${AUTOROUND_TOKENIZER:-$MODEL_DIR}"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Missing model config: $MODEL_DIR/config.json" >&2
  exit 2
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-54272}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
TOKENIZER_MODE="${TOKENIZER_MODE:-hf}"
CONFIG_FORMAT="${CONFIG_FORMAT:-auto}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"

# This checkpoint carries compressed-tensors metadata for mixed group-size-32
# AutoRound weights. vLLM dispatches supported layers through Marlin.
QUANTIZATION="${QUANTIZATION:-compressed-tensors}"

REASONING_PARSER="${REASONING_PARSER:-}"
REASONING_CONFIG="${REASONING_CONFIG:-}"
ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-0}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-mistral}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"

# Preserve the currently validated compiled decode configuration.
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-full_decode_only}"
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-[1,2,4]}"
if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
  COMPILATION_CONFIG="{\"mode\":3,\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\",\"cudagraph_capture_sizes\":${CUDAGRAPH_CAPTURE_SIZES}}"
fi
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"

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
  --config-format "$CONFIG_FORMAT"
  --load-format "$LOAD_FORMAT"
  --dtype auto
  --quantization "$QUANTIZATION"
  --compilation-config "$COMPILATION_CONFIG"
  --disable-custom-all-reduce
)

if [[ "$ENABLE_TOOL_CALLING" == "1" ]]; then
  VLLM_ARGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi
if [[ -n "$REASONING_PARSER" ]]; then
  VLLM_ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi
if [[ -n "$REASONING_CONFIG" ]]; then
  VLLM_ARGS+=(--reasoning-config "$REASONING_CONFIG")
fi
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  VLLM_ARGS+=(--trust-remote-code)
fi
if [[ -n "$MAX_CUDAGRAPH_CAPTURE_SIZE" ]]; then
  VLLM_ARGS+=(--max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE")
fi

echo "Launching AutoRound compressed-tensors/Marlin:"
echo "  MODEL_DIR=$MODEL_DIR"
echo "  SERVED_MODEL_NAME=$SERVED_MODEL_NAME"
echo "  QUANTIZATION=$QUANTIZATION"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  TP_SIZE=$TP_SIZE"
echo "  MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "  GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
echo "  COMPILATION_CONFIG=$COMPILATION_CONFIG"
echo

vllm serve "${VLLM_ARGS[@]}"
