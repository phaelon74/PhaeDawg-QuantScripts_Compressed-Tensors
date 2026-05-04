#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# TheDrummer/Behemoth-R1-123B-v2 on 4x RTX 3090 via vLLM
# ============================================================
#
# IMPORTANT FIT NOTE:
# - The upstream Hugging Face model is BF16, 123B parameters.
# - BF16 weights alone require roughly 245+ GiB before KV cache and runtime
#   overhead, so the raw HF checkpoint will NOT fit on 4x 24 GiB RTX 3090s.
# - This script defaults to your local W4A16 compressed-tensors checkpoint.
#
# This script intentionally does not store API keys. Pass the key as the first
# argument, or set VLLM_API_KEY in the environment.
#
# GPU selection:
# - Physical nvidia-smi GPUs 1,2,3,4 are exposed to the process.
# - Inside vLLM they appear as logical CUDA devices 0,1,2,3.
# ============================================================

# --- Memory Management ---
export PYTORCH_ALLOC_CONF=expandable_segments:True

# --- GPU Selection ---
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2,3,4

# --- vLLM / CUDA behavior ---
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_SLEEP_WHEN_IDLE=1
export VLLM_MARLIN_USE_ATOMIC_ADD=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export disable_custom_all_reduce=True

# --- Loading / CPU threading ---
export SAFETENSORS_FAST_GPU=1
export OMP_NUM_THREADS=8
# --- NCCL notes for consumer PCIe GPUs ---
# Uncomment/tune only if you see NCCL hangs or poor peer-to-peer behavior.
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_LEVEL=PHB
# export NCCL_MIN_NCHANNELS=8

# ============================================================
# Model / serving configuration
# ============================================================

API_KEY="${1:-${VLLM_API_KEY:-}}"

if [[ -z "$API_KEY" ]]; then
  echo "Usage: $0 API-KEY-HERE" >&2
  echo "       or: VLLM_API_KEY=API-KEY-HERE $0" >&2
  exit 2
fi

# Local quantized checkpoint that fits in 4x24 GiB.
# Override at launch time if needed:
#   MODEL_DIR=/path/to/model ./behemoth-r1-123b-v2-4x3090.sh
MODEL_DIR="${MODEL_DIR:-/media/fmodels/TheHouseOfTheDude/Behemoth-R1-123B-v2_Compressed-Tensors/W4A16_GS32}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Behemoth-R1-123B-v2-W4A16-GS32}"
# The compressed-tensors checkpoint includes tokenizer.json/tokenizer_config.json
# and chat_template.jinja, so keep tokenizer resolution local by default.
TOKENIZER="${TOKENIZER:-$MODEL_DIR}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-54272}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

# Keep KV cache compact for 32k context on 4x 3090.
# Use auto by default because some 3090/vLLM combinations reject fp8 KV cache.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

# Use the local Hugging Face tokenizer files from TOKENIZER. tokenizer-mode=mistral
# expects Mistral-common tokenizer assets and can report "No tokenizer file found"
# even when tokenizer.json exists.
TOKENIZER_MODE="${TOKENIZER_MODE:-hf}"
CONFIG_FORMAT="${CONFIG_FORMAT:-auto}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"

# Your checkpoint is compressed-tensors W4A16.
# vLLM CLI spelling uses a hyphen: compressed-tensors.
QUANTIZATION="${QUANTIZATION:-compressed-tensors}"

# Behemoth-R1 can reason, but this checkpoint does not define <think> and
# </think> as tokenizer vocabulary entries. vLLM's built-in deepseek_r1 parser
# requires those entries and fails during startup without them. Leave parser
# extraction off by default; the model can still emit reasoning text in content.
REASONING_PARSER="${REASONING_PARSER:-}"
# Latest docs list --reasoning-config, but your freshly compiled binary rejects
# it. Leave empty by default; set this only when vllm serve --help confirms it.
REASONING_CONFIG="${REASONING_CONFIG:-}"

# PhantomForge uses JSON schema / response_format, not tool calls. Keep tool
# calling disabled by default to avoid parser interactions. Enable only if you
# explicitly need OpenAI tool calling.
ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-0}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-mistral}"

# CUDA graph setup:
# - full_decode_only keeps CUDA graphs on the decode path and avoids the
#   highest-memory full-prefill captures.
# - capture sizes [1,2,4] are the intentionally small subset requested.
#   If your vLLM version rejects cudagraph_capture_sizes in JSON, remove that
#   key or fall back to MAX_CUDAGRAPH_CAPTURE_SIZE below.
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-full_decode_only}"
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-[1,2,4]}"
COMPILATION_CONFIG="${COMPILATION_CONFIG:-{\"mode\":3,\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\",\"cudagraph_capture_sizes\":${CUDAGRAPH_CAPTURE_SIZES}}}"

# Fallback for older vLLM versions that do not understand cudagraph_capture_sizes
# inside --compilation-config. Leave empty unless needed.
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"

# trust-remote-code is not needed for most Mistral checkpoints. Enable only if
# your local quantized checkpoint requires it.
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"

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
  --compilation-config "$COMPILATION_CONFIG"
  --disable-custom-all-reduce
)

if [[ -n "$QUANTIZATION" ]]; then
  VLLM_ARGS+=(--quantization "$QUANTIZATION")
fi

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

echo "Launching vLLM with:"
echo "  MODEL_DIR=${MODEL_DIR}"
echo "  SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "  TOKENIZER=${TOKENIZER}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  TP_SIZE=${TP_SIZE}"
echo "  MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "  KV_CACHE_DTYPE=${KV_CACHE_DTYPE}"
if [[ -n "$REASONING_PARSER" ]]; then
  echo "  REASONING_PARSER=${REASONING_PARSER}"
fi
if [[ -n "$REASONING_CONFIG" ]]; then
  echo "  REASONING_CONFIG=${REASONING_CONFIG}"
fi
echo "  COMPILATION_CONFIG=${COMPILATION_CONFIG}"
echo

vllm serve "${VLLM_ARGS[@]}"

