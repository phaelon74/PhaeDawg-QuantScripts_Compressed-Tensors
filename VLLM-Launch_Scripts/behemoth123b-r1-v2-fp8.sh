#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# TheDrummer/Behemoth-R1-123B-v2 (W8A8-FP8 BLOCK) on 2x RTX 6000 Pro
# Workstation (Blackwell, SM120) via vLLM
# ============================================================
#
# Model:
#   - Base:    TheDrummer/Behemoth-R1-123B-v2 (Mistral-Large-Instruct-2411
#              fine-tune, arch: MistralForCausalLM, 88 layers, 123B params).
#   - Quant:   compressed-tensors W8A8-FP8 block quant (128x128 blocks,
#              dynamic FP8 activations). quantization_config.quant_method=
#              compressed-tensors, format=float-quantized. Only lm_head kept
#              in BF16. On-disk checkpoint ~= 115 GiB.
#
# VRAM fit on 2x 96 GiB Blackwell (TP=2, PCIe only):
#   - FP8 weights resident ~= 115 GiB total -> ~58 GiB / GPU after TP shard.
#   - KV cache (BF16, 88 layers, 8 KV heads, head_dim=128):
#       per-token KV ~= 352 KiB/token total -> ~22 GiB / GPU at 128K ctx, 1 seq.
#   - Default below (64K ctx, 2 seqs) leaves headroom for graphs and activations.
#
# This script does not store API keys. Pass the key as the first argument, or
# set VLLM_API_KEY in the environment.
#
# GPU selection:
#   - Both RTX 6000 Pro cards are exposed by default (logical CUDA 0,1).
#   - Override with CUDA_VISIBLE_DEVICES if your topology differs.
# ============================================================

# --- Memory Management ---
export PYTORCH_ALLOC_CONF=expandable_segments:True

# --- GPU Selection (dual Blackwell workstation cards) ---
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# --- B12X gates: OFF. This is the stock compressed-tensors FP8 path. ---
export B12X_ENABLE_FP6=0

# --- vLLM / CUDA behavior ---
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_SLEEP_WHEN_IDLE=1
export VLLM_MARLIN_USE_ATOMIC_ADD=1
export VLLM_USE_FLASHINFER_SAMPLER=0

# --- Loading / CPU threading ---
export SAFETENSORS_FAST_GPU=1
export OMP_NUM_THREADS=8

# --- Torch profiler (registers POST /start_profile and /stop_profile) ---
PROFILE="${PROFILE:-0}"
PROFILE_DIR="${PROFILE_DIR:-/tmp/vllm_prof}"
if [[ "$PROFILE" == "1" ]]; then
  mkdir -p "$PROFILE_DIR"
  export VLLM_TORCH_PROFILER_DIR="$PROFILE_DIR"
fi

# ============================================================
# Model / serving configuration
# ============================================================

API_KEY="${1:-${VLLM_API_KEY:-}}"

if [[ -z "$API_KEY" ]]; then
  echo "Usage: $0 API-KEY-HERE" >&2
  echo "       or: VLLM_API_KEY=API-KEY-HERE $0" >&2
  exit 2
fi

# Local W8A8-FP8 block compressed-tensors checkpoint. Override at launch:
#   MODEL_DIR=/path/to/model ./behemoth123b-r1-v2-fp8.sh
MODEL_DIR="${MODEL_DIR:-/media/fmodels/TheHouseOfTheDude/Behemoth-R1-123B-v2_W8A8-FP8-BLOCK}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Behemoth-R1-123B-v2-W8A8-FP8-BLOCK}"
# The compressed-tensors checkpoint includes tokenizer.json/tokenizer_config.json
# and chat_template.jinja, so keep tokenizer resolution local by default.
TOKENIZER="${TOKENIZER:-$MODEL_DIR}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"

# Two GPUs -> tensor parallelism across both cards.
TP_SIZE="${TP_SIZE:-2}"

# ------------------------------------------------------------
# Capacity sizing for 2x 96 GiB Blackwell at TP=2.
# FP8 weights are larger than FP6; keep MAX_NUM_SEQS modest at long ctx.
# ------------------------------------------------------------
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

# Blackwell handles fp8 KV cache well; auto (BF16) is the safer default.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

TOKENIZER_MODE="${TOKENIZER_MODE:-hf}"
CONFIG_FORMAT="${CONFIG_FORMAT:-auto}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"

# compressed-tensors W8A8-FP8 block quant. vLLM CLI spelling uses a hyphen.
QUANTIZATION="${QUANTIZATION:-compressed-tensors}"

# Stock Mistral architecture; no custom modeling code needed.
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"

# ============================================================
# Behemoth / Mistral feature toggles
# ============================================================
# Behemoth-R1 can reason, but this checkpoint does not define <think>
# tokens. vLLM's deepseek_r1 parser hard-fails without them -- leave off.
REASONING_PARSER="${REASONING_PARSER:-}"

# Tool calling off by default; flip ENABLE_TOOL_CALLING=1 if needed.
ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-0}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-mistral}"

ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"

# CUDA graph setup:
# - full_decode_only keeps CUDA graphs on the decode path and avoids the
#   highest-memory full-prefill captures.
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-full_decode_only}"
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-[1,2,4]}"
if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
  COMPILATION_CONFIG="{\"mode\":3,\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\",\"cudagraph_capture_sizes\":${CUDAGRAPH_CAPTURE_SIZES}}"
fi

MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"

# Eager escape hatch (debug only; disables torch.compile + CUDA graphs).
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

# ============================================================
# Assemble the argument list
# ============================================================

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
)

if [[ "$ENFORCE_EAGER" == "1" ]]; then
  VLLM_ARGS+=(--enforce-eager)
else
  VLLM_ARGS+=(--compilation-config "$COMPILATION_CONFIG")
fi

# Blackwell sm_120: vLLM's custom all-reduce kernel crashes during CUDA graph
# capture at TP>1. Force the NCCL fallback whenever tensor parallelism is on.
if [[ "$TP_SIZE" -gt 1 ]]; then
  VLLM_ARGS+=(--disable-custom-all-reduce)
fi

if [[ "$PROFILE" == "1" ]]; then
  VLLM_ARGS+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILE_DIR}\"}")
fi

if [[ -n "$QUANTIZATION" ]]; then
  VLLM_ARGS+=(--quantization "$QUANTIZATION")
fi

if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  VLLM_ARGS+=(--trust-remote-code)
fi

if [[ -n "$REASONING_PARSER" ]]; then
  VLLM_ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi

if [[ "$ENABLE_TOOL_CALLING" == "1" ]]; then
  VLLM_ARGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi

if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
  VLLM_ARGS+=(--enable-prefix-caching)
fi

if [[ -n "$MAX_CUDAGRAPH_CAPTURE_SIZE" ]]; then
  VLLM_ARGS+=(--max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE")
fi

echo "Launching vLLM with:"
echo "  MODEL_DIR=${MODEL_DIR}"
echo "  SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "  TOKENIZER=${TOKENIZER}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  B12X_ENABLE_FP6=0 (compressed-tensors FP8 path)"
echo "  TP_SIZE=${TP_SIZE}"
echo "  MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "  KV_CACHE_DTYPE=${KV_CACHE_DTYPE}"
echo "  QUANTIZATION=${QUANTIZATION}"
echo "  TOOL_CALLING=${ENABLE_TOOL_CALLING} (parser=${TOOL_CALL_PARSER})"
echo "  PREFIX_CACHING=${ENABLE_PREFIX_CACHING}"
if [[ "$PROFILE" == "1" ]]; then
  echo "  PROFILER=torch -> ${PROFILE_DIR} (POST /start_profile + /stop_profile)"
else
  echo "  PROFILER=disabled (PROFILE=1 to enable)"
fi
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  echo "  ENFORCE_EAGER=1 (torch.compile + CUDA graphs DISABLED)"
else
  echo "  COMPILATION_CONFIG=${COMPILATION_CONFIG}"
fi
echo

vllm serve "${VLLM_ARGS[@]}"
