#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# TheHouseOfTheDude/Cydonia-24B-v4.3 (FP6) on 1x RTX PRO 6000 (Blackwell) via vLLM
# ============================================================
#
# Model:
#   - Base:    TheDrummer/Cydonia-24B-v4.3 (Mistral-Small-3.x 24B finetune,
#              arch: MistralForCausalLM — plain dense decoder, TEXT-ONLY).
#   - Quant:   B12X MX-FP6, W6A6 (6-bit weights, 6-bit activations).
#              quantization_config = {"quant_method":"modelopt","quant_algo":"W6A6"}
#              Golden-rule Linear coverage: MLP + attention projections (280
#              linears / 40 layers). Kept BF16: lm_head, embeddings, norms.
#              Served by the B12X *static* fused kernel (not the micro kernel).
#
# Architecture notes (from config.json):
#   - 40 uniform decoder layers (full attention; no GDN/SSM, no MTP, no vision).
#   - Attention: 32 Q heads, 8 KV heads, head_dim=128.
#   - hidden=5120, intermediate=32768, vocab=131072.
#   - Native ctx: 131,072 tokens.
#
# This is the model-agnosticism check for the B12X FP6 path: a plain
# transformer with none of the Qwen3.6 specials (no linear_attn fusions, no
# draft model). Expect every mlp/attn linear to bind FP6 at startup
# ("B12X FP6: bound FP6 linear ..."), lm_head/embeds to log BF16 fallback,
# and NO small-N GEMV bindings (no small bf16 projections in this arch).
#
# NOTE on throughput comparisons: no MTP here -> one token per step. Do not
# compare tok/s against the Qwen3.6 numbers (those carry a ~2.4x
# tokens-per-step multiplier from speculative decoding).
#
# VRAM fit on 1x 96 GiB Blackwell:
#   - FP6 weights ~= 19 GiB resident (export wrote ~20 GB total).
#   - KV cache (BF16, TP=1): per-token KV = 40 layers * 2 (K+V) * 8 KV heads
#     * 128 = 160 KiB/token -> 128K ctx = ~20 GiB per sequence. The default
#     gpu-memory-utilization leaves room for ~3 concurrent 128K sequences;
#     drop MAX_MODEL_LEN or KV_CACHE_DTYPE=fp8 for more concurrency.
#
# This script does not store API keys. Pass the key as the first argument, or
# set VLLM_API_KEY in the environment.
# ============================================================

# --- Memory Management ---
export PYTORCH_ALLOC_CONF=expandable_segments:True

# --- GPU Selection (single Blackwell card) ---
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# --- B12X FP6 gates ---
export B12X_ENABLE_FP6="${B12X_ENABLE_FP6:-1}"
export B12X_ENABLE_FP6_MICRO="${B12X_ENABLE_FP6_MICRO:-0}"

# --- vLLM / CUDA behavior ---
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_SLEEP_WHEN_IDLE=1
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

MODEL_DIR="${MODEL_DIR:-/media/fmodels/TheHouseOfTheDude/Cydonia-24B-v4.3/FP6}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Cydonia-24B-v4.3-FP6-W6A6}"
# Tell the B12X vLLM plugin where the FP6 weights live (inherited by workers).
# Set UNCONDITIONALLY: a stale B12X_FP6_MODEL_DIR from a previous launch (e.g.
# the Qwen3.6 dir) makes the plugin index the wrong checkpoint -> every layer
# silently falls back to BF16 -> loader KeyError on FP6 tensors (input_scale).
export B12X_FP6_MODEL_DIR="$MODEL_DIR"
TOKENIZER="${TOKENIZER:-$MODEL_DIR}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"

# Single GPU -> no tensor parallelism.
TP_SIZE="${TP_SIZE:-1}"

# ------------------------------------------------------------
# Capacity sizing (see VRAM notes above): 128K ctx costs ~20 GiB KV/seq here.
# ------------------------------------------------------------
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

# BF16 KV by default; fp8 KV doubles the token budget if you need concurrency.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

TOKENIZER_MODE="${TOKENIZER_MODE:-hf}"
CONFIG_FORMAT="${CONFIG_FORMAT:-auto}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"

# B12X MX-FP6 (W6A6); requires the b12x vLLM plugin (general_plugins entry point).
QUANTIZATION="${QUANTIZATION:-b12x_fp6}"

# Stock Mistral architecture; no custom modeling code needed.
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"

# ============================================================
# Model-feature toggles
# ============================================================
# Cydonia/Mistral specifics vs the Qwen3.6 script:
#   - NO reasoning parser: the tokenizer has no <think> tokens (the qwen3
#     parser hard-fails at startup on this model).
#   - NO MTP / speculative config: the checkpoint has no draft head.
#   - NO vision toggles: text-only architecture.
#   - Tool calling off by default; flip ENABLE_TOOL_CALLING=1 with a parser
#     if your workload needs it (Mistral-family parser: "mistral").
ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-0}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-mistral}"

# Prefix caching is generally a win for chat/agent workloads.
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"

# CUDA graph setup: full decode-only graphs; pure m=1 decode without MTP, so
# capture sizes track concurrent sequences (max_num_seqs=4).
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-full_decode_only}"
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-[1,2,4]}"
COMPILATION_CONFIG="${COMPILATION_CONFIG:-{\"mode\":3,\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\",\"cudagraph_capture_sizes\":${CUDAGRAPH_CAPTURE_SIZES}}}"

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

if [[ "$PROFILE" == "1" ]]; then
  VLLM_ARGS+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILE_DIR}\"}")
fi

if [[ -n "$QUANTIZATION" ]]; then
  VLLM_ARGS+=(--quantization "$QUANTIZATION")
fi

if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  VLLM_ARGS+=(--trust-remote-code)
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
echo "  B12X_ENABLE_FP6=${B12X_ENABLE_FP6} (micro=${B12X_ENABLE_FP6_MICRO})"
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
