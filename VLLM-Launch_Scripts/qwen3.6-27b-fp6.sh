#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# TheHouseOfTheDude/Qwen3.6-27B-FP6 on 1x RTX PRO 6000 (Blackwell) via vLLM
# ============================================================
#
# Model:
#   - Base:    Qwen/Qwen3.6-27B  (arch tag: qwen3_5, ~28B params)
#   - Quant:   B12X MX-FP6, W6A6 (6-bit weights, 6-bit activations).
#              quantization_config = {"quant_method":"modelopt","quant_algo":"W6A6"}
#              Golden-rule Linear coverage: MLP + full attention + linear_attn
#              projections quantized to FP6 (this is the "_la" build that drops
#              below the FP8 size). IGNORED / kept BF16: lm_head, visual tower,
#              mtp head, embeddings, router gates, norms.
#              Served by the B12X *static* fused kernel (not the micro kernel).
#
# Architecture notes (from the upstream Qwen/Qwen3.6-27B card):
#   - Hybrid 64-layer stack:
#       16 x (3 x (Gated DeltaNet -> FFN) + 1 x (Gated Attention -> FFN))
#     i.e. 48 linear-attention (DeltaNet) layers + 16 full-attention layers.
#     KV cache is only allocated for the 16 Gated Attention layers.
#   - Gated Attention: 24 Q heads, 4 KV heads, head_dim=256, partial RoPE.
#   - Gated DeltaNet: 48 V heads, 16 QK heads, head_dim=128 (no KV cache).
#   - Vision encoder present (VL model); MTP head trained-in.
#   - Native ctx: 262,144 tokens (so 131,072 here needs no YaRN override).
#
# ------------------------------------------------------------
# PREREQUISITE -- register the B12X FP6 quantization in vLLM
# ------------------------------------------------------------
#   "b12x_fp6" is NOT a stock vLLM quant method. vLLM's native ModelOpt path
#   does not implement W6A6, so the B12X adapter must be registered first
#   (see b12x/examples/vllm_fp6_adapter.py -> build_b12x_fp6_vllm_config /
#   B12XFp6Config, name "b12x_fp6"). Until that QuantizationConfig is wired
#   into your vLLM (plugin entry point for multi-proc, or in-process for a
#   single worker), `--quantization b12x_fp6` will not resolve.
#
#   The two env gates below are what the B12X serving layer keys off of:
#     B12X_ENABLE_FP6=1        -> select the B12X FP6 path (else native fallback)
#     B12X_ENABLE_FP6_MICRO=0  -> use the fast STATIC kernel, not the BS1 micro
#                                 kernel (micro is slower for this workload).
#
# VRAM fit on 1x 96 GiB Blackwell:
#   - FP6 weights for the quantized Linears (~16B params @ ~6.25 bit) plus the
#     BF16 residue (visual/lm_head/embeddings/mtp) resident ~= 26 GiB.
#   - KV cache (Gated Attention layers only, BF16, TP=1):
#       per-token KV = 16 layers * 2 (K+V) * 4 KV heads * 256 = 64 KiB/token
#       128K ctx, 1 seq = 8 GiB ; 4 seqs = 32 GiB
#   - 26 GiB weights + 32 GiB KV + graphs/vision still fits 96 GiB with margin.
#
# This script does not store API keys. Pass the key as the first argument, or
# set VLLM_API_KEY in the environment.
# ============================================================

# --- Memory Management ---
export PYTORCH_ALLOC_CONF=expandable_segments:True

# --- GPU Selection (single Blackwell card) ---
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# --- B12X FP6 gates (see PREREQUISITE above) ---
export B12X_ENABLE_FP6="${B12X_ENABLE_FP6:-1}"
export B12X_ENABLE_FP6_MICRO="${B12X_ENABLE_FP6_MICRO:-0}"
# The vLLM plugin (b12x.integration.vllm_plugin) reads the FP6 checkpoint dir
# from here in every process, including spawned TP workers. Set below once
# MODEL_DIR is known.

# --- vLLM / CUDA behavior ---
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_SLEEP_WHEN_IDLE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

# --- Loading / CPU threading ---
export SAFETENSORS_FAST_GPU=1
export OMP_NUM_THREADS=8

# --- Torch profiler (registers POST /start_profile and /stop_profile) ---
# Recent vLLM nightlies gate these endpoints on the --profiler-config CLI arg
# (vllm/entrypoints/serve/profile/api_router.py); the old VLLM_TORCH_PROFILER_DIR
# env var no longer registers them. PROFILE=1 enables; traces (.json.gz) land in
# PROFILE_DIR; summarize with b12x scripts/summarize_vllm_trace.py.
PROFILE="${PROFILE:-0}"
PROFILE_DIR="${PROFILE_DIR:-/tmp/vllm_prof}"
if [[ "$PROFILE" == "1" ]]; then
  mkdir -p "$PROFILE_DIR"
  # Kept for older builds that still read the env var; harmless on new ones.
  export VLLM_TORCH_PROFILER_DIR="$PROFILE_DIR"
fi

# --- Long-context (YaRN) escape hatch ---
# 131072 is within the native 262144 window, so no override is needed. Only set
# this if you push MAX_MODEL_LEN past 262144 with an --hf-overrides rope config.
# export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# ============================================================
# Model / serving configuration
# ============================================================

API_KEY="${1:-${VLLM_API_KEY:-}}"

if [[ -z "$API_KEY" ]]; then
  echo "Usage: $0 API-KEY-HERE" >&2
  echo "       or: VLLM_API_KEY=API-KEY-HERE $0" >&2
  exit 2
fi

# Local B12X MX-FP6 (W6A6) checkpoint. Override at launch time if needed:
#   MODEL_DIR=/path/to/model ./qwen3.6-27b-fp6.sh
MODEL_DIR="${MODEL_DIR:-/media/fmodels/TheHouseOfTheDude/qwen3-6_27B_dense_fp6_la}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.6-27B-FP6-W6A6}"
# Tell the B12X vLLM plugin where the FP6 weights live (inherited by workers).
export B12X_FP6_MODEL_DIR="${B12X_FP6_MODEL_DIR:-$MODEL_DIR}"
# The FP6 export copies tokenizer.json/tokenizer_config.json and
# chat_template.jinja, so keep tokenizer resolution local by default.
TOKENIZER="${TOKENIZER:-$MODEL_DIR}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"

# Single GPU -> no tensor parallelism.
TP_SIZE="${TP_SIZE:-1}"

# ------------------------------------------------------------
# Capacity sizing for 1x 96 GiB Blackwell at full VLM (vision enabled).
# 128K context fits comfortably; bump MAX_NUM_SEQS for more concurrency
# (each 128K sequence costs ~8 GiB of KV cache).
# ------------------------------------------------------------
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

# Blackwell handles fp8 KV cache well; auto (BF16) already fits 128K here, so
# keep auto unless you want to trade a little accuracy for more concurrency.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

TOKENIZER_MODE="${TOKENIZER_MODE:-hf}"
CONFIG_FORMAT="${CONFIG_FORMAT:-auto}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"

# B12X MX-FP6 (W6A6). Requires the B12XFp6Config adapter registered in vLLM
# (see PREREQUISITE above). Set QUANTIZATION="" to let vLLM auto-detect from
# config.json (only works if the adapter overrides the modelopt method).
QUANTIZATION="${QUANTIZATION:-b12x_fp6}"

# Qwen3.6 ships a custom modeling file (qwen3_5 / qwen3_next family).
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

# ============================================================
# Qwen3.6 reasoning / tool-call / MTP / VLM toggles
# ============================================================

# Qwen3.6 thinks by default; qwen3 reasoning parser covers the whole family.
REASONING_PARSER="${REASONING_PARSER:-qwen3}"

# Tool calling: qwen3_coder is the model-card parser for OpenAI-style tool use.
ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"

# Multi-Token Prediction (trained-in MTP head, kept BF16 by the FP6 export).
# Disable (ENABLE_MTP=0) if you hit a Mamba/DeltaNet cudagraph assert during
# MTP draft capture, or when benchmarking non-spec throughput.
ENABLE_MTP="${ENABLE_MTP:-1}"
MTP_SPEC="${MTP_SPEC:-{\"method\":\"qwen3_next_mtp\",\"num_speculative_tokens\":2}}"

# Vision. VL model; the FP6 export left the visual tower in BF16 so it works
# out of the box. Set TEXT_ONLY=1 to skip vision profiling and free its
# workspace for additional KV cache (text-only traffic).
TEXT_ONLY="${TEXT_ONLY:-0}"

# Optional: lets clients drive video frame sampling via
# extra_body={"mm_processor_kwargs": {"fps": ...}}. Harmless when no video.
MEDIA_IO_KWARGS="${MEDIA_IO_KWARGS:-{\"video\":{\"num_frames\":-1}}}"

# Prefix caching is generally a win for chat/agent workloads.
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"

# CUDA graph setup:
# - full_decode_only keeps CUDA graphs on the decode path and avoids the
#   highest-memory full-prefill captures.
# - Qwen3.6's Gated DeltaNet layers occasionally trip a cudagraph cache assert
#   ("Mamba/DeltaNet cuda-graph cache" path). If you see it, set
#   MAX_CUDAGRAPH_CAPTURE_SIZE=4 (or 1) below as a fallback.
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-full_decode_only}"
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-[1,2,4]}"
COMPILATION_CONFIG="${COMPILATION_CONFIG:-{\"mode\":3,\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\",\"cudagraph_capture_sizes\":${CUDAGRAPH_CAPTURE_SIZES}}}"

# Fallback for older vLLM versions that do not understand cudagraph_capture_sizes
# inside --compilation-config, OR the DeltaNet cudagraph cache assert above.
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"

# Eager mode (no torch.compile / no CUDA graphs). REQUIRED workaround while the
# B12X FP6 linear is not yet wrapped as an opaque torch custom op: vLLM's
# VLLM_COMPILE (mode 3) traces the language-model forward with Dynamo in
# fullgraph mode, and the B12X path JIT-compiles a CUTE kernel inside the
# forward (tempfile / os.getpid), which Dynamo cannot trace -> hard graph break
# ("Attempted to call function marked as skipped: posix.getpid"). ENFORCE_EAGER=1
# runs the model eagerly so the FP6 kernels JIT normally. Trades compile/cudagraph
# perf for a working server; flip back to 0 once the custom-op wrapper lands.
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

# Eager mode disables torch.compile + CUDA graphs, so the compilation-config is
# both unnecessary and conflicting there; only pass it when compiling.
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

if [[ -n "$REASONING_PARSER" ]]; then
  VLLM_ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi

if [[ "$ENABLE_TOOL_CALLING" == "1" ]]; then
  VLLM_ARGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi

if [[ "$ENABLE_MTP" == "1" ]]; then
  VLLM_ARGS+=(--speculative-config "$MTP_SPEC")
fi

if [[ "$TEXT_ONLY" == "1" ]]; then
  VLLM_ARGS+=(--language-model-only)
elif [[ -n "$MEDIA_IO_KWARGS" ]]; then
  VLLM_ARGS+=(--media-io-kwargs "$MEDIA_IO_KWARGS")
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
echo "  REASONING_PARSER=${REASONING_PARSER}"
echo "  TOOL_CALLING=${ENABLE_TOOL_CALLING} (parser=${TOOL_CALL_PARSER})"
echo "  MTP=${ENABLE_MTP} (${MTP_SPEC})"
echo "  TEXT_ONLY=${TEXT_ONLY}"
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
