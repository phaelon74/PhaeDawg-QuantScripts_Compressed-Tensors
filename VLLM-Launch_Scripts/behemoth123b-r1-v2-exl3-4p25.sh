#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Behemoth-R1-123B-v2 ArtusDev EXL3 4.25-bpw, H6 lm_head
# Native EXL3 SM86 plugin on physical GPUs 1,2,3,4 (TP4)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXL3_ROOT="${EXL3_ROOT:-$REPO_ROOT/Behemoth-R1-123B-v2_Compressed-Tensors/EXL3-vLLM}"

export PYTORCH_ALLOC_CONF=expandable_segments:True

# Physical GPUs 0 and 5 are reserved by other services.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2,3,4

export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_SLEEP_WHEN_IDLE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export SAFETENSORS_FAST_GPU=1
export OMP_NUM_THREADS=8

# EXL3 plugin and native SM86 extension. The guard override is required for
# the tested Torch 2.13.0+cu132 / vLLM b99dae944 stack until the historical
# constants are replaced with a multi-stack manifest.
export VLLM_PLUGINS="vllm_exl3_sm86"
export VLLM_EXL3_EXT_PATH="${VLLM_EXL3_EXT_PATH:-$EXL3_ROOT/build/exllamav3_ext}"
export VLLM_EXL3_SKIP_VERSION_GUARD="${VLLM_EXL3_SKIP_VERSION_GUARD:-1}"
export VLLM_EXL3_CROSSOVER_JSON="${VLLM_EXL3_CROSSOVER_JSON:-$EXL3_ROOT/manifests/sm86_crossover.json}"
unset VLLM_MARLIN_USE_ATOMIC_ADD
unset VLLM_EXL3_FORCE_COMPRESSED
unset VLLM_EXL3_FORCE_RECONSTRUCT

API_KEY="${1:-${VLLM_API_KEY:-}}"
if [[ -z "$API_KEY" ]]; then
  echo "Usage: $0 API-KEY-HERE" >&2
  echo "       or: VLLM_API_KEY=API-KEY-HERE $0" >&2
  exit 2
fi

# Use EXL3-specific override names so stale MODEL_DIR/TOKENIZER exports from
# conversion or KLD sessions cannot silently select another checkpoint.
MODEL_DIR="${EXL3_MODEL_DIR:-/media/fmodels/ArtusDev/TheDrummer_Behemoth-R1-123B-v2-EXL3/4.25bpw_H6}"
SERVED_MODEL_NAME="${EXL3_SERVED_MODEL_NAME:-Behemoth-R1-123B-v2-EXL3-4.25-H6}"
TOKENIZER="${EXL3_TOKENIZER:-$MODEL_DIR}"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Missing model config: $MODEL_DIR/config.json" >&2
  exit 2
fi
if [[ ! -f "$MODEL_DIR/quantization_config.json" ]]; then
  echo "Missing EXL3 metadata: $MODEL_DIR/quantization_config.json" >&2
  exit 2
fi
if ! compgen -G "$VLLM_EXL3_EXT_PATH/exllamav3_ext*.so" >/dev/null; then
  echo "Missing native extension under $VLLM_EXL3_EXT_PATH" >&2
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
QUANTIZATION="${QUANTIZATION:-exl3}"

REASONING_PARSER="${REASONING_PARSER:-}"
REASONING_CONFIG="${REASONING_CONFIG:-}"
ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-0}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-mistral}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"

# Production performance mode: CUDA graphs enabled, eager disabled.
# Set EXL3_ENFORCE_EAGER=1 only for correctness/KLD diagnostics.
ENFORCE_EAGER="${EXL3_ENFORCE_EAGER:-0}"

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
  --disable-custom-all-reduce
)

if [[ "$ENFORCE_EAGER" == "1" ]]; then
  unset VLLM_EXL3_ALLOW_GRAPHS
  VLLM_ARGS+=(--enforce-eager)
else
  CUDAGRAPH_MODE="${EXL3_CUDAGRAPH_MODE:-full_decode_only}"
  CUDAGRAPH_CAPTURE_SIZES="${EXL3_CUDAGRAPH_CAPTURE_SIZES:-[1,2,4]}"
  COMPILATION_CONFIG="${EXL3_COMPILATION_CONFIG:-{\"mode\":3,\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\",\"cudagraph_capture_sizes\":${CUDAGRAPH_CAPTURE_SIZES}}}"
  export VLLM_EXL3_ALLOW_GRAPHS=1
  VLLM_ARGS+=(--compilation-config "$COMPILATION_CONFIG")
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

echo "Launching native EXL3 SM86:"
echo "  MODEL_DIR=$MODEL_DIR"
echo "  SERVED_MODEL_NAME=$SERVED_MODEL_NAME"
echo "  QUANTIZATION=$QUANTIZATION"
echo "  VLLM_PLUGINS=$VLLM_PLUGINS"
echo "  VLLM_EXL3_EXT_PATH=$VLLM_EXL3_EXT_PATH"
echo "  VLLM_EXL3_CROSSOVER_JSON=$VLLM_EXL3_CROSSOVER_JSON"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  TP_SIZE=$TP_SIZE"
echo "  MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "  GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
echo "  ENFORCE_EAGER=$ENFORCE_EAGER"
if [[ "$ENFORCE_EAGER" != "1" ]]; then
  echo "  VLLM_EXL3_ALLOW_GRAPHS=$VLLM_EXL3_ALLOW_GRAPHS"
  echo "  COMPILATION_CONFIG=$COMPILATION_CONFIG"
fi
echo

vllm serve "${VLLM_ARGS[@]}"
