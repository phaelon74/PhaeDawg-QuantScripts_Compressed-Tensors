#!/usr/bin/env bash
# Capture Marlin TP4 serving baselines for the frozen 72G mixed checkpoint.
# Uses the existing launch-script defaults (physical GPUs 1-4, TP=4).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"
OUT="${1:-$ROOT/manifests/marlin_tp4_baseline.captured.json}"
TEMPLATE="$ROOT/manifests/marlin_tp4_baseline.template.json"
BENCH="$ROOT/scripts/bench_exl3_vs_marlin.py"
MODEL_DIR="${MODEL_DIR:-/media/fmodels/TheHouseOfTheDude/Behemoth-R1-123B-v2/NewQuants/AutoRound_GS32_Mixed_72G}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"
export QUANTIZATION="${QUANTIZATION:-compressed-tensors}"
export MODEL_DIR
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Behemoth-R1-123B-v2-Marlin-72G}"

python3 "$BENCH" \
  --mode marlin \
  --model-dir "$MODEL_DIR" \
  --template "$TEMPLATE" \
  --output "$OUT" \
  --tensor-parallel-size 4 \
  --prompt-lengths 128,1024,4096,8192,16384,32768 \
  --decode-tokens 256 \
  --concurrency 1,2,4

echo "Marlin baseline written to $OUT"
echo "Launch script reference: $REPO/VLLM-Launch_Scripts/behemoth123b-r1-v2.sh"
