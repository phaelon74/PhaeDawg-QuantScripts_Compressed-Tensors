#!/usr/bin/env bash
# Convert a small public dense Mistral checkpoint to EXL3 for loader/TP tests.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IN_DIR="${IN_DIR:-${MISTRAL_IN_DIR:-}}"
OUT_DIR="${OUT_DIR:-$ROOT/build/mistral-exl3-small}"
WORK_DIR="${WORK_DIR:-$OUT_DIR.work}"
CONVERT_PY="${CONVERT_PY:-$HOME/src/exllamav3/convert.py}"

if [[ -z "$IN_DIR" ]]; then
  echo "Set IN_DIR or MISTRAL_IN_DIR to a dense Mistral (or Mistral-like) HF dir." >&2
  echo "Example: mistralai/Mistral-7B-Instruct-v0.3 already downloaded locally." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$WORK_DIR" "$OUT_DIR"

python "$CONVERT_PY" \
  -i "$IN_DIR" \
  -w "$WORK_DIR" \
  -o "$OUT_DIR" \
  -b 3.5 \
  -hb 6 \
  -cb mul1 \
  -cr 128 \
  -cc 2048 \
  -d 0 \
  ${RESUME:+-r}

python "$ROOT/scripts/validate_exl3_checkpoint.py" "$OUT_DIR" --allow-non-behemoth
echo "Small EXL3 checkpoint: $OUT_DIR"
