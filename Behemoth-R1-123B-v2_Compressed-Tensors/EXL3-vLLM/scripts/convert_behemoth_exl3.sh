#!/usr/bin/env bash
# Resumable Behemoth-R1-123B-v2 EXL3 conversion: 3.5 bpw, H6 lm_head, mul1.
# Run in ~/exllamav3-convert with CUDA_VISIBLE_DEVICES=0.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IN_DIR="${IN_DIR:-/media/fmodels/TheDrummer/Behemoth-R1-123B-v2/main}"
WORK_DIR="${WORK_DIR:-/media/fmodels/TheHouseOfTheDude/Behemoth-R1-123B-v2/NewQuants/EXL3_3p5_H6_mul1.work}"
OUT_DIR="${OUT_DIR:-/media/fmodels/TheHouseOfTheDude/Behemoth-R1-123B-v2/NewQuants/EXL3_3p5_H6_mul1}"
CONVERT_PY="${CONVERT_PY:-$HOME/src/exllamav3/convert.py}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$WORK_DIR" "$OUT_DIR"

# Work dir must be able to hold a complete output copy plus in-flight tensors.
avail_kb=$(df -Pk "$WORK_DIR" | awk 'NR==2 {print $4}')
need_kb=$((80 * 1024 * 1024))
if [[ "$avail_kb" -lt "$need_kb" ]]; then
  echo "warning: $WORK_DIR has ${avail_kb} KiB free; want >= 80 GiB for a full copy" >&2
fi

ARGS=(
  -i "$IN_DIR"
  -w "$WORK_DIR"
  -o "$OUT_DIR"
  -b 3.5
  -hb 6
  -cb mul1
  -cr 250
  -cc 2048
  -d 0
  -ss 8192
)

if [[ -f "$WORK_DIR/job.json" || -f "$WORK_DIR/job_state.json" || -d "$WORK_DIR/ckpt" ]]; then
  ARGS+=(-r)
  echo "Resuming from $WORK_DIR"
fi

echo "Converting Behemoth EXL3 with ${ARGS[*]}"
python "$CONVERT_PY" "${ARGS[@]}"

python "$ROOT/scripts/validate_exl3_checkpoint.py" "$OUT_DIR"
