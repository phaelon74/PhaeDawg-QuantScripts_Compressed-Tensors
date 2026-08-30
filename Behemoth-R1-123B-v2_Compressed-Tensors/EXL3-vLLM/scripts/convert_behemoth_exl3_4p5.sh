#!/usr/bin/env bash
# Resumable Behemoth-R1-123B-v2 EXL3 conversion: 4.5 bpw, H6, mul1.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN="0c49587a7c235e6303a6bbedc8b665272ad3a2ea"
IN_DIR="${IN_DIR:-/media/fmodels/TheDrummer/Behemoth-R1-123B-v2}"
WORK_DIR="${WORK_DIR:-/media/fmodels/TheHouseOfTheDude/Behemoth-R1-123B-v2/NewQuants/EXL3_4p5_H6_mul1.work}"
OUT_DIR="${OUT_DIR:-/media/fmodels/TheHouseOfTheDude/Behemoth-R1-123B-v2/NewQuants/EXL3_4p5_H6_mul1}"
EXLLAMAV3_SRC="${EXLLAMAV3_SRC:-$HOME/src/exllamav3}"
CONVERT_PY="${CONVERT_PY:-$EXLLAMAV3_SRC/convert.py}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Physical GPU 1 becomes logical device 0. Never use busy physical GPUs 0 or 5.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

if [[ ! -f "$IN_DIR/config.json" && -f "$IN_DIR/main/config.json" ]]; then
  IN_DIR="$IN_DIR/main"
fi
if [[ ! -f "$IN_DIR/config.json" ]]; then
  echo "Missing BF16 source config: $IN_DIR/config.json" >&2
  exit 2
fi
if [[ ! -f "$CONVERT_PY" ]]; then
  echo "Missing ExLlamaV3 converter: $CONVERT_PY" >&2
  exit 2
fi
if ! python -c "import marisa_trie" >/dev/null 2>&1; then
  echo "Missing compile-time dependency: marisa-trie" >&2
  echo "Install it in the active conversion venv: python -m pip install marisa-trie" >&2
  exit 2
fi

actual_pin="$(git -C "$EXLLAMAV3_SRC" rev-parse HEAD)"
if [[ "$actual_pin" != "$PIN" ]]; then
  echo "ExLlamaV3 is $actual_pin; expected $PIN" >&2
  exit 2
fi

mkdir -p "$WORK_DIR" "$OUT_DIR"
echo "Disk space for work and output (plan for about 150 GiB combined):"
df -h "$WORK_DIR" "$OUT_DIR"

if [[ -f "$WORK_DIR/args.json" || -f "$WORK_DIR/ckpt/job.json" ]]; then
  if [[ -f "$WORK_DIR/args.json" ]]; then
    saved_cal_rows="$(
      python - "$WORK_DIR/args.json" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8")).get("cal_rows", "unknown"))
PY
    )"
    if [[ "$saved_cal_rows" != "250" ]]; then
      echo "Saved job uses cal_rows=$saved_cal_rows, expected 250." >&2
      echo "Archive this work/output pair and start a new job; do not resume it." >&2
      exit 2
    fi
  fi
  echo "Resuming the saved 4.5-bpw job from $WORK_DIR"
  python "$CONVERT_PY" -w "$WORK_DIR" -r
else
  shopt -s nullglob dotglob
  output_entries=("$OUT_DIR"/*)
  shopt -u nullglob dotglob
  if (( ${#output_entries[@]} )); then
    echo "Output directory is not empty and no resumable work state exists: $OUT_DIR" >&2
    exit 2
  fi
  python "$CONVERT_PY" \
    -i "$IN_DIR" \
    -w "$WORK_DIR" \
    -o "$OUT_DIR" \
    -b 4.5 \
    -hb 6 \
    -cb mul1 \
    -cr 250 \
    -cc 2048 \
    -d 0 \
    -ss 8192
fi

python "$ROOT/scripts/validate_exl3_checkpoint.py" \
  "$OUT_DIR" \
  --allow-non-behemoth \
  --sha-manifest "$ROOT/results/behemoth_exl3_4p5_inventory.json"
du -sh "$OUT_DIR"
du -sb "$OUT_DIR"
