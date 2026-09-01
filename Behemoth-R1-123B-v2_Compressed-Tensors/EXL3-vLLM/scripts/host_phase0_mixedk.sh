#!/usr/bin/env bash
# Remaining Phase 0 of exl3-int4-speed-decode:
#   1. ArtusDev 4.25 mixed-K inventory
#   2. M=1 native times for all TP4 shapes including lm_head
#   3. Inventory-weighted token budget (K5/K6 fall off GEMV)
#   4. True-shape mgemm: gate/up K4, k/v K4+K6, padded q/k/v K4
#
# Stop vLLM on GPUs 1-4 first. Physical 0 and 5 stay reserved.
# Usage:
#   source /home/phaedawg/exl3vllm/venv/bin/activate
#   bash "$EXL3/scripts/host_phase0_mixedk.sh"
set -euo pipefail

EXL3="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_4P25="${EXL3_MODEL_DIR:-/media/fmodels/ArtusDev/TheDrummer_Behemoth-R1-123B-v2-EXL3/4.25bpw_H6}"
OUT="${EXL3_PHASE0_OUT:-$EXL3/results/phase0_mixedk}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export VLLM_EXL3_EXT_PATH="${VLLM_EXL3_EXT_PATH:-$EXL3/build/exllamav3_ext}"
export VLLM_EXL3_SKIP_VERSION_GUARD=1
export PYTHONPATH="$VLLM_EXL3_EXT_PATH:$EXL3/plugin/src${PYTHONPATH:+:$PYTHONPATH}"
TORCH_LIB="$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")')"
export LD_LIBRARY_PATH="$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$OUT"

echo "EXL3=$EXL3"
echo "OUT=$OUT"
echo "MODEL_4P25=$MODEL_4P25"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
which python
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "=== 1. ArtusDev 4.25 inventory ==="
python "$EXL3/scripts/validate_exl3_checkpoint.py" "$MODEL_4P25" \
  --profile artusdev-4p25 \
  --sha-manifest "$OUT/artusdev_4p25_inventory.json"

echo "=== 2. M=1 kernel microbench (all shapes, K4/5/6, lm_head H6) ==="
python "$EXL3/scripts/kernel_microbench.py" \
  --device 0 \
  --bitrates 4,5,6 \
  --m 1 \
  --output "$OUT/roofline_baseline_m1.json"

echo "=== 3. mixed-K inventory budget ==="
python "$EXL3/scripts/decode_latency_budget.py" \
  --microbench "$OUT/roofline_baseline_m1.json" \
  --inventory "$OUT/artusdev_4p25_inventory.json" \
  --target-tok-s 30 \
  --output "$OUT/decode_latency_budget.json"

echo "=== 4. mgemm true-shape (M=1,2,4) ==="
python "$EXL3/scripts/mgemm_microbench.py" \
  --device 0 --shapes gate_up --bitrate 4 --m 1,2,4 \
  --output "$OUT/mgemm_gate_up_k4.json"
python "$EXL3/scripts/mgemm_microbench.py" \
  --device 0 --shapes kv --bitrate 4 --m 1,2,4 \
  --output "$OUT/mgemm_kv_k4.json"
python "$EXL3/scripts/mgemm_microbench.py" \
  --device 0 --shapes kv --bitrate 6 --m 1,2,4 \
  --output "$OUT/mgemm_kv_k6.json"
python "$EXL3/scripts/mgemm_microbench.py" \
  --device 0 --shapes qkv --bitrate 4 --m 1,2,4 \
  --output "$OUT/mgemm_qkv_padded_k4.json"

echo "Phase 0 mixed-K receipts under $OUT"
echo "Paste decode_latency_budget.json and the four mgemm JSON files back into chat."
