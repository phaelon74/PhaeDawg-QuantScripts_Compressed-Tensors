#!/usr/bin/env bash
# ncu one M=1 projection kernel. Stop serve first. Physical GPU 1 only.
#   export EXL3=...
#   sudo -E bash "$EXL3/scripts/profile_ncu_gate.sh" OUT LEAF BITRATE [regular|staged|register]
set -euo pipefail
EXL3="${EXL3:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT="${1:-$EXL3/results/phase0/ncu_gate_m1}"
LEAF="${2:-gate_proj}"
BITRATE="${3:-4}"
KERNEL="${4:-regular}"
PYTHON_BIN="${EXL3_PYTHON:-$(command -v python)}"
NCU_BIN="${NCU_BIN:-$(command -v ncu || true)}"
if [[ -z "$NCU_BIN" || ! -x "$NCU_BIN" ]]; then
  echo "ncu not found; export NCU_BIN=\"\$(command -v ncu)\" before sudo." >&2
  exit 2
fi
mkdir -p "$(dirname "$OUT")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export VLLM_EXL3_EXT_PATH="${VLLM_EXL3_EXT_PATH:-$EXL3/build/exllamav3_ext}"
export VLLM_EXL3_SKIP_VERSION_GUARD=1
export PYTHONPATH="$VLLM_EXL3_EXT_PATH:$EXL3/plugin/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$("$PYTHON_BIN" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")'):${LD_LIBRARY_PATH:-}"
export VLLM_EXL3_FORCE_COMPRESSED=1
export VLLM_EXL3_NVTX=1
case "$KERNEL" in
  regular) unset EXL3_GEMV_K56 ;;
  staged|k56) export EXL3_GEMV_K56=1 ;;
  register) export EXL3_GEMV_K56=2 ;;
  *)
    echo "kernel must be regular, staged, or register; got: $KERNEL" >&2
    exit 2
    ;;
esac

echo "Profiling leaf=$LEAF bitrate=$BITRATE kernel=$KERNEL with $PYTHON_BIN"
"$NCU_BIN" --set full --target-processes all \
  --metrics sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct \
  --kernel-name regex:exl3 \
  --launch-skip 8 --launch-count 1 \
  -o "$OUT" --force-overwrite \
  "$PYTHON_BIN" "$EXL3/scripts/profile_decode_nsys.py" \
    --device 0 --m 1 --bitrate "$BITRATE" --codebook 3inst \
    --leaf "$LEAF" --warmup 3 --iters 6

echo "Wrote $OUT.ncu-rep"
echo "Open with: ncu-ui $OUT.ncu-rep"
echo "ALU-bound if pipe_alu pct is high and dram pct is well below 70%."
