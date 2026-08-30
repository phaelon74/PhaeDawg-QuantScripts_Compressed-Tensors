#!/usr/bin/env bash
# Phase 1 host capture for EXL3 SM86 decode optimization.
# Run on d011sd02 after stopping any vLLM using GPUs 1-4.
# Does not start the server. Paste output back into chat.
set -euo pipefail

# Locate this tree from the script path. EXL3 is a directory prefix, not PATH.
EXL3="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_4P25="${EXL3_MODEL_DIR:-/media/fmodels/ArtusDev/TheDrummer_Behemoth-R1-123B-v2-EXL3/4.25bpw_H6}"
# Logical device 0 after this mask is physical GPU 1.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export VLLM_EXL3_EXT_PATH="${VLLM_EXL3_EXT_PATH:-$EXL3/build/exllamav3_ext}"
export VLLM_EXL3_SKIP_VERSION_GUARD=1
export PYTHONPATH="$VLLM_EXL3_EXT_PATH:$EXL3/plugin/src${PYTHONPATH:+:$PYTHONPATH}"
TORCH_LIB="$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")')"
export LD_LIBRARY_PATH="$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$EXL3/results" "$EXL3/manifests"

echo "=== 0. paths and python ==="
echo "EXL3=$EXL3"
which python
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
python -c "import vllm; print('vllm', vllm.__version__)"
python -c "import torch, exllamav3_ext, pathlib; print('ext', pathlib.Path(exllamav3_ext.__file__).resolve())"

echo "=== 1. GPU clocks / topo (physical 1-4) ==="
nvidia-smi --query-gpu=index,name,compute_cap,clocks.gr,clocks.mem,clocks.max.gr,clocks.max.mem,power.limit,memory.total,memory.used --format=csv
nvidia-smi topo -m
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE | sed -n '1,220p'

echo "=== 2. stack capture ==="
bash "$EXL3/scripts/capture_manifest.sh" "$EXL3/manifests/stack.captured.json"

echo "=== 3. ArtusDev 4.25 inventory ==="
python "$EXL3/scripts/validate_exl3_checkpoint.py" "$MODEL_4P25" \
  --profile artusdev-4p25 \
  --sha-manifest "$EXL3/results/artusdev_4p25_inventory.json" \
  || echo "validator exited $?; inventory JSON is still written if the file exists"

echo "=== 4. CPU tests ==="
python -m pytest "$EXL3/tests/test_nvtx.py" "$EXL3/tests/test_graph_policy.py" "$EXL3/tests/test_graph_prewarm.py" "$EXL3/tests/test_ops_fake.py" -q

echo "=== 5. 3inst decode microbench (production codebook) ==="
python "$EXL3/scripts/kernel_microbench.py" \
  --device 0 \
  --bitrates 3,4,5,6 \
  --m 1,2,4 \
  --output "$EXL3/results/kernel_microbench_3inst_decode.json"

echo "=== 6. mixed-K token budget ==="
python "$EXL3/scripts/decode_latency_budget.py" \
  --microbench "$EXL3/results/kernel_microbench_3inst_decode.json" \
  --inventory "$EXL3/results/artusdev_4p25_inventory.json" \
  --output "$EXL3/results/decode_latency_budget.json" \
  || echo "budget exited $?; continuing"

echo "=== 7. prewarm 3inst + K3/K4/K5 ==="
python "$EXL3/scripts/prewarm_kernels.py" --device 0 --codebook 3inst --output "$EXL3/results/prewarm_3inst.json"

echo "=== 8. graph capture smoke (10k, all shapes, 3inst K4) ==="
python "$EXL3/scripts/graph_replay_stress.py" --device 0 --steps 10000 --bitrate 4 --all-shapes

echo "=== 9. kernel-only token timer (uniform K4 3inst) ==="
VLLM_EXL3_NVTX=1 python "$EXL3/scripts/profile_decode_nsys.py" --device 0 --codebook 3inst --bitrate 4 --iters 10

echo "Phase 1 kernel block finished. Paste this log plus the JSON files under $EXL3/results/."
