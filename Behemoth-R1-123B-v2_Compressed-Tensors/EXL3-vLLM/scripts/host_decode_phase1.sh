#!/usr/bin/env bash
# Phase 1 host capture for EXL3 SM86 decode optimization.
# Run on d011sd02 after stopping any vLLM using GPUs 1-4.
# Does not start the server. Paste output back into chat.
set -euo pipefail

EXL3="${EXL3:-$HOME/kld-exl3-vllm/PhaeDawg-QuantScripts_Compressed-Tensors/Behemoth-R1-123B-v2_Compressed-Tensors/EXL3-vLLM}"
MODEL_4P25="${EXL3_MODEL_DIR:-/media/fmodels/ArtusDev/TheDrummer_Behemoth-R1-123B-v2-EXL3/4.25bpw_H6}"
# Logical device 0 after this mask is physical GPU 1.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export VLLM_EXL3_EXT_PATH="${VLLM_EXL3_EXT_PATH:-$EXL3/build/exllamav3_ext}"
export VLLM_EXL3_SKIP_VERSION_GUARD=1
export PYTHONPATH="$VLLM_EXL3_EXT_PATH:$EXL3/plugin/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$EXL3"
mkdir -p results manifests

echo "=== 0. paths and python ==="
pwd
which python
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
python -c "import vllm; print('vllm', vllm.__version__)"
python -c "import exllamav3_ext, pathlib; print('ext', pathlib.Path(exllamav3_ext.__file__).resolve())"

echo "=== 1. GPU clocks / topo (physical 1-4) ==="
nvidia-smi --query-gpu=index,name,compute_cap,clocks.gr,clocks.mem,clocks.max.gr,clocks.max.mem,power.limit,memory.total,memory.used --format=csv
nvidia-smi topo -m
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE | sed -n '1,220p'

echo "=== 2. stack capture ==="
bash scripts/capture_manifest.sh manifests/stack.captured.json

echo "=== 3. ArtusDev 4.25 inventory ==="
python scripts/validate_exl3_checkpoint.py "$MODEL_4P25" \
  --profile artusdev-4p25 \
  --sha-manifest results/artusdev_4p25_inventory.json \
  || echo "validator exited $?; inventory JSON is still written if the file exists"

echo "=== 4. CPU tests ==="
python -m pytest tests/test_nvtx.py tests/test_graph_policy.py tests/test_graph_prewarm.py tests/test_ops_fake.py -q

echo "=== 5. 3inst decode microbench (production codebook) ==="
python scripts/kernel_microbench.py \
  --device 0 \
  --bitrates 3,4,5,6 \
  --m 1,2,4 \
  --output results/kernel_microbench_3inst_decode.json

echo "=== 6. mixed-K token budget ==="
python scripts/decode_latency_budget.py \
  --microbench results/kernel_microbench_3inst_decode.json \
  --inventory results/artusdev_4p25_inventory.json \
  --output results/decode_latency_budget.json

echo "=== 7. prewarm 3inst + K3/K4/K5 ==="
python scripts/prewarm_kernels.py --device 0 --codebook 3inst --output results/prewarm_3inst.json

echo "=== 8. graph capture smoke (10k, all shapes, 3inst K4) ==="
python scripts/graph_replay_stress.py --device 0 --steps 10000 --bitrate 4 --all-shapes

echo "=== 9. kernel-only token timer (uniform K4 3inst) ==="
VLLM_EXL3_NVTX=1 python scripts/profile_decode_nsys.py --device 0 --codebook 3inst --bitrate 4 --iters 10

echo "Phase 1 kernel block finished. Paste this log plus the JSON files under results/."
