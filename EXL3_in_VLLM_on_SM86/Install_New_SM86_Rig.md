# Install EXL3 in vLLM on a new SM86 rig

This guide installs the dense EXL3 vLLM plugin on a clean Linux machine with
one or more NVIDIA **compute capability 8.6 (SM86)** GPUs. It does not assume
RTX 3090s, a particular GPU count, fixed GPU indices, or any existing Python
environment.

The GPU architecture makes the CUDA extension compatible. It does **not**
guarantee that a model fits, that a tensor-parallel size is valid for the
model, or that performance tuning from another SM86 GPU transfers.

## 1. Supported scope

### Hardware

Examples of SM86 GPUs include:

- GeForce RTX 3050, 3060, 3060 Ti, 3070, 3070 Ti, 3080, 3080 Ti, and 3090;
- RTX A4000, A4500, A5000, and A6000;
- NVIDIA A10 and A40.

A100 and A30 are Ampere but **SM80**, not SM86. Do not use this SM86-only
extension build for them.

Mixed SM86 cards can participate in one tensor-parallel group, but:

- every rank receives an equal logical shard;
- the smallest-VRAM GPU determines whether the model fits;
- the slowest GPU and PCIe path can determine throughput;
- vLLM tensor-parallel divisibility rules must still hold.

### Models

The current plugin is a dense-only implementation built around vLLM's
standard row-parallel, column-parallel, packed QKV, and packed gate/up linear
layers. It has been validated end-to-end with dense Mistral architecture.

Before treating another architecture as supported, require:

- `quantization_config.json` with a non-empty EXL3 `tensor_storage` map;
- EXL3 `trellis`, `suh`, and `svh` tensors for quantized linear layers;
- either explicit `mul1`/`mcg` markers or supported implicit codebook metadata;
- TP-local input/output dimensions that remain 128-aligned;
- a TP size compatible with attention heads, KV heads, hidden size, and
  intermediate size;
- a successful eager load, deterministic generation, and CUDA parity tests.

Not currently supported as a generic promise:

- MoE/expert-parallel EXL3;
- SparkInfer;
- multimodal vision/audio towers;
- MTP-specific paths;
- sleep-mode weight unload/reload;
- `state_dict()` resave of EXL3 side tensors;
- arbitrary CUDA-graph capture without model-specific prewarm/replay testing.

## 2. Tested software stack

Use the tested stack first. Upgrade one component at a time only after saving
a working environment.

| Component | Tested value |
| --- | --- |
| OS | Linux; Ubuntu 24.04-class environment |
| Python | 3.12 |
| GPU ISA | SM86 |
| Torch | `2.13.0+cu132` |
| Torch CUDA runtime | 13.2 |
| vLLM | `0.1.dev20517+gb99dae944` |
| vLLM source | `phaelon74/vllm` |
| vLLM commit | `b99dae944558c6f4d9978eb10bae25c854b81340` |
| ExLlamaV3 source | `turboderp-org/exllamav3` |
| ExLlamaV3 commit | `0c49587a7c235e6303a6bbedc8b665272ad3a2ea` |
| Plugin | `vllm-exl3-sm86` from this repository |

The vLLM fork contains KLD support but can also be used for ordinary serving.
The KLD-specific prompt-renderer patch is not required for hosting.

## 3. Verify the new rig

Install a sufficiently recent NVIDIA driver and a CUDA 13.x development
toolkit containing `nvcc`. The Torch wheel supplies CUDA runtime libraries,
but compiling `exllamav3_ext` requires the development toolkit.

```bash
nvidia-smi
nvidia-smi \
  --query-gpu=index,name,compute_cap,memory.total \
  --format=csv
nvcc --version
```

Every GPU selected for this plugin must report compute capability `8.6`.

Inspect topology:

```bash
nvidia-smi topo -m
```

PCIe-only TP works, but increasing TP can lose performance to NCCL all-reduce.

## 4. Install system prerequisites

Ubuntu 24.04 example:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  git \
  ninja-build \
  pkg-config \
  python3.12 \
  python3.12-dev \
  python3.12-venv \
  curl
```

Modern vLLM also builds Rust components. Install Rust with the official
`rustup` installer if `rustc` and `cargo` are not already available:

```bash
command -v rustc
command -v cargo
```

If either command is missing:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
  | sh -s -- -y
source "$HOME/.cargo/env"
```

Verify:

```bash
rustc --version
cargo --version
```

## 5. Create an isolated workspace and venv

Do not install into a system Python, an existing production vLLM environment,
or an ExLlama conversion environment.

```bash
export SM86_ROOT="$HOME/exl3-vllm-sm86"
export VENV_SM86="$SM86_ROOT/venv"
export VLLM_SRC="$SM86_ROOT/vllm"
export REPO="$SM86_ROOT/PhaeDawg-QuantScripts_Compressed-Tensors"
export EXL3_WORKSTREAM="$REPO/Behemoth-R1-123B-v2_Compressed-Tensors/EXL3-vLLM"
export EXLLAMAV3_SRC="$SM86_ROOT/exllamav3"

mkdir -p "$SM86_ROOT"
python3.12 -m venv "$VENV_SM86"
source "$VENV_SM86/bin/activate"

python -m pip install --upgrade pip wheel
which python
python --version
```

The expected Python executable is `$VENV_SM86/bin/python`.

## 6. Clone exact source revisions

```bash
git clone \
  https://github.com/phaelon74/PhaeDawg-QuantScripts_Compressed-Tensors.git \
  "$REPO"

git clone https://github.com/phaelon74/vllm.git "$VLLM_SRC"
git -C "$VLLM_SRC" checkout \
  b99dae944558c6f4d9978eb10bae25c854b81340

git clone https://github.com/turboderp-org/exllamav3.git "$EXLLAMAV3_SRC"
git -C "$EXLLAMAV3_SRC" checkout \
  0c49587a7c235e6303a6bbedc8b665272ad3a2ea
```

Confirm:

```bash
git -C "$VLLM_SRC" rev-parse HEAD
git -C "$EXLLAMAV3_SRC" rev-parse HEAD
test -f "$EXL3_WORKSTREAM/plugin/pyproject.toml"
```

Do not run `git pull` after checking out these revisions unless intentionally
testing an upgrade in a separate venv.

## 7. Install Torch and vLLM

Install the tested CUDA 13.2 Torch wheel:

```bash
source "$VENV_SM86/bin/activate"

python -m pip install \
  torch==2.13.0 \
  --index-url https://download.pytorch.org/whl/cu132
```

Verify before building anything:

```bash
python - <<'PY'
import torch

print("torch", torch.__version__)
print("torch CUDA", torch.version.cuda)
print("CUDA available", torch.cuda.is_available())
assert str(torch.__version__).startswith("2.13.0")
assert str(torch.version.cuda).startswith("13.2")
PY
```

Install source-build requirements. `setuptools-rust` is required; omitting it
causes metadata generation to fail.

```bash
python -m pip install \
  "setuptools>=77.0.3,<81.0.0" \
  "setuptools-scm>=8.0" \
  "setuptools-rust>=1.9.0" \
  "cmake>=3.26.1" \
  ninja \
  "packaging>=24.2" \
  jinja2 \
  wheel
```

Build/install the pinned vLLM source against the already installed Torch.
Do not run `use_existing_torch.py`; `--no-build-isolation` plus the installed
build requirements avoids mutating vLLM's tracked requirement files.

```bash
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST=8.6
export VLLM_TARGET_DEVICE=cuda
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

cd "$VLLM_SRC"
python -m pip install -e . --no-build-isolation
```

Verify:

```bash
python - <<'PY'
import pathlib
import torch
import vllm

print("torch", torch.__version__, "CUDA", torch.version.cuda)
print("vLLM", vllm.__version__)
print("vLLM source", pathlib.Path(vllm.__file__).resolve())
PY
```

Expected vLLM version:

```text
0.1.dev20517+gb99dae944
```

If the build changes Torch, stop. Do not continue with a mixed ABI.

## 8. Build the native SM86 ExLlamaV3 extension

Build in the same venv used by vLLM. Choose one installed SM86 GPU for the
build/import check. The physical index is rig-specific.

```bash
python -m pip install marisa-trie

export BUILD_GPU=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$BUILD_GPU"
export TORCH_CUDA_ARCH_LIST=8.6
export EXLLAMAV3_COMMIT=0c49587a7c235e6303a6bbedc8b665272ad3a2ea
export EXLLAMAV3_SRC="$SM86_ROOT/exllamav3"
export EXLLAMAV3_EXT_OUT="$EXL3_WORKSTREAM/build/exllamav3_ext"

cd "$EXL3_WORKSTREAM"
chmod +x scripts/build_exllamav3_ext.sh
./scripts/build_exllamav3_ext.sh
```

Verify the copied extension:

```bash
export VLLM_EXL3_EXT_PATH="$EXL3_WORKSTREAM/build/exllamav3_ext"

python - <<'PY'
import os
import pathlib
import sys
import torch

sys.path.insert(0, os.environ["VLLM_EXL3_EXT_PATH"])
torch_lib = pathlib.Path(torch.__file__).resolve().parent / "lib"
os.environ["LD_LIBRARY_PATH"] = str(torch_lib) + (
    os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
)

import exllamav3_ext

print("extension", exllamav3_ext.__file__)
print("has exl3_gemm", hasattr(exllamav3_ext, "exl3_gemm"))
assert hasattr(exllamav3_ext, "exl3_gemm")
PY
```

## 9. Install and test the vLLM plugin

```bash
source "$VENV_SM86/bin/activate"
cd "$EXL3_WORKSTREAM"

python -m pip install -e plugin
python -m pip install pytest

export VLLM_PLUGINS=vllm_exl3_sm86
export VLLM_EXL3_EXT_PATH="$EXL3_WORKSTREAM/build/exllamav3_ext"
```

The checked-in guard still retains a historical ABI pin. The following drift
override is allowed only for the exact tested stack in this guide:

```bash
export VLLM_EXL3_ALLOW_VLLM_DRIFT=1
```

Run tests on one selected SM86 GPU:

```bash
export CUDA_VISIBLE_DEVICES="$BUILD_GPU"
python -m pytest tests -q
```

Do not continue to model loading unless all tests pass.

## 10. Choose GPUs and TP size

List physical indices:

```bash
nvidia-smi \
  --query-gpu=index,name,compute_cap,memory.total \
  --format=csv
```

Example choices:

```bash
# Four selected physical GPUs:
export GPU_IDS=0,1,2,3
export TP_SIZE=4

# Eight selected physical GPUs:
# export GPU_IDS=0,1,2,3,4,5,6,7
# export TP_SIZE=8
```

After setting `CUDA_VISIBLE_DEVICES`, vLLM sees the selected cards as logical
devices `0..TP_SIZE-1`, regardless of their physical indices:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

python - <<'PY'
import torch

for logical in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(logical)
    print(logical, props.name, f"{props.total_memory / 1024**3:.2f} GiB")
PY
```

Capacity planning:

1. Estimate local weight shard as checkpoint bytes divided by TP size.
2. Add CUDA/vLLM overhead, EXL3 workspaces, activations, NCCL buffers, and KV
   cache.
3. Compare against the smallest card, not aggregate VRAM.
4. Leave margin; disk size divided by TP is only an estimate.
5. Confirm attention/KV heads and linear partitions support the selected TP.

More GPUs do not automatically improve throughput. Measure.

## 11. Validate an EXL3 checkpoint

```bash
export MODEL_DIR="/absolute/path/to/exl3-model"

test -f "$MODEL_DIR/config.json"
test -f "$MODEL_DIR/quantization_config.json"

python "$EXL3_WORKSTREAM/scripts/validate_exl3_checkpoint.py" \
  "$MODEL_DIR" \
  --allow-non-behemoth
```

For a non-Behemoth model, `--allow-non-behemoth` performs general metadata
validation without enforcing Behemoth's layer count and dimensions.

Inspect `quantization_config.json` if validation reports empty
`tensor_storage`; the plugin cannot infer missing packed EXL3 metadata.

## 12. Use a generic crossover policy

The repository's measured crossover table contains RTX 3090 Behemoth shapes.
Do not assume those thresholds are optimal for GA104, GA106, A10, A40, or a
different model.

Create an empty per-rig table so unmatched models use the plugin's conservative
default reconstruct threshold instead of accidentally matching 3090-specific
entries:

```bash
cat > "$SM86_ROOT/crossover_generic.json" <<'JSON'
{
  "schema_version": 1,
  "arch": "sm86",
  "note": "Untuned generic SM86 crossover table",
  "thresholds": []
}
JSON

export VLLM_EXL3_CROSSOVER_JSON="$SM86_ROOT/crossover_generic.json"
```

Tune per model and GPU type only after correctness is established. Keep
separate crossover receipts for different card models.

## 13. First eager launch

Always prove eager serving before trying compilation or CUDA graphs.

```bash
export MODEL_DIR="/absolute/path/to/exl3-model"
export SERVED_MODEL_NAME="my-exl3-model"
export API_KEY="replace-with-a-secret"
export HOST=0.0.0.0
export PORT=8000
export MAX_MODEL_LEN=8192
export GPU_MEMORY_UTILIZATION=0.90
export MAX_NUM_SEQS=4
export MAX_NUM_BATCHED_TOKENS=8192

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export VLLM_PLUGINS=vllm_exl3_sm86
export VLLM_EXL3_EXT_PATH="$EXL3_WORKSTREAM/build/exllamav3_ext"
export VLLM_EXL3_CROSSOVER_JSON="$SM86_ROOT/crossover_generic.json"
export VLLM_EXL3_ALLOW_VLLM_DRIFT=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --api-key "$API_KEY" \
  --host "$HOST" \
  --port "$PORT" \
  --quantization exl3 \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --kv-cache-dtype auto \
  --dtype auto \
  --disable-custom-all-reduce \
  --enforce-eager
```

For one or two GPUs with a supported topology, test vLLM custom all-reduce
separately. It is disabled above for a predictable PCIe-only baseline.

Successful startup must show:

- the `vllm_exl3_sm86` plugin loading;
- all TP ranks initialized;
- each rank loading only its local shard;
- nonzero KV-cache capacity;
- no missing EXL3 tensors or TP alignment failures.

## 14. Smoke-test the API

From another terminal:

```bash
curl \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  http://127.0.0.1:8000/v1/completions \
  -d "{
    \"model\": \"$SERVED_MODEL_NAME\",
    \"prompt\": \"Explain tensor parallel inference in one sentence.\",
    \"max_tokens\": 32,
    \"temperature\": 0
  }"
```

Require coherent output. Garbage or repeated invalid tokens is a correctness
failure, not a performance issue.

## 15. Benchmark

Run the benchmark client from a consistent network location:

```bash
cd "$EXL3_WORKSTREAM"

python scripts/bench_serving_contexts.py \
  --host 127.0.0.1 \
  --api-key "$API_KEY" \
  --model "$SERVED_MODEL_NAME" \
  --label generic-sm86-exl3 \
  --contexts 1024,2048,4096,8192 \
  --output-tokens 256 \
  --warmup-runs 1 \
  --runs 3
```

Do not request a prompt plus completion longer than `MAX_MODEL_LEN`. Add 16K
and 32K only when the server was launched with sufficient model length and KV
capacity.

The benchmark writes JSON and CSV receipts containing TTFT, effective prefill
throughput, decode throughput, TPOT, and streaming inter-chunk latency.

## 16. CUDA graphs are a separate opt-in

Eager mode is the generic supported starting point. The existing prewarm
receipts were created for Behemoth shapes and do not prove graph safety for
another model.

Only after model-specific capture/replay testing:

```bash
export VLLM_EXL3_ALLOW_GRAPHS=1
```

Then remove `--enforce-eager` and supply an explicitly tested vLLM compilation
configuration. If capture fails, outputs change, or replay is unstable, return
to eager mode.

## 17. Persist the environment

Create a host-specific environment file. Do not store the API key in Git:

```bash
cat > "$SM86_ROOT/env.sh" <<EOF
export SM86_ROOT="$SM86_ROOT"
export VENV_SM86="$VENV_SM86"
export VLLM_SRC="$VLLM_SRC"
export REPO="$REPO"
export EXL3_WORKSTREAM="$EXL3_WORKSTREAM"
export EXLLAMAV3_SRC="$EXLLAMAV3_SRC"
export VLLM_PLUGINS=vllm_exl3_sm86
export VLLM_EXL3_EXT_PATH="$EXL3_WORKSTREAM/build/exllamav3_ext"
export VLLM_EXL3_CROSSOVER_JSON="$SM86_ROOT/crossover_generic.json"
export VLLM_EXL3_ALLOW_VLLM_DRIFT=1
export TORCH_CUDA_ARCH_LIST=8.6
EOF
```

For future sessions:

```bash
source "$SM86_ROOT/env.sh"
source "$VENV_SM86/bin/activate"
```

Set `GPU_IDS`, `TP_SIZE`, model paths, server settings, and secrets separately.

## 18. Upgrade/rebuild policy

Treat Python, Torch, CUDA, vLLM, and `exllamav3_ext` as one ABI stack.

Rebuild the extension and rerun all tests whenever any of these changes:

- Python major/minor version;
- Torch version;
- Torch CUDA runtime;
- system CUDA toolkit used for compilation;
- C++ ABI;
- ExLlamaV3 commit.

For a vLLM-only update:

1. create a new venv;
2. install the proposed vLLM revision;
3. rebuild `exllamav3_ext`;
4. install the plugin;
5. run the full test suite;
6. perform eager model load and deterministic generation;
7. benchmark before replacing the working environment.

Never upgrade the production venv in place.

## 19. Common failures

### `No module named setuptools_rust`

```bash
python -m pip install "setuptools-rust>=1.9.0"
```

Then retry the vLLM editable install with `--no-build-isolation`.

### `libc10.so` or another Torch library is missing

Import Torch before `exllamav3_ext` and use the supplied extension build script,
which performs that import in the correct order.

### `No module named marisa_trie` during model compilation

The extension-only installation intentionally uses `--no-deps`, while
ExLlamaV3's final safetensors planner imports `marisa_trie`. Install it in the
active environment and resume the saved conversion:

```bash
python -m pip install marisa-trie
python /path/to/exllamav3/convert.py -w /path/to/work-dir -r
```

Do not delete the work directory; completed quantized modules are reusable.

### Plugin ABI mismatch

Check:

```bash
python - <<'PY'
import torch
import vllm

print(torch.__version__, torch.version.cuda)
print(vllm.__version__)
PY
```

Use `VLLM_EXL3_ALLOW_VLLM_DRIFT=1` only for the exact tested stack documented
above or after completing the full upgrade policy.

### Tensor-parallel group or slicing failure

Confirm that:

- TP size is valid for the model's attention and KV heads;
- TP-local linear dimensions remain 128-aligned;
- `CUDA_VISIBLE_DEVICES` contains exactly the intended number of GPUs;
- `TP_SIZE` equals the number of visible devices.

### GPU out of memory

Reduce model size, increase valid TP size, reduce `MAX_MODEL_LEN`, reduce
`MAX_NUM_SEQS`, or lower `GPU_MEMORY_UTILIZATION`. Aggregate VRAM alone does
not prove that equal TP shards fit the smallest card.

### Extension imports but performance is poor

Confirm all selected cards are SM86, inspect PCIe topology, benchmark NCCL,
and generate model/GPU-specific crossover data. A 3090 crossover table is not
a general SM86 performance profile.

### Checkpoint lacks `tensor_storage`

This plugin requires packed-shape metadata from `quantization_config.json`.
Do not guess tensor layouts or concatenate independently packed Q/K/V tensors.

## 20. Minimum acceptance checklist

- [ ] Every selected GPU reports compute capability 8.6.
- [ ] Python, Torch, Torch CUDA, vLLM, and ExLlamaV3 revisions are recorded.
- [ ] `exllamav3_ext` was built with `TORCH_CUDA_ARCH_LIST=8.6`.
- [ ] Plugin tests pass in the serving venv.
- [ ] EXL3 metadata validation passes.
- [ ] TP-local weight shards fit the smallest GPU with KV/workspace margin.
- [ ] Eager startup succeeds on the intended TP size.
- [ ] Deterministic smoke output is coherent.
- [ ] TTFT, prefill, decode, TPOT, and peak VRAM are captured.
- [ ] CUDA graphs remain disabled until model-specific replay is proven.
