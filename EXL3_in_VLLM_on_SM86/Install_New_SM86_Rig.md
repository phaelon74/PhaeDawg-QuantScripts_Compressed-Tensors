# Install EXL3 in vLLM on this SM86 rig (new venv)

Canonical workspace on this host: **`/home/phaedawg/exl3vllm`**. Use that
root for this bring-up and for future “add EXL3 to vLLM” installs. It is
not a decode-only sandbox.

This is the **new isolated venv** install for the dense EXL3 vLLM plugin on
**this host**: four RTX 3090s (SM86) as tensor-parallel ranks, physical
indices **1, 2, 3, 4**. Physical **0 and 5 stay reserved**. Do not install
into an existing production vLLM environment, `kld-exl3-vllm`,
`EXL3-Behemoth/venv`, or `~/llmcompressor-nightly`.

vLLM **must** come from
[phaelon74/vllm](https://github.com/phaelon74/vllm.git) branch
`feature/score-mode-ppl-kld`. That fork has KLD scoring built in. Do not
clone `vllm-project/vllm` or check out `main`.

After this install, run the decode campaign in
[`../Behemoth-R1-123B-v2_Compressed-Tensors/EXL3-vLLM/HOST_RUN.md`](../Behemoth-R1-123B-v2_Compressed-Tensors/EXL3-vLLM/HOST_RUN.md).
That campaign is what is ready to test. The 30 tok/s gates are **not**
already measured.

## GPU map (do not use 0 or 5)

```
physical nvidia-smi index    role
--------------------------   ------------------------------------------
0                            BUSY — leave alone
1                            ours: extension build, microbench, TP rank 0
2                            ours: TP rank 1
3                            ours: TP rank 2
4                            ours: TP rank 3
5                            BUSY — leave alone
```

Always set:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2,3,4
```

After that, PyTorch/vLLM see four devices numbered 0–3. Single-GPU work
(build import check, pytest CUDA tests, kernel microbench, prewarm) uses
**physical 1 only**:

```bash
export CUDA_VISIBLE_DEVICES=1
# --device 0 in scripts is then logical 0 = physical 1
```

Confirm 0 and 5 still show their existing workers in `nvidia-smi` after
every launch. Stop any serve on 1–4 before microbench or `ncu`.

## 1. Supported scope

### Hardware

This install targets **4× RTX 3090, SM86, TP4**. The CUDA extension is
SM86-only. A100 and A30 are Ampere but **SM80**; do not use this build
there.

The plugin can run on other SM86 cards, but this document’s GPU indices,
VRAM planning, and crossover table are for **this** 4×3090 map.

### Models

Dense-only: vLLM row-parallel, column-parallel, packed QKV, and packed
gate/up. Validated with dense Mistral architecture (Behemoth-R1-123B-v2).

Require:

- `quantization_config.json` with a non-empty EXL3 `tensor_storage` map;
- EXL3 `trellis`, `suh`, and `svh` tensors for quantized linear layers;
- either explicit `mul1`/`mcg` markers or supported implicit codebook metadata;
- TP-local input/output dimensions that remain 128-aligned;
- TP size compatible with attention heads, KV heads, hidden size, and
  intermediate size.

Not currently supported as a generic promise:

- MoE/expert-parallel EXL3;
- SparkInfer;
- multimodal vision/audio towers;
- MTP-specific paths;
- sleep-mode weight unload/reload;
- `state_dict()` resave of EXL3 side tensors;
- CUDA graphs without model-specific prewarm/replay testing.

Kernel overlay (applied at extension build; see §8):

- QTIP GEMV: implicit 3inst (`cb=0`) at **K=2 and K=3** when `EXL3_GEMV>=2`
  or `EXL3_GEMV_3INST=1`. Upstream already allows K=4 3inst. The GEMV
  kernel is 2/3/4 bpw only; **K=5..8 are not instantiated**.
- Optional `EXL3_INT8_GEMV_CB=1` tries int8 activations on 3inst (KLD-gate
  before keeping).
- LUT fill sources are compiled. **Arithmetic `decode_3inst` is still the
  live decode path.** `EXL3_GEMV_LUT` does not change serving until the
  LUT is wired as a GEMV argument (nvcc 20044 without `-rdc`).

Plugin decode path (independent of the LUT):

- CUDA-graph capture sizes `1,2,3,4,5,6,8`;
- fused `exl3_mgemm` for matching gate/up and for matching k/v (same K and
  equal N). Q stays a separate GEMM on mixed-bitrate ArtusDev 4.25;
- n-gram spec decode via `EXL3_NGRAM_SPEC=1` on the 4.25 launcher.

## 2. Tested software stack

Use this stack in the **new** venv. Upgrade one component at a time only
after saving a working environment.

| Component | Tested value |
| --- | --- |
| OS | Linux; Ubuntu 24.04-class environment |
| Python | 3.12 |
| GPU ISA | SM86 (RTX 3090) |
| Serving GPUs | physical **1,2,3,4** (logical 0–3). Never 0 or 5 |
| Torch | pulled by the vLLM precompiled install (`--torch-backend=auto`); record after §7 |
| Torch CUDA runtime | whatever that wheel reports (`torch.version.cuda`); record after §7 |
| CUDA toolkit (`nvcc`) | required later to compile `exllamav3_ext`, not to install vLLM |
| vLLM source | [phaelon74/vllm](https://github.com/phaelon74/vllm.git) |
| vLLM branch | `feature/score-mode-ppl-kld` (KLD built in) |
| vLLM snapshot | `193e8d7ae8a3f60f8b9f1a225dcfed16ed1c66fb` (merge of upstream `main` into the KLD branch, 2026-08-31) |
| vLLM reported version | record `vllm.__version__` after the editable install; it will not be the old `gb99dae944` pin |
| ExLlamaV3 upstream pin | `turboderp-org/exllamav3` `@ 0c49587a7c235e6303a6bbedc8b665272ad3a2ea` |
| ExLlamaV3 kernels | pin + `EXL3-vLLM/kernel/overlay` at build |
| Optional fork | `phaelon74/exllamav3` branch `sm86-decode` |
| Plugin | `vllm-exl3-sm86` from this repository |
| Package installer | `uv pip` after §5 (`uv` bootstrapped into the venv) |
| Hugging Face `datasets` | installed in §5 (KLD / WikiText) |

The plugin’s fail-closed ABI constants still name the older host pin
(`torch 2.9.1`, `vllm 0.1.dev12995+g1f369db5d`). For this KLD-branch
precompiled stack, set `VLLM_EXL3_SKIP_VERSION_GUARD=1` (the 4.25 launcher
already does). `VLLM_EXL3_ALLOW_VLLM_DRIFT=1` is the narrower override if you
prefer it.

Ordinary serving works on this same tree. KLD scoring needs this branch;
upstream `main` does not carry it.

## 3. Verify the rig

Install a sufficiently recent NVIDIA driver and a CUDA development toolkit
containing `nvcc`. You do **not** install Torch yourself. The §7 precompiled
vLLM command brings Torch and its CUDA runtime wheels. `nvcc` is still
required in §8 to compile `exllamav3_ext`.

```bash
nvidia-smi
nvidia-smi \
  --query-gpu=index,name,compute_cap,memory.total \
  --format=csv
nvcc --version
```

Expect six 3090s at compute capability `8.6`. Physical 0 and 5 should already
be occupied. Physical 1–4 should be free enough to take TP4.

Inspect topology:

```bash
nvidia-smi topo -m
```

PCIe-only TP works. NCCL all-reduce on this map is the usual limiter for
prefill, not EXL3 decode.

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

Modern vLLM also builds Rust components **if** you compile it from source.
The §7 precompiled install does not need `rustc`. Skip the rest of this
section unless that command fails and you intentionally compile vLLM locally.

```bash
command -v rustc
command -v cargo
```

Only if both are missing **and** you are compiling vLLM from source:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
  | sh -s -- -y
source "$HOME/.cargo/env"
rustc --version
cargo --version
```

## 5. Create an isolated workspace and venv

Do not reuse `kld-exl3-vllm`, `EXL3-Behemoth/venv/exl3-vllm`, or any other
live serving env. A mixed Torch ABI will fail at `exllamav3_ext` import.

This directory is the install root for EXL3-in-vLLM on this host, including
serving, KLD, and later operator copies of the same recipe:

```bash
export SM86_ROOT="/home/phaedawg/exl3vllm"
export VENV_SM86="$SM86_ROOT/venv"
export VLLM_SRC="$SM86_ROOT/vllm"
export VLLM_BRANCH=feature/score-mode-ppl-kld
# Existing clone is fine; otherwise clone in §6 into $SM86_ROOT.
export REPO="${REPO:-$HOME/Github Repositories/PhaeDawg-QuantScripts_Compressed-Tensors}"
export EXL3_WORKSTREAM="$REPO/Behemoth-R1-123B-v2_Compressed-Tensors/EXL3-vLLM"
export EXLLAMAV3_SRC="$SM86_ROOT/exllamav3"

mkdir -p "$SM86_ROOT"
python3.12 -m venv "$VENV_SM86"
source "$VENV_SM86/bin/activate"

# Last `python -m pip` in this guide: bootstrap uv into the venv.
python -m pip install --upgrade pip wheel uv
export UV_PYTHON="$VENV_SM86/bin/python"
uv pip install datasets

which python
which uv
python --version
uv --version
python -c "import datasets; print('datasets', datasets.__version__)"
```

The expected Python executable is `$VENV_SM86/bin/python`
(`/home/phaedawg/exl3vllm/venv/bin/python`). `datasets` is required for KLD
(WikiText and the Hugging Face loader). After this step, **every** package
install is `uv pip install` with the venv active. Do not use
`python -m pip install` again.

## 6. Clone exact source revisions

vLLM is the KLD fork, not upstream. Clone
[https://github.com/phaelon74/vllm.git](https://github.com/phaelon74/vllm.git)
and check out **`feature/score-mode-ppl-kld`**.

```bash
if [[ ! -d "$REPO/.git" ]]; then
  git clone \
    https://github.com/phaelon74/PhaeDawg-QuantScripts_Compressed-Tensors.git \
    "$REPO"
fi

git clone \
  --branch feature/score-mode-ppl-kld \
  https://github.com/phaelon74/vllm.git \
  "$VLLM_SRC"
git -C "$VLLM_SRC" checkout feature/score-mode-ppl-kld
# Documented snapshot of that branch (2026-08-31):
# 193e8d7ae8a3f60f8b9f1a225dcfed16ed1c66fb
# Optional pin: git -C "$VLLM_SRC" checkout 193e8d7ae8a3f60f8b9f1a225dcfed16ed1c66fb

git clone https://github.com/turboderp-org/exllamav3.git "$EXLLAMAV3_SRC"
git -C "$EXLLAMAV3_SRC" checkout \
  0c49587a7c235e6303a6bbedc8b665272ad3a2ea
```

Do **not** skip the overlay in §8. A vanilla pin without
`kernel/overlay` is the Phase 0 baseline only.

Optional: if `phaelon74/exllamav3` branch `sm86-decode` exists and you want
that checkout as `EXL3-vLLM/exllamav3`:

```bash
cd "$EXL3_WORKSTREAM"
bash scripts/fork_exllamav3.sh
```

`build_exllamav3_ext.sh` prefers that in-tree checkout when present, then
`$EXLLAMAV3_SRC`, then clones the upstream pin. Overlay is applied unless
`EXL3_APPLY_OVERLAY=0`.

Confirm:

```bash
git -C "$VLLM_SRC" rev-parse --abbrev-ref HEAD
git -C "$VLLM_SRC" rev-parse HEAD
git -C "$EXLLAMAV3_SRC" rev-parse HEAD
test -f "$EXL3_WORKSTREAM/plugin/pyproject.toml"
test -f "$EXL3_WORKSTREAM/kernel/overlay/apply_overlay.py"
```

`rev-parse --abbrev-ref HEAD` must print `feature/score-mode-ppl-kld`. If it
prints `main` or a detached SHA from upstream, stop and fix the checkout.
KLD is only on that branch.

Stay on `feature/score-mode-ppl-kld`. Do not `git pull` from
`vllm-project/vllm` `main`; that drops KLD. Pull this branch only when
intentionally upgrading the KLD tree in a separate venv.

## 7. Install vLLM (Torch comes with it)

Do **not** `uv pip install torch` first, and do **not** install vLLM
build-isolation packages (`setuptools-rust`, CMake, Ninja) for this path.
Precompiled vLLM pulls Torch and the CUDA backend that matches this machine.

```bash
source "$VENV_SM86/bin/activate"
export UV_PYTHON="$VENV_SM86/bin/python"

cd "$VLLM_SRC"
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

`--torch-backend=auto` selects the Torch CUDA wheel. `VLLM_USE_PRECOMPILED=1`
uses prebuilt vLLM kernels so this box does not compile them.

Verify both packages came from this tree:

```bash
python - <<'PY'
import pathlib
import torch
import vllm

print("torch", torch.__version__, "CUDA", torch.version.cuda)
print("CUDA available", torch.cuda.is_available())
print("vLLM", vllm.__version__)
print("vLLM source", pathlib.Path(vllm.__file__).resolve())
PY
```

Confirm the import path is under `/home/phaedawg/exl3vllm/vllm`. Record
`torch.__version__`, `torch.version.cuda`, and `vllm.__version__`. Do not
expect the old `0.1.dev20517+gb99dae944` string; this tree is the KLD branch
after merging upstream `main` (`193e8d7…`).

If Torch is missing, CUDA is unavailable, or `vllm.__file__` is not under
`/home/phaedawg/exl3vllm/vllm`, stop. Do not install a second Torch wheel
on top of this environment.

## 8. Build the native SM86 ExLlamaV3 extension

Build in the same venv used by vLLM. Compile and import-check on **physical
GPU 1** only.

```bash
uv pip install marisa-trie

export BUILD_GPU=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$BUILD_GPU"
export TORCH_CUDA_ARCH_LIST=8.6
export EXLLAMAV3_COMMIT=0c49587a7c235e6303a6bbedc8b665272ad3a2ea
export EXLLAMAV3_SRC="$SM86_ROOT/exllamav3"
export EXLLAMAV3_EXT_OUT="$EXL3_WORKSTREAM/build/exllamav3_ext"

cd "$EXL3_WORKSTREAM"
chmod +x scripts/build_exllamav3_ext.sh
# Applies kernel/overlay unless EXL3_APPLY_OVERLAY=0
# Overlay: 3inst GEMV at K=2/3 (not K=5..8), INT8_GEMV_CB gate, LUT fill TUs
./scripts/build_exllamav3_ext.sh
```

The script installs with `--no-build-isolation --no-deps` via `uv pip` when
`uv` is on `PATH`, then copies `exllamav3_ext*.so` to
`$EXL3_WORKSTREAM/build/exllamav3_ext`.

For a Phase 0 vanilla baseline only:

```bash
EXL3_APPLY_OVERLAY=0 ./scripts/build_exllamav3_ext.sh
```

Then rebuild **with** the overlay before Phase 1+ serving.

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
print("has exl3_mgemm", hasattr(exllamav3_ext, "exl3_mgemm"))
assert hasattr(exllamav3_ext, "exl3_gemm")
PY
```

## 9. Install and test the vLLM plugin

```bash
source "$VENV_SM86/bin/activate"
cd "$EXL3_WORKSTREAM"

uv pip install -e plugin
uv pip install pytest

export VLLM_PLUGINS=vllm_exl3_sm86
export VLLM_EXL3_EXT_PATH="$EXL3_WORKSTREAM/build/exllamav3_ext"
export VLLM_EXL3_SKIP_VERSION_GUARD=1
```

Run tests on physical GPU 1:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
python -m pytest tests -q
```

Do not continue to model loading unless CPU tests pass. CUDA tests skip
until the extension loads; that is expected on a CPU-only check, not on
this host.

## 10. Choose GPUs and TP size

On this host the serving map is fixed:

```bash
export GPU_IDS=1,2,3,4
export TP_SIZE=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
```

Never set `GPU_IDS` to include 0 or 5.

After setting `CUDA_VISIBLE_DEVICES`, vLLM sees the selected cards as logical
devices `0..TP_SIZE-1`:

```bash
python - <<'PY'
import torch

print("count", torch.cuda.device_count())
for logical in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(logical)
    print(logical, props.name, f"{props.total_memory / 1024**3:.2f} GiB",
          props.major, props.minor)
PY
```

Expect four GeForce RTX 3090, capability `(8, 6)`. Physical 0 and 5 must
still show their existing workers in `nvidia-smi`.

Capacity planning for ArtusDev 4.25-bpw H6 on TP4 is already measured on
this map (~15.3 GiB weights/GPU plus KV). For any other checkpoint:

1. Estimate local weight shard as checkpoint bytes divided by TP size.
2. Add CUDA/vLLM overhead, EXL3 workspaces, activations, NCCL buffers, and KV
   cache.
3. Compare against the smallest card, not aggregate VRAM.
4. Confirm attention/KV heads and linear partitions support TP=4.

## 11. Validate an EXL3 checkpoint

Default serving candidate on this host:

```bash
export MODEL_DIR=/media/fmodels/ArtusDev/TheDrummer_Behemoth-R1-123B-v2-EXL3/4.25bpw_H6

test -f "$MODEL_DIR/config.json"
test -f "$MODEL_DIR/quantization_config.json"

python "$EXL3_WORKSTREAM/scripts/validate_exl3_checkpoint.py" \
  "$MODEL_DIR" \
  --allow-non-behemoth
```

ArtusDev 4.25 uses implicit 3inst (no `mul1`/`mcg` markers), mixed K=4/5/6
decoder tiles, and an H6 `lm_head`. `--allow-non-behemoth` is required for
that inventory. Expect ~616 EXL3 decoder records and non-empty
`tensor_storage`.

For a non-Behemoth model, the same flag performs general metadata validation
without enforcing Behemoth’s layer count and dimensions.

## 12. Crossover policy

There is **one** crossover file, already in git. It is not created under
`/home/phaedawg/exl3vllm`. On this host:

```text
/home/phaedawg/Github Repositories/PhaeDawg-QuantScripts_Compressed-Tensors/Behemoth-R1-123B-v2_Compressed-Tensors/EXL3-vLLM/manifests/sm86_crossover.json
```

That is `$EXL3_WORKSTREAM/manifests/sm86_crossover.json`. The launcher reads
it via `VLLM_EXL3_CROSSOVER_JSON`. Leave this checked-in 3090 table in place
for the first eager serve.

`select_sm86_crossover.py` **overwrites that same file**. Only re-derive after
a microbench that includes prefill-sized M (128–1024+). An M=1,2,4,8 sweep
never sees reconstruct win, so every threshold would be `null`. Do not run
the re-derive until you have that wider sweep.

This is a 3090 box. Point the env at the existing table:

```bash
export VLLM_EXL3_CROSSOVER_JSON="$EXL3_WORKSTREAM/manifests/sm86_crossover.json"
ls -l "$VLLM_EXL3_CROSSOVER_JSON"
```

Optional later (full M sweep, then overwrite the same path):

```bash
export CUDA_VISIBLE_DEVICES=1
python "$EXL3_WORKSTREAM/scripts/kernel_microbench.py" \
  --device 0 \
  --bitrates 4,5,6 \
  --m 1,2,4,8,16,32,64,128,256,512,1024 \
  --output "$EXL3_WORKSTREAM/results/kernel_microbench.json"

python "$EXL3_WORKSTREAM/scripts/select_sm86_crossover.py" \
  --microbench "$EXL3_WORKSTREAM/results/kernel_microbench.json" \
  --output "$EXL3_WORKSTREAM/manifests/sm86_crossover.json"
```

Do not copy GLM/SM120’s default M=144.

## 13. First eager launch

Always prove eager serving before compilation or CUDA graphs. Use the 4.25
launcher with eager forced so capture sizes and spec decode stay off:

```bash
source "$VENV_SM86/bin/activate"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2,3,4
export VLLM_PLUGINS=vllm_exl3_sm86
export VLLM_EXL3_EXT_PATH="$EXL3_WORKSTREAM/build/exllamav3_ext"
export VLLM_EXL3_SKIP_VERSION_GUARD=1
export VLLM_EXL3_CROSSOVER_JSON="$EXL3_WORKSTREAM/manifests/sm86_crossover.json"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export EXL3_MODEL_DIR="${MODEL_DIR:-/media/fmodels/ArtusDev/TheDrummer_Behemoth-R1-123B-v2-EXL3/4.25bpw_H6}"
export EXL3_ENFORCE_EAGER=1

LAUNCH="$REPO/VLLM-Launch_Scripts/behemoth123b-r1-v2-exl3-4p25.sh"
bash "$LAUNCH" "$API_KEY"
```

The launcher already pins `CUDA_VISIBLE_DEVICES=1,2,3,4`. Successful startup
must show:

- the `vllm_exl3_sm86` plugin loading;
- all four TP ranks initialized;
- each rank loading only its local shard;
- nonzero KV-cache capacity;
- no missing EXL3 tensors or TP alignment failures;
- after overlay + graphs later: `fused exl3_mgemm decode enabled for K/V pairs`
  when that layer’s k and v share bitrate.

Generic `vllm serve` (same GPUs, eager) if you are not using the launcher:

```bash
export MODEL_DIR="$EXL3_MODEL_DIR"
export SERVED_MODEL_NAME="Behemoth-R1-123B-v2-EXL3-4.25-H6"
export HOST=0.0.0.0
export PORT=8000
export MAX_MODEL_LEN=8192
export GPU_MEMORY_UTILIZATION=0.90
export MAX_NUM_SEQS=4
export MAX_NUM_BATCHED_TOKENS=8192

vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --api-key "$API_KEY" \
  --host "$HOST" \
  --port "$PORT" \
  --quantization exl3 \
  --tensor-parallel-size 4 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --kv-cache-dtype auto \
  --dtype auto \
  --disable-custom-all-reduce \
  --enforce-eager
```

Keep first-bring-up `MAX_MODEL_LEN` modest. Production on this host uses
54272 once eager is green.

## 14. Smoke-test the API

From another terminal:

```bash
curl \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  http://127.0.0.1:8000/v1/completions \
  -d "{
    \"model\": \"Behemoth-R1-123B-v2-EXL3-4.25-H6\",
    \"prompt\": \"Explain tensor parallel inference in one sentence.\",
    \"max_tokens\": 32,
    \"temperature\": 0
  }"
```

Require coherent output. Garbage or repeated invalid tokens is a correctness
failure, not a performance issue.

## 15. Benchmark

Run the client on this host or the same LAN path every time:

```bash
cd "$EXL3_WORKSTREAM"

python scripts/bench_serving_contexts.py \
  --host 127.0.0.1 \
  --api-key "$API_KEY" \
  --model Behemoth-R1-123B-v2-EXL3-4.25-H6 \
  --label sm86-exl3-4p25-eager \
  --contexts 1024,2048,4096 \
  --output-tokens 256 \
  --warmup-runs 1 \
  --runs 3
```

Do not request a prompt plus completion longer than `MAX_MODEL_LEN`. Add 16K
and 32K only when the server was launched with sufficient model length and KV
capacity.

The benchmark writes JSON and CSV receipts containing TTFT, effective prefill
throughput, decode throughput, TPOT, and streaming inter-chunk latency.

Targets from the decode plan (not yet claimed):

- G0 (env only): ≥21 tok/s at 1K/256, or a documented no with `ncu` + GB/s;
- G1 (k/v fusion): ≥23 tok/s, prefill ≥320;
- G2 (spec decode): ≥30 tok/s at 1K/256;
- G3 (LUT, **not live yet**): `gate` M=1 ≥550 GB/s and ≥26 tok/s without spec.

## 16. CUDA graphs are a separate opt-in

Eager mode is the first supported operating point. After prewarm and replay
on physical GPU 1:

```bash
export CUDA_VISIBLE_DEVICES=1
python "$EXL3_WORKSTREAM/scripts/prewarm_kernels.py" --device 0
python "$EXL3_WORKSTREAM/scripts/graph_replay_stress.py" \
  --device 0 --all-shapes --mgemm --steps 1000
```

Then serve with graphs. Capture sizes must cover spec-decode M:

```text
[1,2,3,4,5,6,8]
```

```bash
unset EXL3_ENFORCE_EAGER
export VLLM_EXL3_ALLOW_GRAPHS=1
export EXL3_CUDAGRAPH_CAPTURE_SIZES='[1,2,3,4,5,6,8]'
bash "$LAUNCH" "$API_KEY"
```

The 4.25 launcher defaults to those sizes and `full_decode_only` when eager
is off. If capture fails, outputs change, or replay is unstable, return to
`EXL3_ENFORCE_EAGER=1`.

## 17. Speculative decode (after graphs)

Kernel time is nearly flat in M. N-gram is zero extra VRAM:

```bash
export EXL3_NGRAM_SPEC=1
export EXL3_NGRAM_TOKENS=3
bash "$LAUNCH" "$API_KEY"
```

Or run the A/B helper (stop any existing serve first):

```bash
export LAUNCH="$REPO/VLLM-Launch_Scripts/behemoth123b-r1-v2-exl3-4p25.sh"
bash "$EXL3_WORKSTREAM/scripts/host_phase2_specdecode.sh"
```

Draft-model path: verify tokenizer identity first
(`scripts/verify_draft_tokenizer.py`), then `EXL3_DRAFT_MODEL=...` with the
same helper. Must fit beside ~15.3 GiB/GPU of 4.25 weights plus 32K KV.

## 18. Kernel environment (overlay)

Export these before `vllm serve`. Sweep them in Phase 0; do not freeze
`EXL3_GEMV=2` until that A/B exists.

| Var | Default | Meaning |
| --- | --- | --- |
| `EXL3_GEMV` | 1 | 0=off, 1=heuristic, 2=force eligible GEMV |
| `EXL3_GEMV_3INST` | unset | 1=allow cb=0 GEMV at K≠4 (K=2/3 only) |
| `EXL3_GEMV_SMEM` | unset | 0/1/-1 smem heuristic |
| `EXL3_GEMV_LUT` | reserved | LUT fill may compile; decode still uses arithmetic 3inst |
| `EXL3_INT8_GEMV` | upstream | 0=off |
| `EXL3_INT8_GEMV_CB` | 0 | 1=try int8 activations on 3inst (KLD-gate) |
| `EXL3_INT8_GEMV_MAX_K` | arch default | 5 on Ampere |
| `EXL3_NGRAM_SPEC` | 0 | 1=ngram speculative decode |
| `EXL3_CUDAGRAPH_CAPTURE_SIZES` | `[1,2,3,4,5,6,8]` | must cover spec M |
| `VLLM_EXL3_DISABLE_MGEMM` | unset | 1=disable fused gate/up and k/v |

Power A/B (Phase 0) may change power limits on **GPUs 1–4 only**:

```bash
nvidia-smi -i 1,2,3,4 -pl 350   # or 270
```

Never pass `-i 0` or `-i 5`.

## 19. Decode test campaign (this is the actual test)

With the new venv green (plugin tests, eager smoke, then graphs), follow
[`HOST_RUN.md`](../Behemoth-R1-123B-v2_Compressed-Tensors/EXL3-vLLM/HOST_RUN.md)
on this venv, not `kld-exl3-vllm`:

```bash
source "$VENV_SM86/bin/activate"
export EXL3="$EXL3_WORKSTREAM"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_EXL3_EXT_PATH="$EXL3/build/exllamav3_ext"
export VLLM_EXL3_SKIP_VERSION_GUARD=1
export PYTHONPATH="$VLLM_EXL3_EXT_PATH:$EXL3/plugin/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")'):${LD_LIBRARY_PATH:-}"
cd "$EXL3"
```

Order:

0. Stop serve. `CUDA_VISIBLE_DEVICES=1` then `bash scripts/host_phase0.sh`.
1. Overlay rebuild (already done in §8 if `EXL3_APPLY_OVERLAY` was not 0).
2. Microbench + mgemm + prewarm + graph replay on logical device 0.
3. `scripts/host_phase2_specdecode.sh` for n-gram / optional draft.
4. Do **not** treat `EXL3_GEMV_LUT=1` as a decode A/B until the LUT is wired.
5. `scripts/host_quality_gate.sh` and `scripts/validate_exl3_generality.py`.

Host discipline: `CUDA_VISIBLE_DEVICES=1,2,3,4` for serve; physical 0 and 5
reserved. After a CUDA IMA, start a fresh process.

## 20. Persist the environment

Create a host-specific environment file. Do not store the API key in Git:

```bash
cat > "$SM86_ROOT/env.sh" <<EOF
export SM86_ROOT="/home/phaedawg/exl3vllm"
export VENV_SM86="$SM86_ROOT/venv"
export VLLM_SRC="$SM86_ROOT/vllm"
export VLLM_BRANCH=feature/score-mode-ppl-kld
export REPO="$REPO"
export EXL3_WORKSTREAM="$EXL3_WORKSTREAM"
export EXLLAMAV3_SRC="$EXLLAMAV3_SRC"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2,3,4
export VLLM_PLUGINS=vllm_exl3_sm86
export VLLM_EXL3_EXT_PATH="$EXL3_WORKSTREAM/build/exllamav3_ext"
export VLLM_EXL3_CROSSOVER_JSON="$EXL3_WORKSTREAM/manifests/sm86_crossover.json"
export VLLM_EXL3_SKIP_VERSION_GUARD=1
export TORCH_CUDA_ARCH_LIST=8.6
export UV_PYTHON="$VENV_SM86/bin/python"
EOF
```

For future sessions:

```bash
source "$SM86_ROOT/env.sh"
source "$VENV_SM86/bin/activate"
```

Set model paths, server settings, and secrets separately. Override
`CUDA_VISIBLE_DEVICES=1` only for single-GPU scripts.

## 21. Upgrade/rebuild policy

Treat Python, Torch, CUDA, vLLM, and `exllamav3_ext` as one ABI stack.

Rebuild the extension and rerun all tests whenever any of these changes:

- Python major/minor version;
- Torch version;
- Torch CUDA runtime;
- system CUDA toolkit used for compilation;
- C++ ABI;
- ExLlamaV3 commit or `kernel/overlay`.

For a vLLM-only update:

1. create a new venv under `/home/phaedawg/exl3vllm` or a sibling root;
2. install from `feature/score-mode-ppl-kld` again with
   `VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto`
   (never bare upstream `main`, never a separate Torch wheel);
3. rebuild `exllamav3_ext`;
4. install the plugin;
5. run the full test suite;
6. perform eager model load and deterministic generation;
7. benchmark before replacing the working environment.

Never upgrade the production venv in place. Never install this stack into
`kld-exl3-vllm` or `EXL3-Behemoth/venv`.

## 22. Common failures

### `No module named setuptools_rust`

That error means you compiled vLLM from source instead of using precompiled
wheels. Prefer:

```bash
cd "$VLLM_SRC"
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

Only if you must compile vLLM locally:

```bash
uv pip install "setuptools-rust>=1.9.0"
```

### `libc10.so` or another Torch library is missing

Import Torch before `exllamav3_ext` and use the supplied extension build script,
which performs that import in the correct order.

### `No module named marisa_trie` during model compilation

The extension-only installation intentionally uses `--no-deps`, while
ExLlamaV3's final safetensors planner imports `marisa_trie`. Install it in the
active environment and resume the saved conversion:

```bash
uv pip install marisa-trie
python /path/to/exllamav3/convert.py -w /path/to/work-dir -r
```

Do not delete the work directory; completed quantized modules are reusable.
Conversion on this host must use `CUDA_VISIBLE_DEVICES=1`, never physical 0.

### Plugin ABI mismatch

```bash
python - <<'PY'
import torch
import vllm

print(torch.__version__, torch.version.cuda)
print(vllm.__version__)
PY
```

This new venv is the KLD branch at `/home/phaedawg/exl3vllm/vllm` installed
with `VLLM_USE_PRECOMPILED=1` and `--torch-backend=auto`. Keep
`VLLM_EXL3_SKIP_VERSION_GUARD=1` until `constants.py` is retargeted. If
`vllm.__file__` is not under `/home/phaedawg/exl3vllm/vllm`, you built the
wrong tree.

### Tensor-parallel group or slicing failure

Confirm that:

- TP size is 4;
- `CUDA_VISIBLE_DEVICES` is exactly `1,2,3,4`;
- TP-local linear dimensions remain 128-aligned;
- physical 0 and 5 were not included.

### GPU out of memory

Reduce `MAX_MODEL_LEN`, reduce `MAX_NUM_SEQS`, or lower
`GPU_MEMORY_UTILIZATION`. Do not add GPU 0 or 5.

### Extension imports but performance is poor

Confirm the overlay was applied (`EXL3_APPLY_OVERLAY` was not 0), `ncu` the
M=1 gate kernel, and re-derive crossover. Do not expect LUT bandwidth until
that path is wired. Stop serve before microbench so it does not contend on
1–4.

### Checkpoint lacks `tensor_storage`

This plugin requires packed-shape metadata from `quantization_config.json`.
Do not guess tensor layouts or concatenate independently packed Q/K/V tensors.

### CUDA graph capture allocator / autotune

`exl3_gemv_int8` does a one-time 16 MB `cudaMalloc`, and `exl3_gemm` runs
`CoopKernelAutotuner` on a cache miss. Both are fatal inside capture. Run
`prewarm_kernels.py` and `graph_replay_stress.py` before
`VLLM_EXL3_ALLOW_GRAPHS=1`.

## 23. Minimum acceptance checklist

- [ ] Physical 0 and 5 remain untouched; serving uses 1,2,3,4 only.
- [ ] Workspace is `/home/phaedawg/exl3vllm` with an isolated venv.
- [ ] `uv` and `datasets` are installed; later packages used `uv pip install`.
- [ ] Every selected GPU reports compute capability 8.6.
- [ ] Python 3.12; Torch and vLLM from
      `VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto`
      on `feature/score-mode-ppl-kld` (versions recorded; not upstream `main`).
- [ ] `exllamav3_ext` was built with `TORCH_CUDA_ARCH_LIST=8.6` on GPU 1.
- [ ] Overlay applied (or an explicit `EXL3_APPLY_OVERLAY=0` Phase 0 baseline).
- [ ] Plugin tests pass in the new venv.
- [ ] EXL3 metadata validation passes for the 4.25 checkpoint.
- [ ] Eager TP4 smoke output is coherent.
- [ ] TTFT, prefill, decode, TPOT, and peak VRAM are captured.
- [ ] CUDA graphs remain disabled until prewarm + replay on GPU 1 pass.
- [ ] Graph capture sizes are `[1,2,3,4,5,6,8]` when graphs are enabled.
- [ ] Phase 0 receipts exist under `EXL3-vLLM/results/phase0/` before claiming G0.
