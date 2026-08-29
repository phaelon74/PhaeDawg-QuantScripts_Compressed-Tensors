# Install and run on d011sd02 (6× RTX 3090)

This is the hardware install guide for the dense EXL3 vLLM workstream.
Software lives in this directory. Do not install anything into
`~/llmcompressor-nightly`. Do not convert Behemoth until the kernel
microbench and small-Mistral gates pass.

Host snapshot (2026-08-28):

| Item | Value |
| --- | --- |
| Host | `d011sd02` |
| GPUs | 6× RTX 3090 (SM86), 24 GiB each |
| Driver | 580.95.05 |
| CUDA (driver) | 13.0 |
| Python | 3.12.3 |
| Busy GPUs | **physical 0 and 5** (`VLLM::Worker_TP0/TP1`, ~23760 MiB each) |
| Our GPUs | **physical 1, 2, 3, 4** as logical `0,1,2,3` |

## GPU map (do not use 0 or 5)

```
physical nvidia-smi index    role
--------------------------   ------------------------------------------
0                            BUSY — leave alone
1                            ours: conversion + TP rank 0
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

After that, PyTorch/vLLM see four devices numbered 0–3. Conversion uses a
**single** card: keep `CUDA_VISIBLE_DEVICES=1` so physical GPU 1 is the only
visible device (logical 0). Never let convert scripts default to physical 0.

The serve script already defaults to `1,2,3,4`. Convert scripts default to
`0` — you **must** override them.

## Workspace and venvs

Work directory on this host: `/home/phaedawg/EXL3-Behemoth`.
The two venvs **already exist** under `venv/`:

```
/home/phaedawg/EXL3-Behemoth/venv/exl3-vllm
/home/phaedawg/EXL3-Behemoth/venv/exllamav3-convert
```

Set these at the start of every session:

```bash
export EXL3_HOME=/home/phaedawg/EXL3-Behemoth
export VENV_VLLM=$EXL3_HOME/venv/exl3-vllm
export VENV_CONVERT=$EXL3_HOME/venv/exllamav3-convert
source "$VENV_VLLM/bin/activate"    # or VENV_CONVERT when converting
```

| Env | Path | Purpose | Contents |
| --- | --- | --- | --- |
| vLLM | `$EXL3_HOME/venv/exl3-vllm` | Serve, plugin, kernels, tests, KLD, benches | Torch **2.9.1+cu130**, KLD vLLM tree `1f369db5d`, `exllamav3_ext` only, this plugin |
| convert | `$EXL3_HOME/venv/exllamav3-convert` | Quantize small Mistral, then Behemoth | Full ExLlamaV3 at commit `0c49587a7c235e6303a6bbedc8b665272ad3a2ea` |

Do **not** put conversion and vLLM in one env. Do **not** `pip install` current
vLLM. The plugin version-guards this host’s captured ABI:

- `vllm == 0.1.dev12995+g1f369db5d` (tree `1f369db5d` at `$EXL3_HOME/kld-vllm`)
- `torch` starts with `2.9.1`
- Torch CUDA starts with `13.0`

Populate those existing venvs with the pins below. Do not create `~/exl3-vllm`
or `~/exllamav3-convert`.

## Behemoth EXL3 checkpoints and measured accuracy

Both ArtusDev checkpoints load and generate under the native SM86 plugin on
TP4 RTX 3090. Authoritative eager KLD uses 204,700 positions, context 2048,
stride 512, and the reference logits under `/media/netmodels/ref_logits/`.

| Checkpoint | `du -sh` | Mean KLD | Result |
| --- | ---: | ---: | --- |
| ArtusDev `3.5bpw_H6` | 51G (50.70 GiB exact) | 0.045794 twice | Accepted memory-first |
| ArtusDev `4.25bpw_H6` | 62G | **0.015800** | Current quality/size winner |
| Mixed AutoRound GS32 | 72G | 0.034004 | Former quality reference |

The 4.25-bpw checkpoint lowers KLD by 53.5% relative to the 72G AutoRound
checkpoint while using 10G less rounded disk space. This does not establish
serving speed; benchmark both EXL3 candidates after the 4.5-bpw KLD run.
Machine-readable receipt:
[`results/behemoth_exl3_kld.json`](results/behemoth_exl3_kld.json).

```bash
export MODEL_DIR=/media/fmodels/ArtusDev/TheDrummer_Behemoth-R1-123B-v2-EXL3/3.5bpw_H6
export BF16_DIR=/media/fmodels/TheDrummer/Behemoth-R1-123B-v2
ls "$MODEL_DIR/quantization_config.json" "$MODEL_DIR/config.json"
ls "$BF16_DIR/config.json"
du -sb "$MODEL_DIR"
```

The ArtusDev checkpoints use ExLlamaV3 0.0.6 metadata, implicit codebooks
(no `mul1`/`mcg` marker tensors), K3/K4/K5 decoder tiles, and an H6
`lm_head`. The plugin has loaded and scored both successfully.

The local 4.5-bpw experiment is converted directly from BF16 with the pinned
ExLlamaV3 source, its supported default 250×2048 calibration, H6 `lm_head`,
and explicit `mul1`. Never requantize the 4.25-bpw checkpoint. `mul1` is selected
because the Ampere INT8 GEMV path covers decoder tiles at K≤5.

BF16 is on this host at `/media/fmodels/TheDrummer/Behemoth-R1-123B-v2/`
(~245 GiB). You do **not** need it unless you later remake the quant. You
still want disk for KLD reference logits (~25 GiB) if they are not already
here.

This plugin tree (`EXL3-vLLM/`) may live in the git clone, not necessarily
under `$EXL3_HOME`. Set `EXL3` to the directory that contains `plugin/` and
`scripts/`:

```bash
export EXL3_HOME=/home/phaedawg/EXL3-Behemoth
export VENV_VLLM=$EXL3_HOME/venv/exl3-vllm
export VENV_CONVERT=$EXL3_HOME/venv/exllamav3-convert
# example if the clone is the usual Windows-sync path; change if yours differs:
export REPO="$HOME/Github Repositories/PhaeDawg-QuantScripts_Compressed-Tensors"
export EXL3="$REPO/Behemoth-R1-123B-v2_Compressed-Tensors/EXL3-vLLM"
cd "$EXL3"
```

If you copied or cloned the workstream into `$EXL3_HOME`, use
`export EXL3=$EXL3_HOME` (or `$EXL3_HOME/EXL3-vLLM`) instead.

---

## 0. One-time system packages

```bash
sudo apt-get update
sudo apt-get install -y build-essential git ninja-build python3-venv python3-dev
# nvcc must match CUDA 13.0 toolkit used to build the vLLM wheel
nvcc --version
```

You need a CUDA 13.0 toolkit (not only the driver) to compile `exllamav3_ext`
against Torch `2.9.1+cu130`.

---

## 1. Populate `$VENV_VLLM` (pinned vLLM, not latest)

The venv already exists. Activate it and install pins; do not `python3 -m venv` again.

```bash
export EXL3_HOME=/home/phaedawg/EXL3-Behemoth
export VENV_VLLM=$EXL3_HOME/venv/exl3-vllm
source "$VENV_VLLM/bin/activate"
pip install -U pip wheel setuptools ninja
```

### Option A — copy pins from an existing good env (preferred)

If `~/kld-nightly-vllm` exists **on this machine** and already reports the
pinned versions:

```bash
source ~/kld-nightly-vllm/bin/activate
python -c "import torch,vllm; print(torch.__version__, torch.version.cuda, vllm.__version__)"
# must print: 2.11.0+cu130  13.0  0.23.1rc1.dev1114+g7644b1d0a
```

Then install the same torch/vLLM/transformers stack into `$VENV_VLLM` the
same way you built `kld-nightly-vllm` (source tree at SHA `7644b1d0a`, or the
same local wheel). Do not `pip install vllm` from PyPI.

### Option B — build vLLM from the pinned SHA

```bash
source "$VENV_VLLM/bin/activate"
# Install Torch 2.11.0+cu130 first (same index/URL you used for kld-nightly-vllm).
# Example shape only — use the exact cu130 wheel recipe from that env:
#   pip install torch==2.11.0+cu130 --index-url <your cu130 index>
git clone https://github.com/vllm-project/vllm.git ~/src/vllm
git -C ~/src/vllm fetch --all
git -C ~/src/vllm checkout 7644b1d0a
cd ~/src/vllm
# Use the same build flags / VLLM_PRECOMPILED / uv recipe as kld-nightly-vllm.
# After install:
python -c "import torch,vllm; print(torch.__version__, torch.version.cuda, vllm.__version__)"
```

If `vllm.__version__` is not exactly `0.23.1rc1.dev1114+g7644b1d0a`, the plugin
will refuse to load unless you set `VLLM_EXL3_ALLOW_VLLM_DRIFT=1` (rebase risk).
Fix the pin instead of skipping the guard.

Sanity on **our** GPUs:

```bash
source "$VENV_VLLM/bin/activate"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2,3,4
python - <<'PY'
import torch
print("count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
PY
# expect 4 devices, each GeForce RTX 3090, capability (8, 6)
```

Physical 0 and 5 must still show their existing vLLM workers in `nvidia-smi`.

### Captured ABI on d011sd02 (this host)

Do **not** rebuild vLLM to the old 0.23.1 / Torch 2.11 pin. This box is:

| Item | Value |
| --- | --- |
| venv | `/home/phaedawg/EXL3-Behemoth/venv/exl3-vllm` |
| vLLM source | `/home/phaedawg/EXL3-Behemoth/kld-vllm` |
| git | `1f369db5d5680355e8909df56e77592c55ebdbf9` |
| `vllm.__version__` | `0.1.dev12995+g1f369db5d` |
| torch | `2.9.1+cu130` |
| CUDA | 13.0 |
| GPUs | 4× 3090 SM86 via `CUDA_VISIBLE_DEVICES=1,2,3,4` |

This is a KLD-patched vLLM tree, not upstream `g7644b1d0a`. The plugin version
guard is retargeted to this ABI. Loader/graph APIs may still differ; fix the
plugin if import or load fails. Build `exllamav3_ext` against **this** Torch.

Capture the ABI:

```bash
cd "$EXL3"
source "$VENV_VLLM/bin/activate"
export CUDA_VISIBLE_DEVICES=1,2,3,4
./scripts/capture_manifest.sh
```

That writes `manifests/stack.captured.json`.

---

## 2. Build SM86 `exllamav3_ext` into `$VENV_VLLM`

Still in `$VENV_VLLM`. This compiles native SASS for 8.6 only, against
Torch `2.9.1+cu130`.

```bash
source "$VENV_VLLM/bin/activate"
cd "$EXL3"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export TORCH_CUDA_ARCH_LIST=8.6
export EXLLAMAV3_SRC=${EXLLAMAV3_SRC:-$HOME/src/exllamav3}
./scripts/build_exllamav3_ext.sh
```

The script clones/checkouts ExLlamaV3 `0c49587a7c235e6303a6bbedc8b665272ad3a2ea`,
installs with `--no-build-isolation --no-deps`, and copies `exllamav3_ext*.so`
to `$EXL3/build/exllamav3_ext`.

```bash
export VLLM_EXL3_EXT_PATH="$EXL3/build/exllamav3_ext"
pip install -e "$EXL3/plugin"
pip install pytest
export VLLM_PLUGINS=vllm_exl3_sm86
cd "$EXL3"
python -m pytest tests -q
```

CPU tests should pass without the 123B checkpoint. CUDA tests skip until the
extension loads.

Put these in `~/.bashrc` or `$EXL3_HOME/env-exl3.sh` and source every session:

```bash
export EXL3_HOME=/home/phaedawg/EXL3-Behemoth
export VENV_VLLM=$EXL3_HOME/venv/exl3-vllm
export VENV_CONVERT=$EXL3_HOME/venv/exllamav3-convert
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2,3,4
export VLLM_PLUGINS=vllm_exl3_sm86
export VLLM_EXL3_EXT_PATH="$EXL3/build/exllamav3_ext"
# set EXL3 to the workstream dir that contains plugin/ and scripts/
```

---

## 3. Kernel microbench gate (before any 123B convert)

```bash
source "$VENV_VLLM/bin/activate"
cd "$EXL3"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export VLLM_EXL3_EXT_PATH="$EXL3/build/exllamav3_ext"
python scripts/kernel_microbench.py --device 0 --fail-on-gate
```

`--device 0` is **logical** 0 = physical GPU 1.

Gates:

- output parity vs EXL3 reconstruct
- M=1 ≤ 1.25× Marlin (when Marlin is available in this env)
- M=8–32 ≤ 1.5×
- M≥1024 ≤ 1.15×

If this fails, **stop**. Do not convert Behemoth. Profile with Nsight Compute.

After a passing run, fill SM86 crossover thresholds (do not copy GLM’s 144):

```bash
python scripts/select_sm86_crossover.py \
  --microbench results/kernel_microbench.json \
  --output manifests/sm86_crossover.json
```

Optional Marlin speed snapshot of the frozen 72G mixed checkpoint (same four
GPUs, still in `$VENV_VLLM`):

```bash
export CUDA_VISIBLE_DEVICES=1,2,3,4
MODEL_DIR=/media/fmodels/TheHouseOfTheDude/Behemoth-R1-123B-v2/NewQuants/AutoRound_GS32_Mixed_72G \
  ./scripts/capture_marlin_baseline.sh
```

---

## 4. Populate `$VENV_CONVERT`

The convert venv already exists at `$EXL3_HOME/venv/exllamav3-convert`.
Full ExLlamaV3 package. No vLLM.

```bash
export EXL3_HOME=/home/phaedawg/EXL3-Behemoth
export VENV_CONVERT=$EXL3_HOME/venv/exllamav3-convert
source "$VENV_CONVERT/bin/activate"
pip install -U pip wheel ninja
# Torch here only needs to run convert.py on one 3090. CUDA 13.0 wheels are fine.
# If ~/src/exllamav3 already exists from step 2, reuse it:
git -C ~/src/exllamav3 checkout 0c49587a7c235e6303a6bbedc8b665272ad3a2ea
pip install -e ~/src/exllamav3
python -c "import exllamav3, exllamav3_ext; print('ok', exllamav3.__file__)"
```

---

## 5. Small dense Mistral EXL3 (still before Behemoth)

Pick a local dense Mistral tree (example: Mistral-7B-Instruct-v0.3).

**Convert** (physical GPU 1 only):

```bash
source "$VENV_CONVERT/bin/activate"
cd "$EXL3"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export CONVERT_PY=$HOME/src/exllamav3/convert.py
export MISTRAL_IN_DIR=/path/to/local/Mistral-7B-Instruct-v0.3
./scripts/convert_small_mistral.sh
```

**Load TP1 then TP4** in the vLLM env (physical 1–4):

```bash
source "$VENV_VLLM/bin/activate"
cd "$EXL3"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2,3,4
export VLLM_PLUGINS=vllm_exl3_sm86
export VLLM_EXL3_EXT_PATH="$EXL3/build/exllamav3_ext"
python scripts/validate_small_mistral.py \
  --model "$EXL3/build/mistral-exl3-small" \
  --tp 1,4
```

Require: no missing tensors, deterministic greedy, TP1/TP4 agree, host RSS
not ~4× the checkpoint. If each worker keeps a full copy, stop; do not
load the 123B ArtusDev tree yet.

---

## 6. Validate ArtusDev and create the 4.5-bpw candidate

Inventory either downloaded checkpoint:

```bash
source "$VENV_VLLM/bin/activate"
cd "$EXL3"
export MODEL_DIR=/media/fmodels/ArtusDev/TheDrummer_Behemoth-R1-123B-v2-EXL3/3.5bpw_H6
# --allow-non-behemoth: this checkpoint uses K5 on some decoder tensors and
# has no mul1 markers, so the strict K3/K4+mul1 inventory check would fail.
python scripts/validate_exl3_checkpoint.py "$MODEL_DIR" --allow-non-behemoth
```

Expect ~616 EXL3 decoder records, H6 `lm_head`, non-empty `tensor_storage`,
size near 54.4 GB. Then go to step 7 with this `MODEL_DIR`.

### 6.1 Preflight the 4.5-bpw conversion

Convert from the original BF16 model, never from an existing quant. The
4.25-bpw and 5.0-bpw ArtusDev sizes imply approximately 67G rounded for
4.5-bpw, but this is only an estimate. The work directory must hold another
complete quantized copy, so plan for at least 150 GiB free when work and output
share a filesystem.

```bash
export WORK=/home/phaedawg/kld-exl3-vllm
export EXL3=$WORK/PhaeDawg-QuantScripts_Compressed-Tensors/Behemoth-R1-123B-v2_Compressed-Tensors/EXL3-vLLM
export EXLLAMAV3_SRC=$HOME/src/exllamav3
export IN_DIR=/media/fmodels/TheDrummer/Behemoth-R1-123B-v2
export WORK_DIR=/media/fmodels/TheHouseOfTheDude/Behemoth-R1-123B-v2/NewQuants/EXL3_4p5_H6_mul1.work
export OUT_DIR=/media/fmodels/TheHouseOfTheDude/Behemoth-R1-123B-v2/NewQuants/EXL3_4p5_H6_mul1

test -f "$IN_DIR/config.json" || test -f "$IN_DIR/main/config.json"
test -f "$EXLLAMAV3_SRC/convert.py"
git -C "$EXLLAMAV3_SRC" rev-parse HEAD
# Must be 0c49587a7c235e6303a6bbedc8b665272ad3a2ea
df -h "$(dirname "$WORK_DIR")" "$(dirname "$OUT_DIR")"
```

The known-working `kld-exl3-vllm` environment already has the pinned
ExLlamaV3 extension and all converter imports. It may be used directly; do not
install or upgrade packages during conversion. Confirm before starting:

```bash
source /home/phaedawg/kld-exl3-vllm/kld-exl3-vllm/bin/activate
python -c "import torch, exllamav3, exllamav3_ext; print(torch.__version__, torch.version.cuda, exllamav3_ext.__file__)"
```

### 6.2 Start or resume

Use physical GPU 1 only. After `CUDA_VISIBLE_DEVICES=1`, converter `-d 0`
correctly means logical device 0 = physical GPU 1.

```bash
cd "$EXL3"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
chmod +x scripts/convert_behemoth_exl3_4p5.sh
set -o pipefail
./scripts/convert_behemoth_exl3_4p5.sh 2>&1 | tee results/convert_behemoth_exl3_4p5.log
```

Run that block inside `tmux` if the SSH session may disconnect.

The new job uses:

- target average `4.5` bpw;
- H6 `lm_head`;
- explicit `mul1` codebook for the Ampere K≤5 INT8 GEMV path;
- 250 rows × 2048 tokens of ExLlamaV3's default calibration corpus (ArtusDev
  used 100×2048; the pinned bundled wiki splitter overflows at 512 rows);
- 8192 MiB output shards;
- pinned ExLlamaV3 commit `0c49587a7c235e6303a6bbedc8b665272ad3a2ea`.

If SSH disconnects or the process stops, export the same paths and invoke the
same script. It detects `$WORK_DIR/args.json` or
`$WORK_DIR/ckpt/job.json` and runs the official resume form:

```bash
python "$EXLLAMAV3_SRC/convert.py" -w "$WORK_DIR" -r
```

Do not change bitrate, codebook, calibration dimensions, input, or output
while resuming. Do not delete `$WORK_DIR` until inventory, TP4 load, and KLD
are complete.

### 6.3 Validate and score

The script writes `results/behemoth_exl3_4p5_inventory.json` and prints exact
bytes when conversion completes. Verify explicitly:

```bash
python "$EXL3/scripts/validate_exl3_checkpoint.py" \
  "$OUT_DIR" --allow-non-behemoth \
  --sha-manifest "$EXL3/results/behemoth_exl3_4p5_inventory.json"
du -sh "$OUT_DIR"
du -sb "$OUT_DIR"
```

Then run the same patched eager KLD harness used for 3.5 and 4.25 bpw. Require
204,700 positions and record both mean KLD and positions/s. Do not infer
4.5-bpw quality from bitrate alone.

```bash
export KLD_VLLM=/home/phaedawg/kld-exl3-vllm/kldexl3-vllm
export MODEL_DIR="$OUT_DIR"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2,3,4
export VLLM_PLUGINS=vllm_exl3_sm86
export VLLM_EXL3_EXT_PATH="$EXL3/build/exllamav3_ext"
export VLLM_EXL3_SKIP_VERSION_GUARD=1
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd "$KLD_VLLM"
python examples/offline_inference/score_mode_kld.py \
  --model "$MODEL_DIR/" \
  --reference-logits /media/netmodels/ref_logits/ref_logits_Behemoth-R1-123B-v2_ctx2048_s512/ \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.92 \
  --quantization exl3
```

---

## 7. Behemoth quality gate (eager TP4 on 1–4)

```bash
source "$VENV_VLLM/bin/activate"
cd "$EXL3"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2,3,4
export VLLM_PLUGINS=vllm_exl3_sm86
export VLLM_EXL3_EXT_PATH="$EXL3/build/exllamav3_ext"
export MODEL_DIR=/media/fmodels/ArtusDev/TheDrummer_Behemoth-R1-123B-v2-EXL3/3.5bpw_H6

python scripts/quality_gate_behemoth.py --model "$MODEL_DIR" --tp 4 --max-model-len 32768
python scripts/compare_native_logits.py --model "$MODEL_DIR" --tp 4 --native-device 0
```

`--native-device 0` is logical 0 = physical 1. Native ExLlamaV3 comparison
needs the **conversion** env’s Python for the full package, or skip native
with `--skip-native` and compare greedy only.

KLD (copy `ref_logits_Behemoth-R1-123B-v2_ctx2048_s512` onto this host if it
lives on another box; score **here** with the SM86 extension):

```bash
# set models[0].localPath to $MODEL_DIR in kld/Models_KLD_Behemoth_EXL3.json
./scripts/run_kld_exl3.sh
```

Gates:

- 3.5-bpw memory-first receipt: **0.045794** reproducibly
- production candidate: mean KLD < **0.034004**
- current EXL3 leader: 4.25-bpw at **0.015800**
- TP4 BF16 KV pool capacity greater than 32K tokens
- do not compare against GLM KLD numbers

Serve (eager until graphs are proven):

```bash
source "$VENV_VLLM/bin/activate"
export CUDA_VISIBLE_DEVICES=1,2,3,4
export VLLM_PLUGINS=vllm_exl3_sm86
export VLLM_EXL3_EXT_PATH="$EXL3/build/exllamav3_ext"
export MODEL_DIR=/media/fmodels/ArtusDev/TheDrummer_Behemoth-R1-123B-v2-EXL3/3.5bpw_H6
./scripts/serve_exl3_sm86.sh "$VLLM_API_KEY"
```

---

## 8. Prefill crossover, decode graphs, release (after eager is green)

```bash
source "$VENV_VLLM/bin/activate"
cd "$EXL3"
export CUDA_VISIBLE_DEVICES=1
python scripts/prewarm_kernels.py --device 0
python scripts/graph_replay_stress.py --device 0
```

Only then:

```bash
export CUDA_VISIBLE_DEVICES=1,2,3,4
export VLLM_EXL3_ALLOW_GRAPHS=1
ENFORCE_EAGER=0 ./scripts/serve_exl3_sm86.sh "$VLLM_API_KEY"
```

Release checks:

```bash
export MODEL_DIR=/media/fmodels/ArtusDev/TheDrummer_Behemoth-R1-123B-v2-EXL3/3.5bpw_H6
MODEL_DIR="$MODEL_DIR" ./scripts/restart_test.sh
```

### Remote serving benchmark: 1K through 32K

Serve one checkpoint at a time with the same TP4, eager/graph, maximum model
length, batching, and KV-cache settings. For the first comparison, use eager
for all three backends to isolate quantization/backend behavior. The server
must permit at least 32,768 prompt + 256 output tokens.

Run the client on the same machine or same LAN path for every model because
TTFT includes network and scheduler delay:

```bash
export VLLM_API_KEY='replace-me'

python scripts/bench_serving_contexts.py \
  --host 192.0.2.10 \
  --api-key "$VLLM_API_KEY" \
  --model Behemoth-R1-123B-v2-EXL3-4.25-H6 \
  --label artus-exl3-4p25 \
  --contexts 1024,2048,4096,8192,16384,32768 \
  --output-tokens 256 \
  --warmup-runs 1 \
  --runs 3 \
  --output results/serving_artus_exl3_4p25.json
```

Repeat without changing client/server topology:

```bash
# 69.5G mixed Intel AutoRound / compressed-tensors server
python scripts/bench_serving_contexts.py \
  --host 192.0.2.10 --api-key "$VLLM_API_KEY" \
  --model Behemoth-R1-123B-v2-AutoRound-69p5G \
  --label autoround-69p5g \
  --output results/serving_autoround_69p5g.json

# Local 4.5-bpw EXL3 server
python scripts/bench_serving_contexts.py \
  --host 192.0.2.10 --api-key "$VLLM_API_KEY" \
  --model Behemoth-R1-123B-v2-EXL3-4.5-H6 \
  --label local-exl3-4p5 \
  --output results/serving_local_exl3_4p5.json
```

The default contexts are already 1K/2K/4K/8K/16K/32K, with one warmup and
three measured runs each. Every request:

- sends an exact token-ID prompt produced by the server's `/tokenize` endpoint;
- changes the first token block to prevent prefix-cache reuse;
- requests exactly 256 tokens with temperature 0 and `ignore_eos`;
- streams `/v1/completions` to measure TTFT, decode tok/s, TPOT, and
  SSE inter-chunk p50/p95/p99;
- reports effective prefill tok/s as prompt tokens divided by TTFT;
- writes both JSON and CSV after every measured run, preserving partial work.

`effective_prefill_tok_s` is request-level throughput and includes network,
queueing, and first-token decode. It is appropriate for an apples-to-apples
serving comparison but is not raw kernel throughput. Confirm
`stream_chunks_match_tokens=true` before treating chunk ITL as token ITL.

Targets: ≥90% of Marlin prefill/decode on the production 8K batch; p99 TPOT
≤15% worse. If optimized M=1 stays below 60% of Marlin, keep EXL3 as
experimental memory-first and leave the 72G Marlin checkpoint in production.

---

## Checklist (in order)

1. Confirm physical 0 and 5 stay busy; we only touch 1–4.
2. KLD environment has Torch `2.13.0+cu132`, vLLM
   `0.1.dev20517+gb99dae944`, and the score-mode prompt passthrough patch.
3. `capture_manifest.sh` → `manifests/stack.captured.json`.
4. Build `exllamav3_ext` with `TORCH_CUDA_ARCH_LIST=8.6`; `pip install -e plugin`.
5. `pip install pytest` in `$VENV_VLLM`; `python -m pytest tests -q` (not system `pytest`).
6. `kernel_microbench.py --fail-on-gate` on physical GPU 1. **Stop if it fails.**
7. ArtusDev 3.5 and 4.25-bpw TP4 eager load and KLD receipts captured.
8. Convert 4.5-bpw H6 mul1 from BF16 on physical GPU 1; retain work state.
9. Validate inventory and exact bytes; TP4 eager load on physical 1–4.
10. Score 4.5-bpw over all 204,700 positions with the same reference logits.
11. Benchmark 4.25 and 4.5 EXL3 against Marlin; then evaluate graphs and restart.

---

## Do not

- Use physical GPU 0 or 5.
- Requantize an existing EXL3, Marlin, or AutoRound checkpoint.
- Convert from the 72G Marlin / AutoRound checkpoint.
- Install ExLlamaV3 or this plugin into `~/llmcompressor-nightly`.
- Enable CUDA graphs (`ENFORCE_EAGER=0`) before eager TP4 + KLD + prewarm.
- Score this SM86 plugin on a Blackwell box with an 8.6-only `.so`.
- Skip the kernel microbench to “save time” before loading 123B.
