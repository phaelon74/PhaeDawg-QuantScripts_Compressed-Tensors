# Behemoth mixed W4A16 / W8A16 PTQ

Quantize [TheDrummer/Behemoth-R1-123B-v2](https://huggingface.co/TheDrummer/Behemoth-R1-123B-v2) for **4x RTX 3090** (SM86) serving. Use packed INT4 W4A16 plus a small W8A16 promotion budget. Do not use FP8_BLOCK or NVFP4 for the Ampere-bound checkpoint.

KLD scoring stays in the vLLM venv. This harness runs in the **llm-compressor** venv. See [BASELINE.md](BASELINE.md) for pinned versions.

## Size rule

Hard cap **70 GiB**. Aim **65–68 GiB**. Group-size 32 already uses ~66 GiB with uniform W4; GS128 is the donor format when promoting layers to W8.

Preflight (config.json only, no weight load):

```bash
cd MixedPrecision
python estimate_packed_size.py /path/to/Behemoth-R1-123B-v2 --group-size 32
python estimate_packed_size.py /path/to/Behemoth-R1-123B-v2 --group-size 128 \
  --promote-down-proj-layers 0,1,87 --max-disk-gib 70
```

Exit code 2 means over budget.

## Quantize

```bash
# Dry run (size only)
python behemoth_mixed_ptq.py SRC DST recipes/baseline_512.yaml --dry-run

# Candidate 2 was GPTQ GS32: KLD 0.046951, rejected.

# Candidate 3: AutoRound W4A16 GS32, same 512 samples as baseline.
python behemoth_mixed_ptq.py SRC DST recipes/baseline_512.yaml \
  --algorithm autoround --group-size 32 \
  --autoround-iters 200 --autoround-batch-size 1 \
  --autoround-gradient-accumulate-steps 8 --autoround-low-gpu-mem \
  --autoround-device-ids 0,1,2,3 --skip-sample-gen

# Candidate 4: canonical asymmetric AWQ GS32, same calibration.
python behemoth_mixed_ptq.py SRC DST recipes/baseline_512.yaml \
  --algorithm awq --group-size 32 --asymmetric --skip-sample-gen

# Candidate 4 is cancelled. Build the uniform GS128 AutoRound donor instead.
python behemoth_mixed_ptq.py SRC DST_GS128 recipes/baseline_512.yaml \
  --policy-yaml recipes/autoround_gs128_uniform.yaml \
  --autoround-iters 200 --autoround-batch-size 1 \
  --autoround-gradient-accumulate-steps 8 --autoround-low-gpu-mem \
  --autoround-device-ids 0,1,2,3 --skip-sample-gen
```

`--max-disk-gib 70` is enforced before `from_pretrained`.

AutoRound uses the official llm-compressor `AutoRoundModifier` and saves
compressed-tensors directly. AutoRound **0.13.0 or newer** is required for
W4A16 compressed-tensors export. Verify it before the long run:

```bash
python -c "import auto_round; print(auto_round.__version__)"
```

The harness pads AutoRound calibration samples to exactly 2048 tokens and runs
`fix_batch_if_needed`. AutoRound block reconstruction cannot concatenate
variable sequence lengths; errors such as `Expected size 2041 but got 2032`
indicate an older copy of the harness.

AutoRound tunes the inferred full decoder layer. The low-memory compatibility
path keeps the 512x2048 block-input, reference-output, and propagated quantized
input caches on CPU and streams one sample at a time. Each hidden-state cache is
24 GiB for this model; eagerly retaining all three on GPU leaves too little
space for SignSGD backward. Gradient accumulation of 8 restores AutoRound's
effective tuning batch while preserving batch-size-1 peak VRAM.

Allow roughly 72 GiB of host RAM for those three activation caches in addition
to model/offload memory. Disabling `--autoround-low-gpu-mem` is not viable for
the 512x2048 recipe on 96 GiB GPUs.

## Mixed W4/W8 search

After the uniform GS128 donor is running, rank every `down_proj` without loading
the 123B model. The scorer streams safetensor row chunks through one GPU,
compares symmetric GS128 W4 and W8 reconstruction error, and writes nested
policies with a 0.25 GiB safety margin:

```bash
python rank_weight_sensitivity.py SRC \
  --group-size 128 --device cuda:0 \
  --budgets 66,68,69.5 \
  --score-json results/sensitivity_gs128.json \
  --policy-dir recipes/generated
```

`down_proj` is the production-safe default. `--promotion-kinds all` enables
broader experimentation while keeping Q/K/V and gate/up fused units together.

Preflight every generated policy before loading weights:

```bash
for POLICY in recipes/generated/autoround_gs128_mixed_*.yaml; do
  python behemoth_mixed_ptq.py SRC /tmp/not-used recipes/baseline_512.yaml \
    --policy-yaml "$POLICY" --dry-run
done
```

Quantize each policy with the same AutoRound settings as Candidate3:

```bash
python behemoth_mixed_ptq.py SRC DST_MIXED_66 recipes/baseline_512.yaml \
  --policy-yaml recipes/generated/autoround_gs128_mixed_66g.yaml \
  --autoround-iters 200 --autoround-batch-size 1 \
  --autoround-gradient-accumulate-steps 8 --autoround-low-gpu-mem \
  --autoround-device-ids 0,1,2,3 --skip-sample-gen
```

Repeat for `68g` and `69p5g`. Treat the streaming score as a ranking proxy;
the frozen 204700-position KLD result remains the selection authority.

## KLD

Score against `ref_logits_Behemoth-R1-123B-v2_ctx2048_s512` in the vLLM
environment. AutoRound GS32 is the current winner at **0.037094**, versus
0.042380 for the original baseline, 0.042729 for AWQMSK, and 0.046951 for
GPTQ. Use AutoRound for the GS128 donor and all mixed W4/W8 candidates.
