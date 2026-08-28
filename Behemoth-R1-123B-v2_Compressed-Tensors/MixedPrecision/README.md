# Behemoth mixed W4A16 / W8A16 PTQ

Quantize [TheDrummer/Behemoth-R1-123B-v2](https://huggingface.co/TheDrummer/Behemoth-R1-123B-v2) for **4x RTX 3090** (SM86) serving. Use packed INT4 W4A16 plus a small W8A16 promotion budget. Do not use FP8_BLOCK or NVFP4 for the Ampere-bound checkpoint.

KLD scoring stays in the vLLM venv. This harness runs in the **llm-compressor** venv. See [BASELINE.md](BASELINE.md) for pinned versions.

## Size rule

AutoRound GS32 is the fixed donor configuration. The 72 GiB mixed policy
is the frozen accuracy winner at mean KLD 0.034004. Further W4/W8 mixed
expansion is paused in favor of the EXL3 SM86 workstream.

Preflight (config.json only, no weight load):

```bash
cd MixedPrecision
python estimate_packed_size.py /path/to/Behemoth-R1-123B-v2 --group-size 32
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
  --autoround-device-ids 0 --skip-sample-gen

# GS128, AWQ, and pure GPTQ branches are rejected. Continue with GS32 mixed
# W4/W8 policies below.
```

`--max-disk-gib` is enforced before `from_pretrained`; pass the tier ceiling
explicitly for candidates above its conservative 70 GiB default.

Both GPU-using scripts set `CUDA_VISIBLE_DEVICES=0` before importing PyTorch.
AutoRound also defaults to device ID `0`, so the source model remains
CPU-backed and only physical GPU 0 receives streamed block work. GPUs 1–3 stay
available for other processes.

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

Rank GS32 W4-to-W8 promotion benefit from the original BF16 weights. The scorer
streams safetensor row chunks through one GPU, groups vLLM-fused projections
together, and writes nested GS32 policies with a 0.25 GiB safety margin:

```bash
python rank_weight_sensitivity.py SRC \
  --group-size 32 \
  --promotion-kinds all \
  --budgets 69.5,72,74,76 \
  --score-json results/sensitivity_gs32.json \
  --policy-dir recipes/generated \
  --reuse-scores
```

Including 69.5 as the first budget locks the validated policy before selecting
additional units. Reuse mode does not touch the weights or GPU. Each YAML records
its newly added units and incremental proxy utility.

This weight-only score is a screening proxy; the frozen KLD suite remains the
selection authority.

Preflight every generated policy before loading weights:

```bash
for POLICY in recipes/generated/autoround_gs32_mixed_*.yaml; do
  python behemoth_mixed_ptq.py SRC /tmp/not-used recipes/baseline_512.yaml \
    --policy-yaml "$POLICY" --dry-run
done
```

Quantize each expanded policy from the BF16 source with the same AutoRound
settings as Candidate3:

```bash
python behemoth_mixed_ptq.py SRC DST_MIXED_72 recipes/baseline_512.yaml \
  --policy-yaml recipes/generated/autoround_gs32_mixed_72g.yaml \
  --max-disk-gib 72 \
  --autoround-iters 200 --autoround-batch-size 1 \
  --autoround-gradient-accumulate-steps 8 --autoround-low-gpu-mem \
  --autoround-device-ids 0 --skip-sample-gen
```

Repeat for `74g` and `76g`, changing both the policy and `--max-disk-gib`.
Evaluate 72G first; continue only while measured KLD improvement per added GiB
justifies the next full run.

## KLD

Score against `ref_logits_Behemoth-R1-123B-v2_ctx2048_s512` in the vLLM
environment. Mixed AutoRound GS32 at 69.5 GiB is the current winner at
**0.035669**, improving on uniform AutoRound GS32 at 0.037094. The original
baseline scored 0.042380; AWQMSK, GPTQ, and GS128 mixed are rejected.
