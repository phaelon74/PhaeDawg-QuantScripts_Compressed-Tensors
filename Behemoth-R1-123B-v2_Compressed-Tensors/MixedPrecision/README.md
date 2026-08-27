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
python behemoth_mixed_ptq.py SRC DST ../../Recipes/Datasets/General_reasoning.yaml --dry-run

# Candidate 2: uniform GPTQ W4A16 GS32
python behemoth_mixed_ptq.py SRC DST ../../Recipes/Datasets/General_reasoning.yaml \
  --algorithm gptq --group-size 32

# Candidate 3: AWQ then GPTQ, GS32
python behemoth_mixed_ptq.py SRC DST ../../Recipes/Datasets/General_reasoning.yaml \
  --algorithm awq_gptq --group-size 32 --use-loss-mask

# Mixed: GS128 + a few W8 down_proj layers
python behemoth_mixed_ptq.py SRC DST ../../Recipes/Datasets/General_reasoning.yaml \
  --algorithm awq_gptq --group-size 128 \
  --promote-down-proj-layers 0,1,87 \
  --policy-yaml recipes/policy_example.yaml
```

`--max-disk-gib 70` is enforced before `from_pretrained`.

## KLD

Score against `ref_logits_Behemoth-R1-123B-v2_ctx2048_s512` in the vLLM env. The GS32 baseline to beat is **mean KLD 0.042380**. Finish `W4A16_GS32_AWQMSK` before ranking new runs.
