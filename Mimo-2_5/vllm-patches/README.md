# MiMo-V2.5 + sm_120 vLLM patches

These four files unblock running **XiaomiMiMo/MiMo-V2.5** (and any other
sink-attention + decoupled-QKV model) on **Blackwell consumer GPUs
(sm_120)** in vLLM, where FlashAttention 3 is hardware-impossible.

The patch path is: replace the FA3-based `FlashAttentionDiffKVBackend`
with a Triton-based `TritonDiffKVSinksBackend` whenever vLLM detects a
device with compute capability >= 12.

## Files

| File | Purpose |
|---|---|
| `triton_unified_attention_diffkv.py` | Triton kernel that adds DiffKV (`head_size_q != head_size_v`) to vLLM's existing `unified_attention` (which already supports sinks). |
| `triton_diffkv_sinks.py` | vLLM backend wrapper. Subclasses `FlashAttentionDiffKVBackend`, overrides only `forward()` to call our kernel. |
| `mimo_v2_patch.diff` | 8-line diff against `vllm/model_executor/models/mimo_v2.py` that selects the new backend on sm_120+. |
| `test_triton_diffkv_sinks.py` | Pytest comparing the kernel against a pure-PyTorch reference. **Run this first** — never load the 310B model with an unverified kernel. |

## Apply order

Run from the workstation that has the vLLM source checkout (e.g.
`~/nightly-kld-vllm/kld-vllm`):

```bash
VLLM_ROOT=~/nightly-kld-vllm/kld-vllm
PATCHES=~/QuantScripts/Mimo-2_5/vllm-patches    # adjust to your local path

# 1. Drop in the kernel + Python wrapper
cp "$PATCHES/triton_unified_attention_diffkv.py" \
   "$VLLM_ROOT/vllm/v1/attention/ops/triton_unified_attention_diffkv.py"

# 2. Drop in the backend
cp "$PATCHES/triton_diffkv_sinks.py" \
   "$VLLM_ROOT/vllm/v1/attention/backends/triton_diffkv_sinks.py"

# 3. Drop in the test
mkdir -p "$VLLM_ROOT/tests/kernels/attention"
cp "$PATCHES/test_triton_diffkv_sinks.py" \
   "$VLLM_ROOT/tests/kernels/attention/test_triton_diffkv_sinks.py"

# 4. Apply the MiMo backend-selection patch
cd "$VLLM_ROOT"
patch -p1 < "$PATCHES/mimo_v2_patch.diff"
```

If `current_platform` is not already imported in `mimo_v2.py`, add it
near the other vLLM imports:

```python
from vllm.platforms import current_platform
```

(Quick check: `rg "from vllm.platforms" vllm/model_executor/models/mimo_v2.py`)

## Validate the kernel BEFORE loading MiMo-V2.5

The kernel is the entire risk surface of this change. Validate it on
random tensors first — takes ~10 seconds, no model needed:

```bash
cd "$VLLM_ROOT"
pytest -xvs tests/kernels/attention/test_triton_diffkv_sinks.py
```

Expected: 90 test cases (5 seq_q × 3 seq_k × 3 hd combos × 2 sinks ×
2 sliding) all pass with `max_abs < 5e-2`. The seq_q > seq_k cases
auto-skip.

If any case fails, **stop and paste the failure output** — do NOT try
to load the 310B model with a buggy kernel.

## End-to-end smoke test

Once the kernel test passes, start small:

```bash
# A. Load MiMo-V2.5-W4A16 with --max-model-len 1024 first.
cd "$VLLM_ROOT"
python3 examples/offline_inference/score_mode_kld.py \
    --model /media/fmodels/TheHouseOfTheDude/Mimo2-5_PTQ_INT4/ \
    --reference-model /media/fmodels/XiaomiMiMo/MiMo-V2.5/ \
    --dataset wikitext --dataset-config wikitext-2-raw-v1 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 1024 \
    --trust-remote-code 2>&1 | tee ~/Documents/kld_diffkv_run1.log
```

Look for these markers in the log:

| Expected log line | Meaning |
|---|---|
| `Using TritonDiffKVSinksBackend for attention (sm_120+, no FA3).` | Our backend is selected (good!) |
| `Using FlashAttentionDiffKVBackend for attention.` | Wrong — patch didn't take effect |
| `AssertionError: Sinks are only supported in FlashAttention 3` | Wrong — old FA3 path still active |
| Generation completes, KLD numbers print | All-good, scale up to longer max_model_len |

If you hit any unexpected error, save the full log and let me know
which marker (or which Python traceback) you saw.

## Limitations of v1

The first iteration is intentionally minimal:

- **2D launch path only.** The 3D segmented-decode path (used for very
  small batches in pure-decode) is not yet implemented. Throughput at
  batch=1 will be lower than peak. For TP4 with realistic batches this
  doesn't matter.
- **BF16 / FP16 KV cache only.** No FP8 KV cache. (The W4A16 model
  weights are fine — that's a separate axis.)
- **No DCP** (decode context parallel).
- **No cascade attention.** Cross-sequence prefix sharing falls back
  to non-cascade. vLLM's scheduler will still work; the optimization
  for shared prefixes won't trigger.
- **No chunked attention** (Gemma3-style block-local). Not used by
  MiMo.

Each of these is a small follow-up PR once the baseline is validated.

## What was confirmed during diagnosis

For posterity (and for the eventual upstream PR description):

1. The `_vllm_fa3_C.abi3.so` shipped by `vllm-flash-attn` for sm_120
   is **sm_75 PTX only** — vNCC silently dropped the FA3 Hopper
   kernels because they require sm_90+ instructions (`wgmma.async`,
   TMA, TMEM). The 856 MB binary contains no usable FA3 code for
   sm_120.
2. sm_120 (Blackwell consumer) hardware physically lacks `wgmma`,
   TMA, and TMEM. Confirmed by Dao-AILab PR #2329 and the voipmonitor
   RTX 6000 Pro reference notes.
3. Therefore there is no rebuild, patch, or flag that makes
   `FlashAttentionDiffKVBackend` work on sm_120. A non-FA3 backend
   is the only option.
4. vLLM's existing `unified_attention` Triton kernel already supports
   sinks (used by gpt-oss on sm_120). The only missing capability
   for MiMo is `head_size_q != head_size_v`. This patch adds that.
