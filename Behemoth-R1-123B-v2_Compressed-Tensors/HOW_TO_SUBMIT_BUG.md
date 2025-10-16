# How to Submit the NVFP4 Bug Report

## Quick Summary
You've discovered a critical bug in llm-compressor's NVFP4 implementation that prevents quantization of large models (100B+) on consumer GPUs.

## What to Submit

### 1. GitHub Issue
**Repository**: https://github.com/vllm-project/llm-compressor/issues

**Title**: 
```
NVFP4 OOM during initialization on large models (100B+) - update_fused_layer_weight_global_scales loads all layers
```

**Labels to Request**:
- `bug`
- `quantization`
- `nvfp4`
- `memory`

### 2. Files to Attach/Reference

I've created these files for you in this directory:

1. **`BUG_REPORT.md`** - Comprehensive bug description (paste into issue body)
2. **`minimal_repro_nvfp4_bug.py`** - Minimal script to reproduce the bug
3. **`gather_system_info.sh`** - Script to collect system information

### 3. Steps to Submit

#### Step 1: Gather System Information
```bash
cd Behemoth-R1-123B-v2_Compressed-Tensors
bash gather_system_info.sh > system_info.txt
```

Or manually run:
```bash
# Python & PyTorch versions
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

# Package versions
pip list | grep -E "llmcompressor|torch|transformers|compressed-tensors|accelerate"

# GPU info
nvidia-smi

# System info  
uname -a
free -h
```

#### Step 2: Create GitHub Issue

1. Go to: https://github.com/vllm-project/llm-compressor/issues/new
2. Copy the entire contents of `BUG_REPORT.md`
3. Paste into the issue description
4. Add your system information (from Step 1) to the bottom
5. Attach `minimal_repro_nvfp4_bug.py` as a file or gist

#### Step 3: Optional - Test NVFP4A16
Before submitting, you could also test if NVFP4A16 (weights-only) has the same issue:

```bash
cd Behemoth-R1-123B-v2_Compressed-Tensors
python llama_NVFP4A16.py 2>&1 | tee nvfp4a16_test.log
```

Add the results to your issue (whether it worked or failed).

## Key Points to Emphasize in Issue

1. **This is a blocker** for quantizing large models (100B+) with NVFP4 on consumer hardware
2. **FP8_DYNAMIC works** on the same hardware/model, proving sequential onloading is possible
3. **Failure happens during initialization**, not calibration, so `sequential_targets` doesn't help
4. **Root cause identified**: `update_fused_layer_weight_global_scales()` loads all MLP layers at once
5. **Proposed solutions** are included in the bug report

## What to Expect

The llm-compressor team will likely:
1. Ask for your system info (you'll have it ready!)
2. May ask you to test a patch
3. May ask for more details about the model
4. Will likely fix this by making the initialization sequential

## Alternative: Discussion First

If you'd prefer to start with a discussion rather than immediately filing a bug:

**Repository**: https://github.com/vllm-project/llm-compressor/discussions

**Category**: Q&A or Help

**Title**:
```
NVFP4 quantization fails on large models (100B+) - memory issue during initialization
```

This allows for back-and-forth before opening a formal issue.

## Related Issues to Reference

Search for these in the llm-compressor repo to see if anyone else has reported similar:
- "NVFP4 OOM"
- "update_fused_layer_weight_global_scales"
- "large model quantization"
- "sequential onloading NVFP4"

If you find related issues, reference them in your report.

## Contact Information

When submitting, include:
- Your GitHub username
- How you'd like to be contacted for testing fixes
- Whether you're willing to test patches

## After Submission

Monitor the issue for:
1. Questions from maintainers
2. Requests for additional information
3. Proposed patches to test
4. Workarounds they might suggest

## Quick Submission Template

If you want to file quickly, here's a minimal template:

```markdown
## Problem
NVFP4 quantization fails with CUDA OOM during initialization on 123B Mistral-based model on 24GB GPU.

## Error Location
`llmcompressor/modifiers/utils/helpers.py:85` - `update_fused_layer_weight_global_scales()`

## Root Cause
Function loads ALL MLP layers (gate_proj + up_proj) onto GPU simultaneously instead of sequentially.

## Reproduction
[Attach minimal_repro_nvfp4_bug.py]

## System Info
- GPU: RTX 4090 / A5000 (24GB)
- Model: 123B parameters
- llmcompressor: [version]
- PyTorch: [version]
- CUDA: [version]

## Expected
Sequential processing keeps VRAM under 24GB (like FP8_DYNAMIC does)

## Actual
OOM trying to allocate 128GB+ VRAM for all layers

## Impact
Blocks NVFP4 quantization of any 100B+ model on consumer GPUs
```

---

**Good luck with the submission! This is a valuable bug report that will help the entire community.**

