# NVFP4 OOM Bug Report for llm-compressor

## Issue Summary
NVFP4 (W4A4) quantization fails with CUDA OOM error during initialization phase for large models (123B parameters) on 24GB GPUs, even with `device_map=None` and `sequential_targets` configured. The failure occurs **before calibration starts**, during `update_fused_layer_weight_global_scales()`.

## Environment

### Hardware
- **GPU**: RTX 4090 / A5000 (24GB VRAM)
- **System RAM**: 400GB+
- **GPUs Available**: Multiple (restricted to single GPU via `CUDA_VISIBLE_DEVICES=0`)

### Software
```bash
# Please run these commands and paste output:
pip list | grep -E "llmcompressor|torch|transformers|compressed-tensors|accelerate"

# Example expected output:
# llmcompressor==0.X.X
# torch==2.X.X
# transformers==4.X.X
# compressed-tensors==0.X.X
# accelerate==0.X.X
```

### Model
- **Model**: Mistral-Large-Instruct-2411 based (Behemoth-R1-123B-v2)
- **Architecture**: Mistral (123B parameters)
- **Size**: ~245GB (BF16)
- **Source**: Local directory (HuggingFace format)

## Problem Description

### What Works ✅
- **FP8_DYNAMIC** quantization works perfectly with sequential onloading on same hardware
- **NVFP4** on smaller models (< 20B parameters) works fine
- **Llama4-17B** NVFP4 example works (from llm-compressor examples)

### What Fails ❌
- **NVFP4 (W4A4)** with calibration data on 123B model
- **NVFP4A16 (W4A16)** likely has same issue (needs testing)

### When It Fails
The failure occurs during initialization, specifically:
1. After model loads to CPU (`device_map=None`)
2. After dataset preparation completes
3. When `oneshot()` begins processing
4. During `CALIBRATION_EPOCH_START` event
5. In `update_fused_layer_weight_global_scales()` function

**It fails BEFORE any calibration samples are processed.**

## Error Traceback

```python
File "/home/phaedawg/llmcompressor/venv/lib/python3.12/site-packages/llmcompressor/pipelines/sequential/pipeline.py", line 74, in __call__
    LifecycleCallbacks.calibration_epoch_start()

File "/home/phaedawg/llmcompressor/venv/lib/python3.12/site-packages/llmcompressor/modifiers/quantization/quantization/base.py", line 80, in on_start
    update_fused_layer_weight_global_scales(module)

File "/home/phaedawg/llmcompressor/venv/lib/python3.12/site-packages/llmcompressor/modifiers/utils/helpers.py", line 85, in update_fused_layer_weight_global_scales
    with align_modules([submodule.gate_proj, submodule.up_proj]):

File "/home/phaedawg/llmcompressor/venv/lib/python3.12/site-packages/compressed_tensors/utils/offload.py", line 437, in align_modules
    stack.enter_context(align_module_device(module, execution_device))

File "/home/phaedawg/llmcompressor/venv/lib/python3.12/site-packages/compressed_tensors/utils/offload.py", line 663, in align_module_device
    module._hf_hook.pre_forward(module)

File "/home/phaedawg/llmcompressor/venv/lib/python3.12/site-packages/accelerate/hooks.py", line 360, in pre_forward
    set_module_tensor_to_device(

File "/home/phaedawg/llmcompressor/venv/lib/python3.12/site-packages/accelerate/utils/modeling.py", line 343, in set_module_tensor_to_device
    new_value = value.to(device, non_blocking=non_blocking)

torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 672.00 MiB. 
GPU 0 has a total capacity of 23.57 GiB of which 585.00 MiB is free. 
Including non-PyTorch memory, this process has 22.99 GiB memory in use. 
Of the allocated memory 22.45 GiB is allocated by PyTorch, and 260.00 MiB is reserved by PyTorch but unallocated.
```

## Root Cause Analysis

### The Problem
`update_fused_layer_weight_global_scales()` in `llmcompressor/modifiers/utils/helpers.py:85` calls:

```python
with align_modules([submodule.gate_proj, submodule.up_proj]):
```

This attempts to align **ALL** MLP layers' `gate_proj` and `up_proj` modules onto GPU simultaneously to compute global scales for fused layers.

For a 123B model with ~80 layers, this tries to load:
- 80 layers × 2 modules (gate_proj + up_proj) × ~800MB each = **~128GB VRAM** 
- Available VRAM: **24GB**
- Result: **OOM**

### Why FP8_DYNAMIC Works
FP8_DYNAMIC doesn't require this fused layer initialization step. It processes layers one-by-one from the start.

### Why Llama4-17B Works  
Llama4-17B is much smaller:
- Fewer layers (~32 vs 80)
- Smaller hidden dimensions
- Total fused layers fit in 24GB VRAM

### Why sequential_targets Doesn't Help
`sequential_targets=["MistralMLP"]` only affects the **calibration phase**, but the failure occurs during the **initialization phase** (before calibration).

## Minimal Reproducible Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Model configuration
MODEL_ID = "mistralai/Mistral-Large-Instruct-2411"  # or any 100B+ Mistral-based model
NUM_CALIBRATION_SAMPLES = 20
MAX_SEQUENCE_LENGTH = 2048

# Load model to CPU
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map=None,  # Load to CPU
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Prepare minimal calibration data
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]")

def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

ds = ds.map(preprocess)
ds = ds.map(
    lambda batch: tokenizer(batch["text"], padding=False, max_length=MAX_SEQUENCE_LENGTH, truncation=True, add_special_tokens=False),
    batched=True,
    remove_columns=ds.column_names,
)

# Configure NVFP4 quantization
recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])

# This will fail during initialization with OOM
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    sequential_targets=["MistralMLP"],  # Doesn't help - OOM happens before this takes effect
)
```

## Expected Behavior
`update_fused_layer_weight_global_scales()` should respect sequential processing and:
1. Load one MLP layer at a time onto GPU
2. Compute global scales for that layer
3. Offload before processing next layer

OR

Provide a configuration option to skip fused layer global scale computation for memory-constrained environments.

## Actual Behavior
All MLP layers are loaded onto GPU simultaneously during initialization, causing OOM on GPUs < 80GB for large models.

## Proposed Solutions

### Solution 1: Sequential Global Scale Computation
Modify `update_fused_layer_weight_global_scales()` to process layers sequentially:

```python
# Current (loads all layers at once):
for name, submodule in model.named_modules():
    if has_fused_mlp:
        with align_modules([submodule.gate_proj, submodule.up_proj]):
            # compute global scale

# Proposed (process one layer at a time):
for name, submodule in model.named_modules():
    if has_fused_mlp:
        # Load gate_proj
        with align_module_device(submodule.gate_proj, execution_device):
            # compute partial scale
        # Offload gate_proj
        
        # Load up_proj
        with align_module_device(submodule.up_proj, execution_device):
            # compute partial scale
        # Offload up_proj
        
        # Combine scales
```

### Solution 2: Lazy Global Scale Initialization
Defer global scale computation until calibration phase when sequential processing is active.

### Solution 3: Optional Fused Layer Handling
Add a parameter to disable fused layer global scale optimization for memory-constrained scenarios:

```python
recipe = QuantizationModifier(
    targets="Linear", 
    scheme="NVFP4", 
    ignore=["lm_head"],
    skip_fused_layer_init=True  # New parameter
)
```

## Questions for llm-compressor Team

1. Is `update_fused_layer_weight_global_scales()` required for NVFP4, or can it be skipped?
2. Can this initialization step be deferred to the calibration phase where sequential processing is active?
3. Is there a workaround to quantize 100B+ models with NVFP4 on 24GB GPUs?
4. Should NVFP4A16 (weights-only) bypass this initialization step?

## Additional Context

### What We've Tried
- ✅ `device_map=None` (loads model to CPU)
- ✅ `sequential_targets=["MistralMLP"]` (doesn't help - OOM before calibration)
- ✅ `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (doesn't help - too many layers)
- ✅ `torch.cuda.empty_cache()` before starting (doesn't help)
- ✅ Single GPU restriction (doesn't help)
- ❌ Reducing calibration samples (doesn't help - OOM before calibration starts)

### Files to Investigate
1. `llmcompressor/modifiers/utils/helpers.py` - Line 85 (`update_fused_layer_weight_global_scales`)
2. `llmcompressor/modifiers/quantization/quantization/base.py` - Line 80 (calls the helper)
3. `compressed_tensors/utils/offload.py` - Line 437 & 663 (alignment logic)

### Comparison with Working Code
FP8_DYNAMIC quantization on the same hardware/model works perfectly:

```python
recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])

model = oneshot(
    model=model_path_string,  # Can even pass path directly
    recipe=recipe,
    trust_remote_code_model=True,
)
# This works perfectly - no OOM!
```

## System Information Template

Please fill this in when submitting:

```bash
# Python version
python --version

# PyTorch + CUDA info
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'cuDNN: {torch.backends.cudnn.version()}')"

# Package versions
pip list | grep -E "llmcompressor|torch|transformers|compressed-tensors|accelerate|datasets"

# GPU info
nvidia-smi

# System info
uname -a
free -h
```

## Contact
- **Issue Reporter**: [Your GitHub username]
- **Date**: [Current date]
- **llmcompressor commit/version**: [Get from `pip show llmcompressor`]

