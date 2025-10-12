# Multi-GPU Setup for FP8_BLOCK Quantization

## Why Multi-GPU is Required

FP8_BLOCK quantization **accumulates the quantized model in GPU memory** during processing. For Behemoth-R1-123B-v2:

- **FP8 quantized model size**: ~123GB
- **Quantization metadata** (block scales, etc.): ~20-30GB  
- **Total GPU memory needed**: **~150-160GB**

### Single GPU Failure Points

| GPU | VRAM | Fails At | Memory Used |
|-----|------|----------|-------------|
| RTX 3090 | 24GB | 90% (1588/1767) | ~23.5GB |
| L40S | 44GB | 90% (557/616) | ~44GB |
| H200 | 144GB | ~87% | 125GB/144GB |

**Pattern**: OOMs when accumulated quantized model approaches GPU capacity

## Multi-GPU Configuration

### Current Script Setup

The script is configured to use **2 GPUs by default**:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
```

### GPU Requirements for Success

| Configuration | Total VRAM | Expected Result |
|---------------|------------|-----------------|
| **2× L40S (44GB)** | **88GB** | ⚠️ May still OOM around 85-90% |
| **2× A100 (80GB)** | **160GB** | ✅ Should complete |
| **2× H100 (80GB)** | **160GB** | ✅ Should complete |
| **1× H200 (144GB)** | **144GB** | ⚠️ Borderline - may OOM |
| **2× H200 (144GB)** | **288GB** | ✅ Plenty of headroom |

## How to Run

### Default (2 GPUs)

```bash
cd /workspace/llmcompressor
python3 llama_W8A8-FP8_BLOCK.py
```

The script will automatically use GPUs 0 and 1.

### Custom GPU Selection

```bash
# Use GPUs 0 and 2
export CUDA_VISIBLE_DEVICES=0,2
python3 llama_W8A8-FP8_BLOCK.py

# Use GPUs 1, 2, and 3 (3 GPUs total)
export CUDA_VISIBLE_DEVICES=1,2,3
python3 llama_W8A8-FP8_BLOCK.py
```

### Use All 4 L40S GPUs

```bash
# Use all 4 GPUs for maximum memory
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 llama_W8A8-FP8_BLOCK.py
```

With 4× L40S (176GB total), you should have enough memory to complete.

## What to Monitor

The script will show GPU memory usage:

```
📊 GPU Memory Status (Before Quantization):
   GPU 0: 0.12GB used / 44.40GB total
   GPU 1: 0.12GB used / 44.40GB total
   Total Available: 88.80GB across 2 GPU(s)

💡 FP8_BLOCK Note: Quantized model accumulates in GPU memory
   Expected peak usage: ~150-160GB for 123B model
```

Watch `nvidia-smi` during quantization to see memory grow:

```bash
watch -n 1 nvidia-smi
```

## Expected Behavior

### Successful Completion
- Memory usage grows gradually as layers are quantized
- Peak usage reaches ~140-150GB (spread across GPUs)
- Completes all layers without OOM

### If Still OOMs
- Try using more GPUs (3 or 4 L40S = 132-176GB)
- Or consider using FP8_DYNAMIC instead (works on single 24GB+ GPU)

## Alternative: FP8_DYNAMIC

If multi-GPU still fails, FP8_DYNAMIC is the recommended alternative:
- ✅ Works on single 24GB+ GPU
- ✅ Does NOT accumulate full model in GPU memory
- ✅ Recommended for Blackwell inference
- ✅ Faster quantization process

```bash
python3 llama_W8A8-FP8.py  # Uses FP8_DYNAMIC scheme
```

## Technical Details

### Why FP8_BLOCK Accumulates Memory

The sequential onloading pipeline processes layers one at a time, but:
1. Each **quantized layer** stays in GPU memory after processing
2. Block-wise quantization metadata is also retained on GPU
3. By 85-90% completion, the accumulated quantized model fills available VRAM
4. Final layers trigger OOM when trying to allocate quantization buffers

### Memory Distribution Across GPUs

PyTorch and llm-compressor automatically distribute the model across available GPUs. You don't need to manually specify device placement - the framework handles it.

## Troubleshooting

### Still Getting OOM with 2 GPUs?

**Try 4 GPUs:**
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 llama_W8A8-FP8_BLOCK.py
```

### Want to see exact memory usage?

Add this to monitor during quantization:
```bash
# In another terminal
watch -d -n 0.5 nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
```

### Success Criteria

You'll know it's working if:
- ✅ Memory usage grows gradually across all GPUs
- ✅ Passes the 90% mark without OOM
- ✅ Completes "Calibrating weights" progress bar to 100%
- ✅ Shows "Quantization completed successfully!"

## Summary

**For 4× L40S (176GB total)**: Should work fine ✅  
**For 2× L40S (88GB total)**: May still OOM ⚠️  
**For 1× H200 (144GB)**: Borderline, may OOM ⚠️  
**For 2× A100/H100 (160GB+)**: Should work ✅

