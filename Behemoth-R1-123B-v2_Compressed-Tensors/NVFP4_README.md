# NVFP4 Quantization Scripts for Behemoth-R1-123B-v2

This directory contains two NVFP4 quantization scripts optimized for NVIDIA Blackwell GPUs.

## 📊 Script Comparison

| Feature | `llama_NVFP4A16.py` | `llama_NVFP4.py` |
|---------|---------------------|------------------|
| **Scheme** | W4A16 | W4A4 |
| **Weights** | FP4 (4-bit) | FP4 (4-bit) |
| **Activations** | FP16/BF16 (16-bit) | FP4 (4-bit) |
| **Calibration Data** | ❌ Not required | ✅ Required (256 samples) |
| **Compression** | ~4x | ~8x |
| **Speed** | ⚡ Faster (no calibration) | 🐢 Slower (needs calibration) |
| **Quality** | 🎯 Excellent | 🎯 Very Good |
| **Runtime** | ~30-60 min | ~2-4 hours |

## 🎯 Which Script Should You Use?

### Use `llama_NVFP4A16.py` when:
- ✅ You want **faster quantization** (no calibration needed)
- ✅ You prioritize **maximum quality** retention
- ✅ You're okay with **~4x compression** instead of ~8x
- ✅ You want a **simpler, quicker process**
- ✅ **Recommended for most users**

### Use `llama_NVFP4.py` when:
- ✅ You need **maximum compression** (4-bit activations too)
- ✅ You have time for calibration (~2-4 hours)
- ✅ You're deploying on **Blackwell GPUs** (SM 9.0+)
- ✅ Memory constraints require the smallest possible model

## 🚀 Running the Scripts

Both scripts use the same `.env` file format:

```bash
# .env file (in the same directory as the scripts)
SRC_DIR=/path/to/Behemoth-R1-123B-v2
DST_DIR=/path/to/output/directory
```

### Run NVFP4A16 (W4A16):
```bash
cd Behemoth-R1-123B-v2_Compressed-Tensors
python llama_NVFP4A16.py
```

### Run NVFP4 (W4A4):
```bash
cd Behemoth-R1-123B-v2_Compressed-Tensors
python llama_NVFP4.py
```

## 💾 Memory Requirements

Both scripts use **sequential onloading**, which loads layers one at a time:

- **Minimum GPU VRAM**: 24GB (e.g., RTX 4090, A5000)
- **Recommended GPU VRAM**: 48GB+ (e.g., A6000, L40S)
- **System RAM**: 300GB+ recommended for 123B model

The scripts automatically restrict to a single GPU for sequential processing.

## 🎮 GPU Compatibility

### On Blackwell GPUs (SM 9.0+):
- ✅ Full acceleration for both W4A16 and W4A4
- ✅ Dynamic per-token activation quantization (W4A4)
- ✅ Maximum performance

### On Pre-Blackwell GPUs (< SM 9.0):
- ✅ Both scripts will work
- ⚠️ W4A4 will run as **weights-only** (activations stay FP16)
- ⚠️ Performance similar to W4A16 on older GPUs

## 📝 Key Technical Details

### NVFP4A16 (W4A16)
- **Weight Quantization**: FP4 with per-group-16 scales
- **Activation Quantization**: None (stays FP16/BF16)
- **No calibration needed**: Weights-only PTQ
- **Fast execution**: Simple forward pass

### NVFP4 (W4A4)
- **Weight Quantization**: FP4 with per-group-16 scales
- **Activation Quantization**: FP4 dynamic per-token
- **Calibration required**: Generates global activation scales
- **Dataset**: UltraChat 200k (256 samples, 2048 seq length)
- **Runtime scales**: Generated dynamically during inference

## 🔧 Customization

### Adjust calibration samples (NVFP4 only):
```python
NUM_CALIBRATION_SAMPLES = 512  # Increase for better accuracy
MAX_SEQUENCE_LENGTH = 4096     # Increase for longer context
```

### Change calibration dataset (NVFP4 only):
```python
# For reasoning models like Behemoth-R1:
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Alternative datasets:
# DATASET_ID = "wikitext"
# DATASET_NAME = "wikitext-103-raw-v1"
```

## 📚 References

- [LLM Compressor FP4 Guide](https://github.com/vllm-project/llm-compressor/tree/main/examples/quantization_w4a4_fp4)
- [Compression Schemes Docs](https://github.com/vllm-project/llm-compressor/blob/main/docs/guides/compression_schemes.md)
- [Sequential Onloading Guide](https://github.com/vllm-project/llm-compressor/tree/main/examples/big_models_with_sequential_onloading)

## ⚡ Performance Tips

1. **Single GPU**: Scripts automatically use one GPU for sequential onloading
2. **Dataset Loading**: Uses 4 parallel workers for faster preprocessing
3. **Memory**: Close other applications to maximize available RAM
4. **Monitoring**: Use `nvidia-smi` to monitor GPU utilization during quantization

## 🐛 Troubleshooting

**Error: "Missing environment variable: SRC_DIR"**
- Ensure you have a `.env` file in the same directory as the script
- Check that `SRC_DIR` and `DST_DIR` are defined

**Error: "Model directory does not exist"**
- Verify your `SRC_DIR` path points to the actual model directory
- Use absolute paths or ensure relative paths are correct

**Out of Memory Error:**
- Close other GPU applications
- Ensure you have enough system RAM (300GB+ for 123B)
- Try reducing `NUM_CALIBRATION_SAMPLES` (NVFP4 only)

**Slow calibration (NVFP4):**
- This is normal! Calibration with 256 samples takes 2-4 hours
- You can reduce `NUM_CALIBRATION_SAMPLES` to 128 for faster testing
- Higher sample counts generally improve quality

