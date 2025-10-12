from transformers import AutoTokenizer
import torch

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# =========================
# Load ENV Variables
# =========================
from pathlib import Path
import os
from dotenv import load_dotenv

# Load the .env that sits next to this script (works regardless of where you run it)
load_dotenv(Path(__file__).with_name(".env"))

# =========================
# Configure Multi-GPU Setup for FP8_BLOCK
# =========================
# FP8_BLOCK accumulates the quantized model in GPU memory during processing
# For 123B models, this requires ~150GB+ total GPU memory
# Using 2 GPUs provides more memory headroom
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    print("🔧 Multi-GPU Setup: Using GPUs 0 and 1 for FP8_BLOCK quantization")
    print("   FP8_BLOCK requires significant GPU memory due to block-wise quantization")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    num_gpus = 2
else:
    gpu_list = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    num_gpus = len(gpu_list)
    print(f"Using {num_gpus} GPU(s): {os.environ['CUDA_VISIBLE_DEVICES']}")

def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing environment variable: {name}")
    return val

# =========================
# Model
# =========================
MODEL_ID = require_env("SRC_DIR").rstrip("/")  # Remove trailing slash if present

# Verify the model directory exists
if not os.path.isdir(MODEL_ID):
    raise RuntimeError(f"Model directory does not exist: {MODEL_ID}")

# Convert to absolute path to ensure it's recognized as a local directory
model_path = str(Path(MODEL_ID).resolve())
print(f"Loading model from: {model_path}")

# Load tokenizer ONLY - do NOT load the model yet!
# Sequential onloading requires passing the model PATH to oneshot(), not a loaded model
tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    trust_remote_code=True,
    local_files_only=True  # Use local files, don't try to download from HF Hub
)
print("✓ Tokenizer loaded successfully")

# =========================
# Configure the quantization algorithm and scheme.
# =========================
# In this case, we:
#   * quantize the weights to fp8 with block-wise quantization (128x128 tiles) via ptq
#   * quantize the activations to fp8 with dynamic per-token-group (128) quantization
#
# Key difference from FP8_DYNAMIC:
#   - FP8_BLOCK uses block-wise quantization for weights instead of channel-wise
#   - Provides better granularity for quantization in large models
#   - No calibration dataset is required (quantization is done without data)
recipe = QuantizationModifier(
    targets="Linear", 
    scheme="FP8_BLOCK", 
    ignore=["lm_head"]
)

# =========================
# Apply quantization with sequential onloading.
# =========================
# CRITICAL: Pass the model PATH (string), NOT a loaded model object!
# Sequential onloading in llm-compressor works by:
#   1. Load one layer from disk directly to GPU
#   2. Apply FP8_BLOCK quantization
#   3. Offload to RAM before loading next layer
#   4. Repeat for all layers
#
# This prevents VRAM exhaustion for large models like Behemoth-R1-123B-v2.
#
# Note: W8A8-FP8_BLOCK does NOT require a calibration dataset.
# The quantization is performed via PTQ (Post-Training Quantization) without data.

print("\n" + "="*70)
print("Starting FP8_BLOCK quantization with sequential onloading...")
print("="*70)
print(f"✓ Model path: {model_path}")
print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"✓ Current CUDA device: {torch.cuda.current_device()}")
    print("\n📊 GPU Memory Status (Before Quantization):")
    for i in range(torch.cuda.device_count()):
        mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
        mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
        print(f"   GPU {i}: {mem_allocated:.2f}GB used / {mem_total:.2f}GB total")
    total_vram = sum(torch.cuda.get_device_properties(i).total_memory / (1024**3) 
                     for i in range(torch.cuda.device_count()))
    print(f"   Total Available: {total_vram:.2f}GB across {torch.cuda.device_count()} GPU(s)")
    print(f"\n💡 FP8_BLOCK Note: Quantized model accumulates in GPU memory")
    print(f"   Expected peak usage: ~150-160GB for 123B model")
print("="*70 + "\n")

model = oneshot(
    model=model_path,  # Pass PATH string for sequential onloading
    recipe=recipe,
    trust_remote_code_model=True,
    cache_dir=None,  # Don't use HF cache since we're loading locally
)

print("\n" + "="*70)
print("✅ Quantization completed successfully!")
print("="*70)

# Report final GPU memory usage
if torch.cuda.is_available():
    print("\n📊 GPU Memory Status (After Quantization):")
    for i in range(torch.cuda.device_count()):
        mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
        mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
        print(f"   GPU {i}: {mem_allocated:.2f}GB used / {mem_total:.2f}GB total ({mem_allocated/mem_total*100:.1f}%)")
    total_used = sum(torch.cuda.memory_allocated(i) / (1024**3) for i in range(torch.cuda.device_count()))
    total_vram = sum(torch.cuda.get_device_properties(i).total_memory / (1024**3) 
                     for i in range(torch.cuda.device_count()))
    print(f"   Total Used: {total_used:.2f}GB / {total_vram:.2f}GB ({total_used/total_vram*100:.1f}%)")
    print()



# =========================
# Save to disk in compressed-tensors format.
# =========================
SAVE_DIR = require_env("DST_DIR").rstrip("/")  # Remove trailing slash if present
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print(f"\n✅ Model successfully quantized and saved to: {SAVE_DIR}")
print("Quantization scheme: W8A8-FP8_BLOCK (block-wise FP8 quantization)")

