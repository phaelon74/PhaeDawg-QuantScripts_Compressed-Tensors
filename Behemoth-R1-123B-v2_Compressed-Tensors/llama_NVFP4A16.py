import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
# Force Single GPU for Sequential Onloading
# =========================
# In multi-GPU environments, restrict to ONE GPU for sequential processing
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    print("⚠️  Multi-GPU detected. Restricting to GPU 0 for sequential onloading.")
    print("   (Sequential onloading uses ONE GPU to process layers one at a time)")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    print(f"Using GPU(s): {os.environ['CUDA_VISIBLE_DEVICES']}")

# Reduce CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

# CRITICAL: Load model to CPU first (device_map=None) for sequential onloading
print("\n" + "="*70)
print("Loading model to CPU (device_map=None)")
print("Sequential onloading will process layers one-by-one during quantization")
print("="*70 + "\n")

# Clear GPU cache before starting
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("✓ CUDA cache cleared")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map=None,  # Load to CPU, not GPU! Critical for sequential onloading
    trust_remote_code=True,
    local_files_only=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    trust_remote_code=True,
    local_files_only=True  # Use local files, don't try to download from HF Hub
)

# =========================
# Configure the quantization scheme.
# =========================
# NVFP4A16 quantizes:
#   * Weights to FP4 (4-bit floating point) with per-group-16 quantization
#   * Activations remain in FP16/BF16 (A16)
#
# Key advantages:
#   - NO calibration data needed (weights-only quantization)
#   - Fast quantization process
#   - Optimized for NVIDIA Blackwell GPUs (SM 9.0+)
#   - Good balance of compression and quality

recipe = QuantizationModifier(
    targets="Linear", 
    scheme="NVFP4A16", 
    ignore=["lm_head"]
)

# =========================
# Apply quantization with sequential onloading.
# =========================
# Model is loaded to CPU (device_map=None), then oneshot() will:
# 1. Load each layer one at a time: CPU -> GPU
# 2. Apply FP4 weight quantization to that layer
# 3. Store quantized layer and offload from GPU
# This prevents VRAM exhaustion for large models like Behemoth-R1-123B-v2.

print("\n" + "="*70)
print("Starting NVFP4A16 quantization with sequential onloading...")
print("="*70)
print(f"✓ Scheme: NVFP4A16 (W4A16)")
print(f"✓ Weights: FP4 per-group-16")
print(f"✓ Activations: FP16/BF16 (unquantized)")
print(f"✓ Calibration: NOT required (weights-only)")
print(f"✓ Optimized for: NVIDIA Blackwell GPUs (SM 9.0+)")
print(f"✓ Memory efficient: Works on 24GB+ GPUs for 123B models")
print("="*70 + "\n")

oneshot(
    model=model,
    recipe=recipe,
    sequential_targets=["MistralMLP"],  # CRITICAL: Process MLP layers one at a time to avoid OOM
)

print("\n" + "="*70)
print("✅ Quantization completed successfully!")
print("="*70)

# =========================
# Save to disk in compressed-tensors format.
# =========================
SAVE_DIR = require_env("DST_DIR").rstrip("/")  # Remove trailing slash if present
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print(f"\n✅ Model successfully quantized and saved to: {SAVE_DIR}")
print("Quantization scheme: NVFP4A16 (W4A16)")
print("Ready for inference on Blackwell GPUs with vLLM! 🚀")
print("\nNote: On non-Blackwell GPUs (< SM 9.0), vLLM will run as weights-only quantization.")

