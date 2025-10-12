from transformers import AutoTokenizer

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

# Load tokenizer (model will be loaded layer-by-layer by oneshot)
tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    trust_remote_code=True,
    local_files_only=True  # Use local files, don't try to download from HF Hub
)

# =========================
# Configure the quantization algorithm and scheme.
# =========================
# In this case, we:
#   * quantize the weights to fp8 with per channel via ptq
#   * quantize the activations to fp8 with dynamic per token
recipe = QuantizationModifier(
    targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
)

# =========================
# Apply quantization with sequential onloading.
# =========================
# By passing model_path as a string (instead of a loaded model), oneshot() will
# load layers one at a time from disk -> GPU -> process -> store in system RAM.
# This prevents VRAM exhaustion for large models like Behemoth-R1-123B-v2.
#
# FP8_DYNAMIC uses per-channel quantization which is MUCH more memory-efficient
# than FP8_BLOCK's block-wise quantization, and is recommended for Blackwell GPUs.

print("\n" + "="*70)
print("Starting FP8_DYNAMIC quantization with sequential onloading...")
print("="*70)
print(f"✓ Scheme: FP8_DYNAMIC (per-channel weights, dynamic per-token activations)")
print(f"✓ Recommended for: Blackwell GPUs (compute capability 9.0+)")
print(f"✓ Memory efficient: Works on 24GB+ GPUs for 123B models")
print("="*70 + "\n")

model = oneshot(
    model=model_path,
    recipe=recipe,
    trust_remote_code_model=True,
    cache_dir=None,  # Don't use HF cache since we're loading locally
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
print("Quantization scheme: W8A8-FP8_DYNAMIC (Blackwell-optimized)")
print("Ready for inference on Blackwell GPUs with vLLM! 🚀")

