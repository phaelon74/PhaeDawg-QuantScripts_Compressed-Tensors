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
# By passing model_path as a string (instead of a loaded model), oneshot() will
# load layers one at a time from disk -> GPU -> process -> store in system RAM.
# This prevents VRAM exhaustion for large models like Behemoth-R1-123B-v2.
#
# Note: W8A8-FP8_BLOCK does NOT require a calibration dataset.
# The quantization is performed via PTQ (Post-Training Quantization) without data.
model = oneshot(
    model=model_path,
    recipe=recipe,
    trust_remote_code_model=True,
    cache_dir=None,  # Don't use HF cache since we're loading locally
)



# =========================
# Save to disk in compressed-tensors format.
# =========================
SAVE_DIR = require_env("DST_DIR").rstrip("/")  # Remove trailing slash if present
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print(f"\n✅ Model successfully quantized and saved to: {SAVE_DIR}")
print("Quantization scheme: W8A8-FP8_BLOCK (block-wise FP8 quantization)")

