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
MODEL_ID = require_env("SRC_DIR")

# Load tokenizer (model will be loaded layer-by-layer by oneshot)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

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
# By passing MODEL_ID as a string (instead of a loaded model), oneshot() will
# load layers one at a time from disk -> GPU -> process -> store in system RAM.
# This prevents VRAM exhaustion for large models like Behemoth-R1-123B-v2.
model = oneshot(
    model=MODEL_ID,
    recipe=recipe,
    trust_remote_code_model=True,
)



# =========================
# Save to disk in compressed-tensors format.
# =========================
SAVE_DIR = require_env("DST_DIR")
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

