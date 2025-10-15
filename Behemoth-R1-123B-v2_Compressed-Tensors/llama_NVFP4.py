from datasets import load_dataset
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
# Calibration data config
# =========================
# NVFP4 (W4A4) requires calibration data to determine global activation scales.
# These scales are used during inference for dynamic per-token activation quantization.

NUM_CALIBRATION_SAMPLES = 512      # Adjust as needed (512 can improve accuracy)
MAX_SEQUENCE_LENGTH = 2048         # Increased for better context coverage

DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# =========================
# Load + sample UltraChat
# =========================
print("\n" + "="*70)
print("Loading calibration dataset...")
print("="*70)
print(f"Dataset: {DATASET_ID}")
print(f"Split: {DATASET_SPLIT}")
print(f"Samples: {NUM_CALIBRATION_SAMPLES}")
print(f"Max sequence length: {MAX_SEQUENCE_LENGTH}")
print("="*70 + "\n")

ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)

# =========================
# Preprocess (batch-aware)
# =========================
def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }

# Render chat template in batches
ds = ds.map(preprocess, batched=False, num_proc=4)

# =========================
# Tokenize in batches
# =========================
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )

ds = ds.map(tokenize, remove_columns=ds.column_names, num_proc=4)

# =========================
# Configure the quantization scheme.
# =========================
# NVFP4 (W4A4) quantizes:
#   * Weights to FP4 (4-bit floating point) with per-group-16 quantization
#   * Activations to FP4 with dynamic per-token quantization
#
# Key advantages:
#   - Maximum compression (4-bit weights AND activations)
#   - Optimized for NVIDIA Blackwell GPUs (SM 9.0+)
#   - Maintains quality through FP4 format vs INT4
#
# Calibration:
#   - Generates per-tensor global scales for activations
#   - Per-group local activation scales generated dynamically during inference

recipe = QuantizationModifier(
    targets="Linear", 
    scheme="NVFP4", 
    ignore=["lm_head"]
)

# =========================
# Apply quantization with sequential onloading.
# =========================
# By passing model_path as a string (instead of a loaded model), oneshot() will
# load layers one at a time from disk -> GPU -> process -> store in system RAM.
# This prevents VRAM exhaustion for large models like Behemoth-R1-123B-v2.

print("\n" + "="*70)
print("Starting NVFP4 quantization with sequential onloading...")
print("="*70)
print(f"✓ Scheme: NVFP4 (W4A4)")
print(f"✓ Weights: FP4 per-group-16")
print(f"✓ Activations: FP4 dynamic per-token")
print(f"✓ Calibration: {NUM_CALIBRATION_SAMPLES} samples from {DATASET_ID}")
print(f"✓ Optimized for: NVIDIA Blackwell GPUs (SM 9.0+)")
print(f"✓ Memory efficient: Works on 24GB+ GPUs for 123B models")
print("="*70 + "\n")

model = oneshot(
    model=model_path,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
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
print("Quantization scheme: NVFP4 (W4A4)")
print("Ready for inference on Blackwell GPUs with vLLM! 🚀")
print("\nNote: On non-Blackwell GPUs (< SM 9.0), vLLM will run weights-only quantization")
print("      (activations will remain unquantized for compatibility).")

