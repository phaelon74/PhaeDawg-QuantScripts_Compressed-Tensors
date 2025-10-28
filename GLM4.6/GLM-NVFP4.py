import torch
from datasets import load_dataset
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

def require_env(key: str) -> str:
    val = os.getenv(key)
    if not val or not val.strip():
        raise RuntimeError(f"Missing environment variable: {key}")
    return val.strip()

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

# Reduce CUDA memory fragmentation and improve memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

# =========================
# Model (GLM-4.6 MoE)
# =========================
MODEL_ID = require_env("SRC_DIR").rstrip("/")  # Remove trailing slash if present

# Verify the model directory exists
if not os.path.isdir(MODEL_ID):
    raise RuntimeError(f"Model directory does not exist: {MODEL_ID}")

# Convert to absolute path to ensure it's recognized as a local directory
model_path = str(Path(MODEL_ID).resolve())
print(f"Loading GLM-4.6 model from: {model_path}")

# Sequential onloading is enabled by default in llmcompressor 0.6.0+
# During calibration, llmcompressor will automatically load only one layer at a time to GPU
# This allows quantization of very large models (357B params) on a single GPU
print("\n" + "="*70)
print("Loading GLM-4.6 MoE model with automatic sequential onloading enabled")
print("llmcompressor will process layers one-by-one during calibration")
print("="*70 + "\n")

# Clear GPU cache before starting
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("✓ CUDA cache cleared and memory stats reset")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    trust_remote_code=True,
    local_files_only=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True
)

# =========================
# Calibration data config
# =========================
# NVFP4 (W4A4) requires calibration data to determine global activation scales.
# These scales are used during inference for dynamic per-token activation quantization.
# Using Neural Magic's default LLM compression calibration dataset.
# Source: https://huggingface.co/datasets/neuralmagic/LLM_compression_calibration
# - 10,000 samples from garage-bAInd/Open-Platypus
# - Specifically designed for one-shot LLM quantization
# - Provides high-quality diverse calibration data

NUM_CALIBRATION_SAMPLES = 512      # Use subset for faster calibration (max 10,000 available)
MAX_SEQUENCE_LENGTH = 2048         # Increased to leverage GLM-4.6's extended context (200K)

DATASET_ID = "neuralmagic/LLM_compression_calibration"
DATASET_SPLIT = "train"

# =========================
# Load Neural Magic calibration dataset
# =========================
print("\n" + "="*70)
print("Loading calibration dataset...")
print("="*70)
print(f"Dataset: {DATASET_ID}")
print(f"Source: Open-Platypus (Neural Magic curated)")
print(f"Split: {DATASET_SPLIT}")
print(f"Samples: {NUM_CALIBRATION_SAMPLES} (of 10,000 available)")
print(f"Max sequence length: {MAX_SEQUENCE_LENGTH}")
print("="*70 + "\n")

ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

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
# Configure the quantization scheme for GLM-4.6 MoE
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
# MoE-specific handling:
#   - Keep first layer (layer 0) and lm_head unquantized for stability
#   - Keep shared_experts in all MoE layers unquantized (critical for MoE routing)
#   - This preserves model quality while achieving significant compression
#
# Calibration:
#   - Generates per-tensor global scales for activations
#   - Per-group local activation scales generated dynamically during inference

moe_ignores = [
    # Layer 0 (first layer) - keep unquantized for stability
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.0.mlp.down_proj",
    # Shared experts across all MoE layers (layers 1-45)
    "model.layers.1.mlp.shared_experts.gate_proj",
    "model.layers.1.mlp.shared_experts.up_proj",
    "model.layers.1.mlp.shared_experts.down_proj",
    "model.layers.2.mlp.shared_experts.gate_proj",
    "model.layers.2.mlp.shared_experts.up_proj",
    "model.layers.2.mlp.shared_experts.down_proj",
    "model.layers.3.mlp.shared_experts.gate_proj",
    "model.layers.3.mlp.shared_experts.up_proj",
    "model.layers.3.mlp.shared_experts.down_proj",
    "model.layers.4.mlp.shared_experts.gate_proj",
    "model.layers.4.mlp.shared_experts.up_proj",
    "model.layers.4.mlp.shared_experts.down_proj",
    "model.layers.5.mlp.shared_experts.gate_proj",
    "model.layers.5.mlp.shared_experts.up_proj",
    "model.layers.5.mlp.shared_experts.down_proj",
    "model.layers.6.mlp.shared_experts.gate_proj",
    "model.layers.6.mlp.shared_experts.up_proj",
    "model.layers.6.mlp.shared_experts.down_proj",
    "model.layers.7.mlp.shared_experts.gate_proj",
    "model.layers.7.mlp.shared_experts.up_proj",
    "model.layers.7.mlp.shared_experts.down_proj",
    "model.layers.8.mlp.shared_experts.gate_proj",
    "model.layers.8.mlp.shared_experts.up_proj",
    "model.layers.8.mlp.shared_experts.down_proj",
    "model.layers.9.mlp.shared_experts.gate_proj",
    "model.layers.9.mlp.shared_experts.up_proj",
    "model.layers.9.mlp.shared_experts.down_proj",
    "model.layers.10.mlp.shared_experts.gate_proj",
    "model.layers.10.mlp.shared_experts.up_proj",
    "model.layers.10.mlp.shared_experts.down_proj",
    "model.layers.11.mlp.shared_experts.gate_proj",
    "model.layers.11.mlp.shared_experts.up_proj",
    "model.layers.11.mlp.shared_experts.down_proj",
    "model.layers.12.mlp.shared_experts.gate_proj",
    "model.layers.12.mlp.shared_experts.up_proj",
    "model.layers.12.mlp.shared_experts.down_proj",
    "model.layers.13.mlp.shared_experts.gate_proj",
    "model.layers.13.mlp.shared_experts.up_proj",
    "model.layers.13.mlp.shared_experts.down_proj",
    "model.layers.14.mlp.shared_experts.gate_proj",
    "model.layers.14.mlp.shared_experts.up_proj",
    "model.layers.14.mlp.shared_experts.down_proj",
    "model.layers.15.mlp.shared_experts.gate_proj",
    "model.layers.15.mlp.shared_experts.up_proj",
    "model.layers.15.mlp.shared_experts.down_proj",
    "model.layers.16.mlp.shared_experts.gate_proj",
    "model.layers.16.mlp.shared_experts.up_proj",
    "model.layers.16.mlp.shared_experts.down_proj",
    "model.layers.17.mlp.shared_experts.gate_proj",
    "model.layers.17.mlp.shared_experts.up_proj",
    "model.layers.17.mlp.shared_experts.down_proj",
    "model.layers.18.mlp.shared_experts.gate_proj",
    "model.layers.18.mlp.shared_experts.up_proj",
    "model.layers.18.mlp.shared_experts.down_proj",
    "model.layers.19.mlp.shared_experts.gate_proj",
    "model.layers.19.mlp.shared_experts.up_proj",
    "model.layers.19.mlp.shared_experts.down_proj",
    "model.layers.20.mlp.shared_experts.gate_proj",
    "model.layers.20.mlp.shared_experts.up_proj",
    "model.layers.20.mlp.shared_experts.down_proj",
    "model.layers.21.mlp.shared_experts.gate_proj",
    "model.layers.21.mlp.shared_experts.up_proj",
    "model.layers.21.mlp.shared_experts.down_proj",
    "model.layers.22.mlp.shared_experts.gate_proj",
    "model.layers.22.mlp.shared_experts.up_proj",
    "model.layers.22.mlp.shared_experts.down_proj",
    "model.layers.23.mlp.shared_experts.gate_proj",
    "model.layers.23.mlp.shared_experts.up_proj",
    "model.layers.23.mlp.shared_experts.down_proj",
    "model.layers.24.mlp.shared_experts.gate_proj",
    "model.layers.24.mlp.shared_experts.up_proj",
    "model.layers.24.mlp.shared_experts.down_proj",
    "model.layers.25.mlp.shared_experts.gate_proj",
    "model.layers.25.mlp.shared_experts.up_proj",
    "model.layers.25.mlp.shared_experts.down_proj",
    "model.layers.26.mlp.shared_experts.gate_proj",
    "model.layers.26.mlp.shared_experts.up_proj",
    "model.layers.26.mlp.shared_experts.down_proj",
    "model.layers.27.mlp.shared_experts.gate_proj",
    "model.layers.27.mlp.shared_experts.up_proj",
    "model.layers.27.mlp.shared_experts.down_proj",
    "model.layers.28.mlp.shared_experts.gate_proj",
    "model.layers.28.mlp.shared_experts.up_proj",
    "model.layers.28.mlp.shared_experts.down_proj",
    "model.layers.29.mlp.shared_experts.gate_proj",
    "model.layers.29.mlp.shared_experts.up_proj",
    "model.layers.29.mlp.shared_experts.down_proj",
    "model.layers.30.mlp.shared_experts.gate_proj",
    "model.layers.30.mlp.shared_experts.up_proj",
    "model.layers.30.mlp.shared_experts.down_proj",
    "model.layers.31.mlp.shared_experts.gate_proj",
    "model.layers.31.mlp.shared_experts.up_proj",
    "model.layers.31.mlp.shared_experts.down_proj",
    "model.layers.32.mlp.shared_experts.gate_proj",
    "model.layers.32.mlp.shared_experts.up_proj",
    "model.layers.32.mlp.shared_experts.down_proj",
    "model.layers.33.mlp.shared_experts.gate_proj",
    "model.layers.33.mlp.shared_experts.up_proj",
    "model.layers.33.mlp.shared_experts.down_proj",
    "model.layers.34.mlp.shared_experts.gate_proj",
    "model.layers.34.mlp.shared_experts.up_proj",
    "model.layers.34.mlp.shared_experts.down_proj",
    "model.layers.35.mlp.shared_experts.gate_proj",
    "model.layers.35.mlp.shared_experts.up_proj",
    "model.layers.35.mlp.shared_experts.down_proj",
    "model.layers.36.mlp.shared_experts.gate_proj",
    "model.layers.36.mlp.shared_experts.up_proj",
    "model.layers.36.mlp.shared_experts.down_proj",
    "model.layers.37.mlp.shared_experts.gate_proj",
    "model.layers.37.mlp.shared_experts.up_proj",
    "model.layers.37.mlp.shared_experts.down_proj",
    "model.layers.38.mlp.shared_experts.gate_proj",
    "model.layers.38.mlp.shared_experts.up_proj",
    "model.layers.38.mlp.shared_experts.down_proj",
    "model.layers.39.mlp.shared_experts.gate_proj",
    "model.layers.39.mlp.shared_experts.up_proj",
    "model.layers.39.mlp.shared_experts.down_proj",
    "model.layers.40.mlp.shared_experts.gate_proj",
    "model.layers.40.mlp.shared_experts.up_proj",
    "model.layers.40.mlp.shared_experts.down_proj",
    "model.layers.41.mlp.shared_experts.gate_proj",
    "model.layers.41.mlp.shared_experts.up_proj",
    "model.layers.41.mlp.shared_experts.down_proj",
    "model.layers.42.mlp.shared_experts.gate_proj",
    "model.layers.42.mlp.shared_experts.up_proj",
    "model.layers.42.mlp.shared_experts.down_proj",
    "model.layers.43.mlp.shared_experts.gate_proj",
    "model.layers.43.mlp.shared_experts.up_proj",
    "model.layers.43.mlp.shared_experts.down_proj",
    "model.layers.44.mlp.shared_experts.gate_proj",
    "model.layers.44.mlp.shared_experts.up_proj",
    "model.layers.44.mlp.shared_experts.down_proj",
    "model.layers.45.mlp.shared_experts.gate_proj",
    "model.layers.45.mlp.shared_experts.up_proj",
    "model.layers.45.mlp.shared_experts.down_proj",
    # Output layer - keep unquantized
    "lm_head",
]

# NVFP4 recipe - applies to all Linear layers except those in ignore list
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=moe_ignores
)

# Clear cache again before quantization starts
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("✓ Cleared GPU cache before quantization\n")

# =========================
# Apply quantization with sequential onloading.
# =========================
# Sequential onloading (default in llmcompressor 0.6.0+) automatically:
# 1. Loads each layer one at a time to GPU
# 2. Runs calibration forward passes on that layer
# 3. Collects activation statistics and quantizes
# 4. Offloads layer before loading the next
# This prevents VRAM exhaustion for large MoE models like GLM-4.6 (357B params).

print("\n" + "="*70)
print("Starting NVFP4 quantization with sequential onloading...")
print("="*70)
print(f"✓ Scheme: NVFP4 (W4A4)")
print(f"✓ Weights: FP4 per-group-16")
print(f"✓ Activations: FP4 dynamic per-token")
print(f"✓ Calibration: {NUM_CALIBRATION_SAMPLES} samples from Neural Magic dataset")
print(f"  (Open-Platypus curated for LLM compression)")
print(f"✓ Model: GLM-4.6 MoE (357B params)")
print(f"✓ MoE handling: Shared experts + first layer + lm_head unquantized")
print(f"✓ Optimized for: NVIDIA Blackwell GPUs (SM 9.0+)")
print(f"✓ Memory efficient: Works on 24GB+ GPUs via sequential processing")
print("="*70 + "\n")

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
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
print("Quantization scheme: NVFP4 (W4A4) for GLM-4.6 MoE")
print("Ready for inference on Blackwell GPUs with vLLM! 🚀")
print("\nNote: On non-Blackwell GPUs (< SM 9.0), vLLM will run weights-only quantization")
print("      (activations will remain unquantized for compatibility).")
print("\nMoE-specific optimizations:")
print("  - Shared experts preserved in FP16/BF16 for routing stability")
print("  - First layer and lm_head kept unquantized for quality")
print("  - Routed experts quantized to NVFP4 for maximum compression")

