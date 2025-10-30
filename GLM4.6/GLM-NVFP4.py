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
# Multi-GPU Configuration for NVFP4
# =========================
# NVFP4 calibration requires ~195GB VRAM for GLM-4.6 (357B params)
# Using BOTH RTX PRO 6000 cards (~194GB total) for model parallelism
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    print("✅ Multi-GPU setup detected - Using BOTH GPUs for NVFP4 calibration")
    print("   GPU 0 + GPU 1 = ~194GB total VRAM (required for 357B model)")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
else:
    print(f"Using GPU(s): {os.environ['CUDA_VISIBLE_DEVICES']}")

# Reduce CUDA memory fragmentation and improve memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

# Check available GPUs
import torch
num_gpus = torch.cuda.device_count()
print(f"✓ Detected {num_gpus} GPU(s) available for NVFP4 quantization")
for i in range(num_gpus):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name} - {props.total_memory / 1024**3:.1f} GB VRAM")

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

# Multi-GPU model parallelism for NVFP4 calibration
# device_map="auto" will automatically distribute the model across both GPUs
# This enables NVFP4 calibration of 357B models on 2x RTX PRO 6000 (~194GB total)
print("\n" + "="*70)
print("Loading GLM-4.6 MoE model with multi-GPU parallelism")
print("Model will be distributed across both GPUs for NVFP4 calibration")
print("="*70 + "\n")

# Clear GPU cache before starting
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("✓ CUDA cache cleared on all GPUs\n")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    trust_remote_code=True,
    local_files_only=True
)

# Print model structure to identify correct sequential_targets
print("\n" + "="*70)
print("MODEL ARCHITECTURE (identifying layer types for sequential processing):")
print("="*70)
layer_types = set()
for name, module in model.named_modules():
    module_type = type(module).__name__
    if "MLP" in module_type or "Block" in module_type or "Layer" in module_type:
        layer_types.add(module_type)
print("Key module types found:")
for lt in sorted(layer_types):
    print(f"  - {lt}")
print("="*70 + "\n")

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
                                   # Can increase to 1024+ if desired
MAX_SEQUENCE_LENGTH = 2048         # Leverages GLM-4.6's extended context capability (200K)
                                   # 2048 tokens works well with dual-GPU setup (~194GB VRAM)

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
# Apply quantization with multi-GPU parallelism.
# =========================
# Multi-GPU setup (device_map="auto") automatically:
# 1. Distributes model layers across GPU 0 and GPU 1
# 2. Runs calibration forward passes through the distributed model
# 3. Collects activation statistics and quantizes
# 4. Utilizes combined VRAM (~194GB) for 357B model NVFP4 calibration

print("\n" + "="*70)
print("Starting NVFP4 quantization with multi-GPU parallelism...")
print("="*70)
print(f"✓ Scheme: NVFP4 (W4A4)")
print(f"✓ Weights: FP4 per-group-16")
print(f"✓ Activations: FP4 dynamic per-token")
print(f"✓ Calibration: {NUM_CALIBRATION_SAMPLES} samples from Neural Magic dataset")
print(f"  (Open-Platypus curated for LLM compression)")
print(f"✓ Model: GLM-4.6 MoE (357B params)")
print(f"✓ MoE handling: Shared experts + first layer + lm_head unquantized")
print(f"✓ Optimized for: NVIDIA Blackwell GPUs (SM 9.0+)")
print(f"✓ Multi-GPU setup: Model distributed across 2x RTX PRO 6000 (~194GB VRAM)")
print("="*70 + "\n")

# Apply NVFP4 quantization with sequential processing
# sequential_targets tells llmcompressor which module types to process one at a time
# This allows the model to use both GPUs without running out of memory
# Common patterns: MLP layers, Block layers, or specific model architecture layers
# 
# For GLM-4.6: May need to adjust based on model architecture printed above
# Try in order: ["GLMBlock"], ["GLMMLP"], or check printed module types

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    sequential_targets=["GLMBlock"],  # Adjust if OOM: try ["GLMMLP"] or other module types
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
print("Quantized using 2x RTX PRO 6000 Blackwell GPUs (~194GB VRAM)")
print("Ready for inference on Blackwell GPUs with vLLM! 🚀")
print("\nNote: For full W4A4 performance, inference requires Blackwell GPUs (SM 9.0+)")
print("      On non-Blackwell GPUs, vLLM will run weights-only quantization")
print("      (activations will remain unquantized for compatibility).")
print("\nMoE-specific optimizations:")
print("  - Shared experts preserved in FP16/BF16 for routing stability")
print("  - First layer and lm_head kept unquantized for quality")
print("  - Routed experts quantized to NVFP4 for maximum compression")

