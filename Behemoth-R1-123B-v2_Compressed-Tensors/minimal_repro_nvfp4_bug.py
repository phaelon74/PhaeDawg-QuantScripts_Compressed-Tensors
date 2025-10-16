#!/usr/bin/env python3
"""
Minimal reproducible example for NVFP4 OOM bug on large models.

This script demonstrates the CUDA OOM error that occurs during NVFP4 initialization
for models > 100B parameters on 24GB GPUs.

Expected: Sequential processing should keep VRAM usage under 24GB
Actual: OOM during update_fused_layer_weight_global_scales() before calibration starts

GitHub Issue: [To be created]
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Configuration
MODEL_ID = "mistralai/Mistral-Large-Instruct-2411"  # 123B model - triggers bug
# MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"  # 8B model - works fine
NUM_CALIBRATION_SAMPLES = 20  # Small number to minimize time to failure
MAX_SEQUENCE_LENGTH = 2048

# Force single GPU and reduce memory fragmentation
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("="*80)
print("NVFP4 OOM Bug Reproduction Script")
print("="*80)
print(f"Model: {MODEL_ID}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA'}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"Calibration samples: {NUM_CALIBRATION_SAMPLES}")
print("="*80)

# Clear any existing GPU allocations
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Load model to CPU (device_map=None is critical for sequential processing)
print("\n[1/5] Loading model to CPU...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map=None,  # Load to CPU, not GPU
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
print(f"✓ Model loaded to CPU")

# Prepare minimal calibration dataset
print("\n[2/5] Loading calibration dataset...")
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]")
print(f"✓ Loaded {len(ds)} samples")

print("\n[3/5] Preprocessing dataset...")
def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }

ds = ds.map(preprocess)

def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )

ds = ds.map(tokenize, remove_columns=ds.column_names)
print(f"✓ Dataset preprocessed")

# Configure NVFP4 quantization
print("\n[4/5] Configuring NVFP4 quantization...")
recipe = QuantizationModifier(
    targets="Linear", 
    scheme="NVFP4",  # W4A4 with calibration
    ignore=["lm_head"]
)
print(f"✓ Recipe configured: NVFP4 (W4A4)")

# Attempt quantization - THIS WILL FAIL
print("\n[5/5] Running oneshot quantization...")
print("NOTE: This will fail with OOM during initialization (before calibration starts)")
print("-" * 80)

try:
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        sequential_targets=["MistralMLP"],  # This doesn't help - OOM happens before this
    )
    print("\n✓ SUCCESS: Quantization completed (unexpected!)")
    
except torch.cuda.OutOfMemoryError as e:
    print("\n❌ FAILED: CUDA Out of Memory")
    print("-" * 80)
    print("Error occurred during initialization, before calibration started.")
    print("This is the bug - update_fused_layer_weight_global_scales() loads")
    print("all MLP layers onto GPU simultaneously instead of sequentially.")
    print("-" * 80)
    print(f"\nError details:\n{e}")
    
except Exception as e:
    print(f"\n❌ FAILED: Unexpected error")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("Bug Reproduction Complete")
print("="*80)

