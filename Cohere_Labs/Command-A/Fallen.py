#!/usr/bin/env python3
"""
Modern AWQ Quantization using llm-compressor
The official successor to AutoAWQ, maintained by vLLM project

This script uses llm-compressor for AWQ quantization, which is:
- Actively maintained by the vLLM team
- Better compatibility with modern PyTorch/Transformers
- More stable and reliable quantization
"""

import os
import sys
import json
import time
import torch
import datasets
from datetime import datetime
from pathlib import Path

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.awq.mappings import AWQMapping
from transformers import AutoModelForCausalLM, AutoTokenizer
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

# === Model Configuration ===
MODEL_PATH = "/media/models/TheDrummer/Fallen-Command-A-111B-v1/main"
OUTPUT_DIR = "/media/models/TheHouseOfTheDude/Fallen-Command-A-111B-v1_Compresses-Tensors/W4A16"

def log_message(message: str):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def check_llm_compressor():
    """Check if llm-compressor is available"""
    try:
        import llmcompressor
        log_message(f"✅ llm-compressor version: {llmcompressor.__version__}")
        return True
    except ImportError:
        log_message("❌ llm-compressor not found. Install with:")
        log_message("   pip install llm-compressor")
        return False

def quantize_with_llm_compressor():
    """Quantize using llm-compressor (modern AutoAWQ successor)"""
    log_message("Starting quantization with llm-compressor...")
    
    try:
        # Configure explicit symmetric W4A16 via config_groups
        recipe = [
            AWQModifier(
                ignore=["lm_head"],
                mappings=[
                    AWQMapping(
                        "re:.*input_layernorm$",
                        [
                            "re:.*self_attn.q_proj$",
                            "re:.*self_attn.k_proj$",
                            "re:.*self_attn.v_proj$",
                            "re:.*mlp.gate_proj$",
                            "re:.*mlp.up_proj$",
                        ],
                    ),
                    AWQMapping("re:.*v_proj$", ["re:.*o_proj$"]),
                    AWQMapping("re:.*up_proj$", ["re:.*down_proj$"]),
                ],
                config_groups={
                    "group_0": QuantizationScheme(
                        targets=["Linear"],
                        weights=QuantizationArgs(
                            num_bits=4,
                            group_size=128,
                            symmetric=True,
                        ),
                    )
                },
            )
        ]

        # Load the model and tokenizer with multi-GPU support and H100 optimizations
        log_message(f"Loading model {MODEL_PATH} with torch_dtype='auto' and trust_remote_code...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        log_message("✅ Model and tokenizer loaded across available GPUs.")

        # === Dynamic Monkey-Patch for Multi-GPU Compatibility ===
        # The RMSNorm layer's forward pass causes device mismatches with device_map="auto".
        # We dynamically get the class from the loaded model and patch its forward method
        # to ensure the input tensor is on the correct device. This is more robust
        # than a static import and resilient to library updates.
        try:
            # Access the class from an actual layer instance
            rms_norm_class = model.model.layers[0].input_layernorm.__class__

            # Check if the patch is already applied to avoid recursion
            if not hasattr(rms_norm_class, '_original_forward'):
                _original_rmsnorm_forward = rms_norm_class.forward
                
                def _patched_rmsnorm_forward(self, hidden_states):
                    """Patched forward method to ensure tensor device alignment."""
                    hidden_states = hidden_states.to(self.weight.device)
                    return _original_rmsnorm_forward(self, hidden_states)

                rms_norm_class.forward = _patched_rmsnorm_forward
                rms_norm_class._original_forward = _original_rmsnorm_forward # Mark as patched
                log_message("✅ Dynamically patched RMSNorm layer for multi-GPU compatibility.")
            else:
                log_message("✅ RMSNorm layer already patched.")
        except Exception as e:
            log_message(f"⚠️  Could not apply RMSNorm patch: {e}")
        # ===============================================

        # Let the oneshot function handle the entire quantization and saving process.
        # It does not support sharding directly, so we will save a single file first.
        # Prepare calibration dataset (Ultrachat subset) per v0.7.x examples
        log_message("Preparing calibration dataset: HuggingFaceH4/ultrachat_200k [train_sft][:256] ...")
        ds = datasets.load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:256]")
        ds = ds.shuffle(seed=42)

        def _preprocess(example):
            return {
                "text": tokenizer.apply_chat_template(
                    example["messages"],
                    tokenize=False,
                )
            }

        ds = ds.map(_preprocess)

        log_message("Starting end-to-end quantization...")
        oneshot(
            model=model,
            dataset=ds,
            recipe=recipe,
            max_seq_length=512,
            num_calibration_samples=256,
        )
        
        # Save compressed model and tokenizer
        model.save_pretrained(OUTPUT_DIR, save_compressed=True)
        tokenizer.save_pretrained(OUTPUT_DIR)
        log_message(f"✅ Quantization complete. Model saved to {OUTPUT_DIR}")

        return True
        
    except BaseException as e:
        log_message(f"❌ llm-compressor quantization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_info_file(method_used):
    """Create info file for the quantization method used"""
    info = {
        "quantization_method": method_used,
        "source_model": MODEL_PATH,
        "quantized_at": datetime.now().isoformat(),
        "bits": 4,
        "group_size": 128,
        "note": "Generated with modern quantization tools (llm-compressor)"
    }
    
    info_path = os.path.join(OUTPUT_DIR, "quantization_info.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    log_message(f"✅ Info saved: {info_path}")

def main():
    """Main function with multiple quantization approaches"""
    print("🔄 Modern AWQ Quantization Script")
    print("Using llm-compressor (AutoAWQ successor)")
    print("=" * 50)
	
    # === Fix for [Errno 28] No space left on device ===
    # Set a new cache directory on the persistent volume to avoid filling up the container disk.
    # This must be done before any 'datasets' operations.
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
    os.makedirs(cache_dir, exist_ok=True)
    datasets.config.HF_DATASETS_CACHE = cache_dir
    log_message(f"Hugging Face datasets cache redirected to: {cache_dir}")
    # ===================================================
    
    # Check CUDA
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        log_message(f"✅ CUDA available with {gpu_count} GPU(s)")
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            log_message(f"   GPU {i}: {name} ({memory:.1f}GB)")
    else:
        log_message("⚠️  No CUDA available")
        return
    
    # Validate model path
    if not os.path.exists(MODEL_PATH):
        log_message(f"❌ Model path not found: {MODEL_PATH}")
        return
    
    # Try llm-compressor first (modern AutoAWQ)
    if check_llm_compressor():
        log_message("🔧 Starting llm-compressor AWQ quantization...")
        if quantize_with_llm_compressor():
            create_info_file("llm-compressor (AWQ/GPTQ W4A16)")
            log_message("🎉 Quantization Complete!")
        else:
            log_message("❌ Quantization Failed.")
    else:
        log_message("Could not run quantization.")

if __name__ == "__main__":
    main()
