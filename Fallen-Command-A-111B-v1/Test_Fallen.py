#!/usr/bin/env python3
"""
Modern AWQ Quantization using llm-compressor
The official successor to AutoAWQ, maintained by vLLM project

This script uses llm-compressor for AWQ quantization of large models:
- Simple AWQModifier targeting Linear layers
- Lets llm-compressor handle layer-by-layer processing automatically
- NO manual device management - trust the library!
- Works with large models (111B) on consumer GPUs (24GB)
- W4A16 symmetric quantization with group_size=128
"""

import os
import sys
import json
import time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import datasets
from datetime import datetime
from pathlib import Path

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
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
        # Simple AWQ recipe - let llm-compressor handle layer-by-layer processing automatically
        # This is the CORRECT approach - trust llm-compressor to manage memory
        recipe = AWQModifier(
            targets=["Linear"],
            config_groups={
                "group_0": QuantizationScheme(
                    targets=["Linear"],
                    weights=QuantizationArgs(num_bits=4, group_size=128, symmetric=True),
                )
            },
            ignore=["lm_head"],
        )

        # Load model simply - let llm-compressor's oneshot() handle layer loading
        # NO device_map, NO manual offloading - trust the library!
        log_message(f"Loading model {MODEL_PATH} with torch_dtype='auto'...")
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        log_message("✅ Model and tokenizer loaded.")

        # Prepare calibration dataset
        log_message("Preparing calibration dataset: wikitext-2-raw-v1 [train][:256] ...")
        ds = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:256]")
        ds = ds.shuffle(seed=42)

        def _preprocess(example):
            return {"text": example["text"]}

        ds = ds.map(_preprocess)

        log_message("Starting end-to-end quantization...")
        log_message("Quantization config: 256 samples, max_seq_length=512, W4A16 symmetric group_size=128")
        
        # Let llm-compressor handle everything - it will load layers sequentially as needed
        oneshot(
            model=model,
            dataset=ds,
            recipe=recipe,
            max_seq_length=256,
            num_calibration_samples=128,
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
        "note": "Generated with modern quantization tools (llm-compressor) - AWQ"
    }
    
    info_path = os.path.join(OUTPUT_DIR, "quantization_info.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    log_message(f"✅ Info saved: {info_path}")

def main():
    """Main function with AWQ quantization"""
    print("🔄 Modern AWQ Quantization Script")
    print("Using llm-compressor (AutoAWQ successor) - simplified approach")
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
    
    # Try llm-compressor AWQ quantization with proper observer handling
    if check_llm_compressor():
        log_message("🔧 Starting llm-compressor AWQ quantization...")
        if quantize_with_llm_compressor():
            create_info_file("llm-compressor (AWQ W4A16)")
            log_message("🎉 Quantization Complete!")
        else:
            log_message("❌ Quantization Failed.")
    else:
        log_message("Could not run quantization.")

if __name__ == "__main__":
    main()
