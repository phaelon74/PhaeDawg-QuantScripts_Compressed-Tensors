#!/usr/bin/env python3
"""
NVFP4 Quantization - NVIDIA Style

This script replicates NVIDIA's exact PTQ approach for NVFP4 quantization
using the same methods as their official hf_ptq.py script.

Key differences from basic PTQ:
1. Uses NVIDIA's exact calibration datasets (cnn_dailymail)
2. Uses FP8 KV cache quantization
3. Uses left padding for calibration
4. Uses their exact export method

For best quality (near-BF16 accuracy), you should use QAT or QAD instead.
See: https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/llm_qat

Usage:
    python quantize_nvfp4_nvidia_style.py \
        --pyt_ckpt_path meta-llama/Llama-3.1-8B-Instruct \
        --export_path ./Llama-3.1-8B-NVFP4 \
        --qformat nvfp4

Requirements:
    pip install nvidia-modelopt>=0.35.0 transformers datasets accelerate
"""

import argparse
import random
import time
import warnings
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check ModelOpt version
try:
    import modelopt
    import modelopt.torch.opt as mto
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint
    from modelopt.torch.quantization.config import need_calibration
    
    modelopt_version = getattr(modelopt, "__version__", "unknown")
    print(f"ModelOpt version: {modelopt_version}")
except ImportError as e:
    print("ERROR: nvidia-modelopt is not installed.")
    print("Install with: pip install nvidia-modelopt>=0.35.0")
    raise e

# Quantization configs (same as NVIDIA's hf_ptq.py)
QUANT_CFG_CHOICES: dict[str, dict[str, Any]] = {
    "fp8": mtq.FP8_DEFAULT_CFG,
    "nvfp4": mtq.NVFP4_DEFAULT_CFG,
    "nvfp4_awq": getattr(mtq, "NVFP4_AWQ_LITE_CFG", mtq.NVFP4_DEFAULT_CFG),
}

KV_QUANT_CFG_CHOICES = {
    "none": None,
    "fp8": "FP8_KV_CFG",
}

RAND_SEED = 1234

mto.enable_huggingface_checkpointing()


def get_tokenizer(ckpt_path, trust_remote_code=False):
    """Get tokenizer with proper padding setup."""
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path, trust_remote_code=trust_remote_code
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # NVIDIA uses left padding for calibration
    tokenizer.padding_side = "left"
    
    return tokenizer


def get_dataset_dataloader(tokenizer, batch_size=1, num_samples=512, max_seq_len=512):
    """
    Create calibration dataloader using NVIDIA's preferred datasets.
    
    NVIDIA uses cnn_dailymail for their official Llama-3.1-8B-Instruct-NVFP4 checkpoint.
    """
    from datasets import load_dataset
    from torch.utils.data import DataLoader, Dataset
    
    print(f"Loading calibration dataset: cnn_dailymail")
    print(f"  Samples: {num_samples}")
    print(f"  Max sequence length: {max_seq_len}")
    
    # Load cnn_dailymail - NVIDIA's calibration dataset
    ds = load_dataset("cnn_dailymail", "3.0.0", split="train", streaming=True)
    
    # Collect samples
    samples = []
    for i, item in enumerate(ds):
        if i >= num_samples * 2:  # Get extra samples for filtering
            break
        text = item.get("article", "") + " " + item.get("highlights", "")
        if len(text.strip()) > 100:  # Filter very short samples
            samples.append(text)
    
    print(f"  Loaded {len(samples)} samples")
    
    # Tokenize
    tokenized_samples = []
    for text in samples[:num_samples]:
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        tokenized_samples.append({
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
        })
    
    class CalibDataset(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    def collate_fn(batch):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        }
    
    dataloader = DataLoader(
        CalibDataset(tokenized_samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    return dataloader


def create_forward_loop(dataloader):
    """Create forward loop for calibration."""
    def forward_loop(model):
        device = next(model.parameters()).device
        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            
            if (i + 1) % 50 == 0:
                print(f"  Calibration progress: {i + 1}/{len(dataloader)}")
        
        print(f"  Calibration complete: {len(dataloader)} batches processed")
    
    return forward_loop


def main():
    parser = argparse.ArgumentParser(description="NVFP4 Quantization - NVIDIA Style")
    parser.add_argument("--pyt_ckpt_path", required=True, help="Model path or HuggingFace ID")
    parser.add_argument("--export_path", default="exported_model", help="Output directory")
    parser.add_argument("--qformat", default="nvfp4", choices=list(QUANT_CFG_CHOICES.keys()))
    parser.add_argument("--kv_cache_qformat", default="fp8", choices=list(KV_QUANT_CFG_CHOICES.keys()))
    parser.add_argument("--calib_size", type=int, default=512, help="Calibration samples")
    parser.add_argument("--calib_seq", type=int, default=512, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--trust_remote_code", action="store_true")
    
    args = parser.parse_args()
    
    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)
    torch.manual_seed(RAND_SEED)
    
    print("\n" + "=" * 70)
    print("NVFP4 Quantization - NVIDIA Style")
    print("=" * 70)
    print(f"Input model:  {args.pyt_ckpt_path}")
    print(f"Output:       {args.export_path}")
    print(f"Format:       {args.qformat}")
    print(f"KV Cache:     {args.kv_cache_qformat}")
    print("=" * 70 + "\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = get_tokenizer(args.pyt_ckpt_path, args.trust_remote_code)
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.pyt_ckpt_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    
    print(f"Model loaded: {model.config.model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    # Prepare calibration data
    print("\nPreparing calibration data...")
    calib_dataloader = get_dataset_dataloader(
        tokenizer,
        batch_size=args.batch_size,
        num_samples=args.calib_size,
        max_seq_len=args.calib_seq,
    )
    
    # Get quantization config
    print("\nConfiguring quantization...")
    quant_cfg = QUANT_CFG_CHOICES[args.qformat].copy()
    
    # Add KV cache quantization if specified (NVIDIA uses FP8 KV by default)
    if args.kv_cache_qformat != "none":
        kv_cfg_name = KV_QUANT_CFG_CHOICES[args.kv_cache_qformat]
        if hasattr(mtq, kv_cfg_name):
            kv_cache_cfg = getattr(mtq, kv_cfg_name)["quant_cfg"]
            quant_cfg = mtq.update_quant_cfg_with_kv_cache_quant(quant_cfg, kv_cache_cfg)
            print(f"  KV cache quantization: {args.kv_cache_qformat}")
    
    # Print config details
    print(f"\nQuantization config: {args.qformat}")
    weight_cfg = quant_cfg.get("quant_cfg", {}).get("*weight_quantizer", {})
    input_cfg = quant_cfg.get("quant_cfg", {}).get("*input_quantizer", {})
    print(f"  Weight: num_bits={weight_cfg.get('num_bits')}, block_sizes={weight_cfg.get('block_sizes')}")
    print(f"  Input:  num_bits={input_cfg.get('num_bits')}, block_sizes={input_cfg.get('block_sizes')}")
    print(f"  Algorithm: {quant_cfg.get('algorithm')}")
    
    # Create forward loop for calibration
    forward_loop = create_forward_loop(calib_dataloader)
    
    # Run quantization
    print("\n" + "=" * 70)
    print("Running calibration and quantization...")
    print("=" * 70 + "\n")
    
    torch.cuda.empty_cache()
    
    model = mtq.quantize(model, quant_cfg, forward_loop)
    
    # Print summary
    print("\nQuantization Summary:")
    mtq.print_quant_summary(model)
    
    # Export checkpoint
    print("\n" + "=" * 70)
    print("Exporting checkpoint...")
    print("=" * 70 + "\n")
    
    from pathlib import Path
    export_path = Path(args.export_path)
    export_path.mkdir(parents=True, exist_ok=True)
    
    with torch.inference_mode():
        export_hf_checkpoint(model, export_dir=str(export_path))
    
    tokenizer.save_pretrained(export_path)
    
    print(f"\nCheckpoint saved to: {export_path.absolute()}")
    print("\nTo use with vLLM:")
    print(f'  vllm serve {export_path} --quantization modelopt')
    print("\n" + "=" * 70)
    print("IMPORTANT: For best quality (near-BF16 accuracy), use QAT or QAD instead!")
    print("See: https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/llm_qat")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
