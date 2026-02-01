#!/usr/bin/env python3
"""
NVFP4 Quantization with NVIDIA Model Optimizer (ModelOpt)

This script quantizes a Hugging Face model to NVFP4 format using NVIDIA's
TensorRT Model Optimizer. The resulting checkpoint is compatible with:
  - TensorRT-LLM (Blackwell builds)
  - vLLM with quantization="modelopt" or "modelopt_fp4"
  - Other ModelOpt-compatible inference runtimes

Usage:
    python quantize_nvfp4.py \
        --input_model meta-llama/Llama-3.3-70B-Instruct \
        --output_model ./Llama-3.3-70B-NVFP4 \
        --dataset_yaml Datasets/Dataset_Example.yaml

Requirements:
    pip install nvidia-modelopt transformers datasets accelerate pyyaml
"""

import argparse
import gc
import os
import random
import sys
from pathlib import Path
from typing import Callable, Iterator, Optional

import torch
import yaml
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Try to load dotenv for environment variable support
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# Import ModelOpt quantization module
try:
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint
except ImportError as e:
    print("ERROR: nvidia-modelopt is not installed.")
    print("Install with: pip install nvidia-modelopt")
    print(f"Import error: {e}")
    sys.exit(1)


# ============================================================================
# NVFP4 Configuration
# ============================================================================
# NVFP4 uses FP4 (E2M1) format with block-16 quantization for weights
# and FP16/FP8 for activations. This provides ~3.5x compression vs FP16.
#
# Key characteristics:
#   - Weights: FP4 with per-block-16 scaling (FP8 scale factors)
#   - Activations: FP16 or FP8 (depending on config)
#   - Second-level FP32 scaling for larger dynamic range
#
# Reference: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference
# ============================================================================

# NVFP4 quantization configuration
# FP4 uses (E2M1) format - 2 exponent bits, 1 mantissa bit
NVFP4_CONFIG = {
    "quant_cfg": {
        # Weight quantization: FP4 with block-16 quantization
        "*weight_quantizer": {
            "num_bits": (2, 1),  # FP4 = E2M1 format
            "block_sizes": {
                -1: 16,  # Block size of 16 along last dimension
                "type": "static",  # Static calibrated quantization
            },
            "enable": True,
        },
        # Input/activation quantization: Keep in higher precision
        # NVFP4 typically uses FP16 activations for accuracy
        "*input_quantizer": {
            "enable": False,  # Disable activation quantization for W4A16 mode
        },
        # Skip quantization for specific layers (optional)
        "*lm_head*": {"enable": False},
        "*embed*": {"enable": False},
    },
    "algorithm": {"method": "max"},  # Max calibration for scale factors
}

# Alternative: NVFP4 with FP8 activations (W4A8) - more aggressive
NVFP4_W4A8_CONFIG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": (2, 1),  # FP4 = E2M1
            "block_sizes": {-1: 16, "type": "static"},
            "enable": True,
        },
        "*input_quantizer": {
            "num_bits": (4, 3),  # FP8 = E4M3
            "block_sizes": {-1: None, "type": "dynamic"},  # Per-token dynamic
            "enable": True,
        },
        "*lm_head*": {"enable": False},
        "*embed*": {"enable": False},
    },
    "algorithm": {"method": "max"},
}


# ============================================================================
# Dataset Loading and Formatting
# ============================================================================

def format_raw_text(example: dict, columns: list[str]) -> str:
    """Extract raw text from specified columns."""
    for col in columns:
        if col in example and example[col]:
            return str(example[col])
    # Fallback to common column names
    for col in ["text", "content", "body"]:
        if col in example and example[col]:
            return str(example[col])
    return ""


def format_prompt_answer(example: dict, columns: list[str]) -> str:
    """Format prompt/answer pairs into conversation."""
    if len(columns) >= 2:
        prompt = example.get(columns[0], "")
        answer = example.get(columns[1], "")
        return f"Human: {prompt}\n\nAssistant: {answer}"
    return format_raw_text(example, columns)


def format_sharegpt(example: dict, columns: list[str]) -> str:
    """Format ShareGPT-style conversations."""
    conv_col = columns[0] if columns else "conversations"
    conversations = example.get(conv_col, [])
    
    if not conversations:
        return ""
    
    parts = []
    for turn in conversations:
        role = turn.get("from", turn.get("role", ""))
        content = turn.get("value", turn.get("content", ""))
        
        if role in ["human", "user"]:
            parts.append(f"Human: {content}")
        elif role in ["gpt", "assistant"]:
            parts.append(f"Assistant: {content}")
        elif role == "system":
            parts.append(f"System: {content}")
    
    return "\n\n".join(parts)


def format_chat_completion(example: dict, columns: list[str]) -> str:
    """Format OpenAI-style chat messages."""
    msg_col = columns[0] if columns else "messages"
    messages = example.get(msg_col, example.get("data", []))
    
    if not messages:
        return ""
    
    # Handle nested list format
    if messages and isinstance(messages[0], list):
        messages = messages[0]
    
    parts = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "user":
                parts.append(f"Human: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            elif role == "system":
                parts.append(f"System: {content}")
    
    return "\n\n".join(parts)


FORMATTERS = {
    "raw_text": format_raw_text,
    "prompt_answer": format_prompt_answer,
    "sharegpt": format_sharegpt,
    "chat_completion": format_chat_completion,
}


def load_calibration_dataset(
    yaml_path: str,
    tokenizer: AutoTokenizer,
    max_samples_override: Optional[int] = None,
    max_seq_len_override: Optional[int] = None,
) -> Dataset:
    """
    Load and prepare calibration dataset from YAML configuration.
    
    Args:
        yaml_path: Path to dataset configuration YAML
        tokenizer: Tokenizer for the model
        max_samples_override: Override max_samples from config
        max_seq_len_override: Override max_seq_len from config
    
    Returns:
        Dataset with tokenized samples ready for calibration
    """
    print(f"\n{'='*70}")
    print("Loading calibration dataset configuration...")
    print(f"{'='*70}")
    
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # Get calibration settings
    calib_cfg = cfg.get("calibration", {})
    max_samples = max_samples_override or calib_cfg.get("max_samples", 512)
    max_seq_len = max_seq_len_override or calib_cfg.get("max_seq_len", 2048)
    seed = calib_cfg.get("seed", 42)
    
    print(f"Max samples: {max_samples}")
    print(f"Max sequence length: {max_seq_len}")
    print(f"Random seed: {seed}")
    
    datasets_cfg = cfg.get("datasets", [])
    if not datasets_cfg:
        raise ValueError("No datasets specified in configuration")
    
    # Load and sample datasets
    all_samples = []
    total_weight = sum(d.get("weight", 1.0) for d in datasets_cfg)
    
    print(f"\nLoading {len(datasets_cfg)} datasets...")
    
    for ds_cfg in datasets_cfg:
        name = ds_cfg.get("name", ds_cfg["path"])
        path = ds_cfg["path"]
        split = ds_cfg.get("split", "train")
        weight = ds_cfg.get("weight", 1.0) / total_weight
        formatter_name = ds_cfg.get("formatter", "raw_text")
        columns = ds_cfg.get("columns", [])
        streaming = ds_cfg.get("streaming", False)
        
        # Calculate samples for this dataset
        n_samples = max(1, int(max_samples * weight))
        
        print(f"  - {name}: {n_samples} samples (weight={weight:.2%})")
        
        try:
            # Load dataset
            if streaming:
                ds = load_dataset(path, split=split, streaming=True)
                # Take samples from streaming dataset
                samples = []
                for i, item in enumerate(ds):
                    if i >= n_samples:
                        break
                    samples.append(item)
                ds = Dataset.from_list(samples)
            else:
                ds = load_dataset(path, split=split)
                # Shuffle and select
                ds = ds.shuffle(seed=seed)
                ds = ds.select(range(min(len(ds), n_samples)))
            
            # Get formatter
            formatter = FORMATTERS.get(formatter_name, format_raw_text)
            
            # Format text
            def format_example(example):
                text = formatter(example, columns)
                return {"text": text}
            
            ds = ds.map(format_example, remove_columns=ds.column_names)
            
            # Filter empty samples
            ds = ds.filter(lambda x: len(x["text"].strip()) > 0)
            
            all_samples.append(ds)
            
        except Exception as e:
            print(f"    WARNING: Failed to load {name}: {e}")
            continue
    
    if not all_samples:
        raise RuntimeError("No datasets were successfully loaded")
    
    # Concatenate all datasets
    combined = concatenate_datasets(all_samples)
    print(f"\nTotal samples loaded: {len(combined)}")
    
    # Tokenize
    print("Tokenizing samples...")
    
    def tokenize_fn(example):
        tokens = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_seq_len,
            padding=False,
            return_tensors=None,
        )
        return {"input_ids": tokens["input_ids"]}
    
    combined = combined.map(
        tokenize_fn,
        remove_columns=combined.column_names,
        desc="Tokenizing",
    )
    
    # Filter out very short sequences
    min_length = 32
    combined = combined.filter(lambda x: len(x["input_ids"]) >= min_length)
    
    print(f"Final dataset size: {len(combined)} samples")
    print(f"{'='*70}\n")
    
    return combined


def create_calibration_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
) -> Callable:
    """
    Create a forward loop function for ModelOpt calibration.
    
    Args:
        dataset: Tokenized calibration dataset
        batch_size: Batch size for calibration
    
    Returns:
        Forward loop function for mtq.quantize()
    """
    def forward_loop(model):
        """Forward loop for calibration."""
        device = next(model.parameters()).device
        
        for i, example in enumerate(dataset):
            input_ids = torch.tensor([example["input_ids"]], device=device)
            
            with torch.no_grad():
                try:
                    model(input_ids)
                except Exception as e:
                    print(f"Warning: Calibration step {i} failed: {e}")
                    continue
            
            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"  Calibration progress: {i + 1}/{len(dataset)} samples")
        
        print(f"  Calibration complete: {len(dataset)} samples processed")
    
    return forward_loop


# ============================================================================
# Main Quantization Flow
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NVFP4 Quantization with NVIDIA Model Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python quantize_nvfp4.py --input_model ./llama-70b --output_model ./llama-70b-nvfp4 --dataset_yaml datasets.yaml
  
  # With HuggingFace model
  python quantize_nvfp4.py --input_model meta-llama/Llama-3.3-70B-Instruct --output_model ./nvfp4 --dataset_yaml datasets.yaml
  
  # Override calibration settings
  python quantize_nvfp4.py --input_model ./model --output_model ./nvfp4 --dataset_yaml cfg.yaml --max_samples 1024 --max_seq_len 4096
        """,
    )
    
    # Required arguments
    parser.add_argument(
        "--input_model",
        required=True,
        help="Path or HuggingFace model ID of the FP16/BF16 model to quantize",
    )
    parser.add_argument(
        "--output_model",
        required=True,
        help="Output directory for the NVFP4 quantized checkpoint",
    )
    parser.add_argument(
        "--dataset_yaml",
        required=True,
        help="Path to YAML file specifying calibration datasets",
    )
    
    # Optional arguments
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Override max calibration samples from YAML config",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="Override max sequence length from YAML config",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for calibration (default: 1)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype for loading (default: auto)",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="cuda",
        help="Device map for model loading (default: cuda)",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code when loading model",
    )
    parser.add_argument(
        "--w4a8",
        action="store_true",
        help="Use W4A8 mode (FP4 weights + FP8 activations) instead of W4A16",
    )
    parser.add_argument(
        "--skip_layers",
        type=str,
        nargs="*",
        default=None,
        help="Additional layer patterns to skip quantization (e.g., 'gate' 'router')",
    )
    
    args = parser.parse_args()
    
    # ========================================================================
    # Setup
    # ========================================================================
    print("\n" + "=" * 70)
    print("NVFP4 Quantization with NVIDIA Model Optimizer")
    print("=" * 70)
    print(f"Input model:  {args.input_model}")
    print(f"Output model: {args.output_model}")
    print(f"Dataset YAML: {args.dataset_yaml}")
    print(f"Mode:         {'W4A8 (FP4 weights + FP8 activations)' if args.w4a8 else 'W4A16 (FP4 weights + FP16 activations)'}")
    print("=" * 70 + "\n")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Calibration will be very slow on CPU.")
    else:
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Reduce CUDA memory fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # ========================================================================
    # Load Tokenizer
    # ========================================================================
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.input_model,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
    
    # ========================================================================
    # Load Model
    # ========================================================================
    print("\nLoading model...")
    
    # Determine dtype
    torch_dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    
    model = AutoModelForCausalLM.from_pretrained(
        args.input_model,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    
    print(f"Model loaded: {model.config.model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    if torch.cuda.is_available():
        print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # ========================================================================
    # Prepare Calibration Data
    # ========================================================================
    print("\nPreparing calibration dataset...")
    calib_dataset = load_calibration_dataset(
        args.dataset_yaml,
        tokenizer,
        max_samples_override=args.max_samples,
        max_seq_len_override=args.max_seq_len,
    )
    
    forward_loop = create_calibration_dataloader(calib_dataset, args.batch_size)
    
    # ========================================================================
    # Configure Quantization
    # ========================================================================
    print("\nConfiguring NVFP4 quantization...")
    
    # Select configuration
    if args.w4a8:
        quant_config = NVFP4_W4A8_CONFIG.copy()
        print("Using W4A8 configuration (FP4 weights + FP8 activations)")
    else:
        quant_config = NVFP4_CONFIG.copy()
        print("Using W4A16 configuration (FP4 weights + FP16 activations)")
    
    # Add any additional skip layers
    if args.skip_layers:
        for pattern in args.skip_layers:
            quant_config["quant_cfg"][f"*{pattern}*"] = {"enable": False}
            print(f"  Skipping quantization for: *{pattern}*")
    
    # ========================================================================
    # Run Quantization
    # ========================================================================
    print("\n" + "=" * 70)
    print("Running calibration and quantization...")
    print("=" * 70 + "\n")
    
    # Clear cache before quantization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Run ModelOpt quantization
    model = mtq.quantize(model, quant_config, forward_loop)
    
    # Print quantization summary
    print("\nQuantization Summary:")
    mtq.print_quant_summary(model)
    
    # ========================================================================
    # Export Checkpoint
    # ========================================================================
    print("\n" + "=" * 70)
    print("Exporting NVFP4 checkpoint...")
    print("=" * 70 + "\n")
    
    output_path = Path(args.output_model)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Export using ModelOpt's HF checkpoint export
    with torch.inference_mode():
        export_hf_checkpoint(
            model,
            export_dir=str(output_path),
        )
    
    # Save tokenizer
    tokenizer.save_pretrained(output_path)
    
    # ========================================================================
    # Verify Output
    # ========================================================================
    print("\nVerifying output checkpoint...")
    
    required_files = ["config.json", "hf_quant_config.json"]
    for f in required_files:
        if (output_path / f).exists():
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} (missing)")
    
    # Check for model weights
    weight_files = list(output_path.glob("*.safetensors")) + list(output_path.glob("*.bin"))
    if weight_files:
        total_size = sum(f.stat().st_size for f in weight_files)
        print(f"  ✓ Model weights: {len(weight_files)} files, {total_size / 1e9:.2f} GB")
    else:
        print("  ✗ No model weight files found!")
    
    # ========================================================================
    # Done
    # ========================================================================
    print("\n" + "=" * 70)
    print("NVFP4 quantization complete!")
    print("=" * 70)
    print(f"\nOutput saved to: {output_path.absolute()}")
    print("\nTo use with vLLM:")
    print(f'  llm = LLM(model="{output_path}", quantization="modelopt")')
    print("\nTo serve with vLLM:")
    print(f'  vllm serve {output_path} --quantization modelopt')
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
