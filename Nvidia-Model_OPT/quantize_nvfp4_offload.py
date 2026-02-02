#!/usr/bin/env python3
"""
NVFP4 Quantization with NVIDIA Model Optimizer (ModelOpt) - CPU Offload Version

This script quantizes a Hugging Face model to NVFP4 format using NVIDIA's
TensorRT Model Optimizer with CPU/disk offloading support for large models
that don't fit in GPU memory.

Key Features:
  - CPU offloading for models larger than available VRAM
  - Disk offloading for extremely large models
  - Automatic memory management with accelerate
  - Compatible with multi-GPU setups

Usage:
    python quantize_nvfp4_offload.py \
        --input_model meta-llama/Llama-3.1-405B-Instruct \
        --output_model ./Llama-405B-NVFP4 \
        --dataset_yaml Datasets/Dataset_Example.yaml \
        --offload_folder ./offload_cache

Memory Requirements (approximate):
    - 8B model:   ~20 GB VRAM (no offload needed)
    - 70B model:  ~160 GB (offload reduces to ~40 GB VRAM)
    - 123B model: ~280 GB (offload reduces to ~60-80 GB VRAM)
    - 405B model: ~900 GB (offload reduces to ~80-100 GB VRAM)

Requirements:
    pip install nvidia-modelopt transformers datasets accelerate pyyaml
"""

import argparse
import gc
import os
import random
import shutil
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

# Import accelerate for offloading
try:
    from accelerate import infer_auto_device_map, init_empty_weights
    from accelerate.utils import get_balanced_memory
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    print("WARNING: accelerate not installed. Offloading may not work optimally.")
    print("Install with: pip install accelerate")


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
#   - IMPORTANT: FP4 format requires DYNAMIC block quantization
#
# Reference: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference
# ============================================================================


def get_nvfp4_config(skip_layers: list[str] = None):
    """
    Get NVFP4 quantization config, trying predefined configs first.
    
    ModelOpt FP4 requires dynamic block quantization (not static).
    """
    # Try to use predefined NVFP4 config if available
    predefined_configs = [
        "NVFP4_DEFAULT_CFG",
        "FP4_DEFAULT_CFG",
        "NVFP4_AWQ_CFG",
        "FP4_AWQ_CFG",
    ]
    
    for config_name in predefined_configs:
        if hasattr(mtq, config_name):
            print(f"Using predefined config: mtq.{config_name}")
            config = getattr(mtq, config_name).copy()
            # Add skip layers if specified
            if skip_layers:
                for pattern in skip_layers:
                    config["quant_cfg"][f"*{pattern}*"] = {"enable": False}
            return config
    
    # Fall back to custom config
    # FP4 (E2M1) requires DYNAMIC block quantization per ModelOpt validation
    print("Using custom NVFP4 config (dynamic block quantization)")
    
    config = {
        "quant_cfg": {
            # Weight quantization: FP4 with dynamic block-16 quantization
            "*weight_quantizer": {
                "num_bits": (2, 1),  # FP4 = E2M1 format
                "block_sizes": {
                    "-1": 16,  # Block size of 16 along last dimension (string key)
                    "type": "dynamic",  # FP4 requires dynamic quantization
                },
                "enable": True,
            },
            # Input/activation quantization: Keep in higher precision (W4A16)
            "*input_quantizer": {
                "enable": False,
            },
            # Skip quantization for specific layers
            "*lm_head*": {"enable": False},
            "*embed*": {"enable": False},
        },
        "algorithm": {"method": "max"},
    }
    
    # Add additional skip layers
    if skip_layers:
        for pattern in skip_layers:
            config["quant_cfg"][f"*{pattern}*"] = {"enable": False}
    
    return config


def get_nvfp4_w4a8_config(skip_layers: list[str] = None):
    """
    Get NVFP4 W4A8 config (FP4 weights + FP8 activations).
    
    More aggressive quantization for maximum performance.
    """
    config = {
        "quant_cfg": {
            "*weight_quantizer": {
                "num_bits": (2, 1),  # FP4 = E2M1
                "block_sizes": {"-1": 16, "type": "dynamic"},
                "enable": True,
            },
            "*input_quantizer": {
                "num_bits": (4, 3),  # FP8 = E4M3
                "block_sizes": {"-1": None, "type": "dynamic"},  # Per-token dynamic
                "enable": True,
            },
            "*lm_head*": {"enable": False},
            "*embed*": {"enable": False},
        },
        "algorithm": {"method": "max"},
    }
    
    if skip_layers:
        for pattern in skip_layers:
            config["quant_cfg"][f"*{pattern}*"] = {"enable": False}
    
    return config


def list_available_quant_configs():
    """List all available quantization configs in ModelOpt."""
    configs = []
    for name in dir(mtq):
        if name.endswith("_CFG") and name.isupper():
            configs.append(name)
    return configs


# ============================================================================
# Memory Management Utilities
# ============================================================================

def get_memory_info():
    """Get current memory usage information."""
    info = {}
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            total = props.total_memory
            
            info[f"cuda:{i}"] = {
                "name": props.name,
                "total_gb": total / 1e9,
                "allocated_gb": allocated / 1e9,
                "reserved_gb": reserved / 1e9,
                "free_gb": (total - reserved) / 1e9,
            }
    
    # CPU memory
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["cpu"] = {
            "total_gb": mem.total / 1e9,
            "available_gb": mem.available / 1e9,
            "used_gb": mem.used / 1e9,
            "percent": mem.percent,
        }
    except ImportError:
        pass
    
    return info


def print_memory_status(prefix=""):
    """Print current memory status."""
    info = get_memory_info()
    
    if prefix:
        print(f"\n{prefix}")
    
    for device, stats in info.items():
        if device.startswith("cuda"):
            print(f"  {device} ({stats['name']}): "
                  f"{stats['allocated_gb']:.1f} / {stats['total_gb']:.1f} GB "
                  f"(free: {stats['free_gb']:.1f} GB)")
        elif device == "cpu":
            print(f"  CPU RAM: {stats['used_gb']:.1f} / {stats['total_gb']:.1f} GB "
                  f"(available: {stats['available_gb']:.1f} GB, {stats['percent']:.0f}% used)")


def estimate_model_memory(num_params: int, dtype: torch.dtype = torch.bfloat16) -> float:
    """Estimate memory required for a model in GB."""
    bytes_per_param = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
    }.get(dtype, 2)
    
    return (num_params * bytes_per_param) / 1e9


def clear_memory():
    """Aggressively clear GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


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
    oversample_ratio_override: Optional[float] = None,
) -> Dataset:
    """
    Load and prepare calibration dataset from YAML configuration.
    
    Uses oversampling to ensure we get the target number of samples after
    filtering. Samples are requested at `max_samples * oversample_ratio`,
    then trimmed back to `max_samples` after filtering.
    
    Args:
        yaml_path: Path to dataset configuration YAML
        tokenizer: Tokenizer for the model
        max_samples_override: Override max_samples from config
        max_seq_len_override: Override max_seq_len from config
        oversample_ratio_override: Override oversample_ratio from config
    
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
    min_token_length = calib_cfg.get("min_token_length", 32)
    
    # Oversample ratio: request more samples to account for filtering losses
    # Default 1.5 means request 50% more samples than target
    oversample_ratio = oversample_ratio_override or calib_cfg.get("oversample_ratio", 1.5)
    
    # Calculate oversampled target
    oversample_target = int(max_samples * oversample_ratio)
    
    print(f"Target samples: {max_samples}")
    print(f"Oversample ratio: {oversample_ratio}x (requesting {oversample_target} samples)")
    print(f"Max sequence length: {max_seq_len}")
    print(f"Min token length filter: {min_token_length}")
    print(f"Random seed: {seed}")
    
    datasets_cfg = cfg.get("datasets", [])
    if not datasets_cfg:
        raise ValueError("No datasets specified in configuration")
    
    # Load and sample datasets with oversampling
    all_samples = []
    dataset_weights = []  # Track weights for proportional trimming later
    total_weight = sum(d.get("weight", 1.0) for d in datasets_cfg)
    
    print(f"\nLoading {len(datasets_cfg)} datasets (with {oversample_ratio}x oversampling)...")
    
    for ds_cfg in datasets_cfg:
        name = ds_cfg.get("name", ds_cfg["path"])
        path = ds_cfg["path"]
        split = ds_cfg.get("split", "train")
        weight = ds_cfg.get("weight", 1.0) / total_weight
        formatter_name = ds_cfg.get("formatter", "raw_text")
        columns = ds_cfg.get("columns", [])
        streaming = ds_cfg.get("streaming", False)
        
        # Calculate samples for this dataset using OVERSAMPLED target
        n_samples = max(1, int(oversample_target * weight))
        
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
    print(f"\nTotal samples loaded (before filtering): {len(combined)}")
    
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
    combined = combined.filter(lambda x: len(x["input_ids"]) >= min_token_length)
    
    samples_after_filter = len(combined)
    print(f"Samples after filtering (>={min_token_length} tokens): {samples_after_filter}")
    
    # Trim to exact target if we have enough samples
    if samples_after_filter >= max_samples:
        # Shuffle and take exactly max_samples to maintain randomness
        combined = combined.shuffle(seed=seed)
        combined = combined.select(range(max_samples))
        print(f"Trimmed to target: {max_samples} samples")
    elif samples_after_filter < max_samples:
        # We don't have enough samples - warn but continue
        shortfall = max_samples - samples_after_filter
        shortfall_pct = (shortfall / max_samples) * 100
        print(f"WARNING: Only {samples_after_filter} samples available "
              f"(target: {max_samples}, shortfall: {shortfall} / {shortfall_pct:.1f}%)")
        print(f"         Consider increasing oversample_ratio (current: {oversample_ratio}x) "
              f"or adding more datasets")
    
    print(f"\nFinal calibration dataset: {len(combined)} samples")
    print(f"{'='*70}\n")
    
    return combined


def create_calibration_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
) -> Callable:
    """
    Create a forward loop function for ModelOpt calibration.
    
    For offloaded models, we need to be careful about device placement.
    The model handles moving data to the correct device internally.
    
    Args:
        dataset: Tokenized calibration dataset
        batch_size: Batch size for calibration
    
    Returns:
        Forward loop function for mtq.quantize()
    """
    def forward_loop(model):
        """Forward loop for calibration with offloading support."""
        # Get the device of the first parameter to determine where to place inputs
        # For offloaded models, this might be the first layer on GPU
        first_param = next(model.parameters())
        device = first_param.device
        
        # For models with device_map, the model handles device placement
        # We just need to provide inputs on the correct initial device
        
        for i, example in enumerate(dataset):
            input_ids = torch.tensor([example["input_ids"]], device=device)
            
            with torch.no_grad():
                try:
                    model(input_ids)
                except Exception as e:
                    # For offloaded models, try with CPU and let the model handle it
                    try:
                        input_ids_cpu = torch.tensor([example["input_ids"]])
                        model(input_ids_cpu)
                    except Exception as e2:
                        print(f"Warning: Calibration step {i} failed: {e2}")
                        continue
            
            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"  Calibration progress: {i + 1}/{len(dataset)} samples")
                # Print memory status periodically for offloaded models
                if (i + 1) % 100 == 0:
                    print_memory_status("  Memory status:")
            
            # Periodic memory cleanup for large models
            if (i + 1) % 100 == 0:
                clear_memory()
        
        print(f"  Calibration complete: {len(dataset)} samples processed")
    
    return forward_loop


# ============================================================================
# Main Quantization Flow with Offloading
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NVFP4 Quantization with NVIDIA Model Optimizer (CPU Offload Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with offloading
  python quantize_nvfp4_offload.py \\
      --input_model ./llama-123b \\
      --output_model ./llama-123b-nvfp4 \\
      --dataset_yaml datasets.yaml \\
      --offload_folder ./offload_cache
  
  # Large model with aggressive offloading
  python quantize_nvfp4_offload.py \\
      --input_model meta-llama/Llama-3.1-405B-Instruct \\
      --output_model ./llama-405b-nvfp4 \\
      --dataset_yaml datasets.yaml \\
      --offload_folder /fast_ssd/offload \\
      --max_gpu_memory 80GiB
  
  # Multi-GPU with offloading
  python quantize_nvfp4_offload.py \\
      --input_model ./model \\
      --output_model ./nvfp4 \\
      --dataset_yaml cfg.yaml \\
      --offload_folder ./offload \\
      --device_map auto

Memory Notes:
  - Offload folder should be on fast storage (SSD/NVMe)
  - Expect ~50-100 GB disk usage for 123B+ models
  - Use --max_gpu_memory to limit GPU usage and force offloading
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
    parser.add_argument(
        "--offload_folder",
        required=True,
        help="Directory for CPU/disk offloading (should be on fast SSD)",
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
        "--oversample_ratio",
        type=float,
        default=None,
        help="Override oversample ratio from YAML config (default: 1.5). "
             "Request this many times more samples to account for filtering losses.",
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
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype for loading (default: bfloat16). "
             "bfloat16 recommended for stability during offloading.",
    )
    parser.add_argument(
        "--max_gpu_memory",
        type=str,
        default=None,
        help="Maximum GPU memory to use per GPU (e.g., '80GiB', '40GB'). "
             "Lower values force more offloading to CPU.",
    )
    parser.add_argument(
        "--max_cpu_memory",
        type=str,
        default=None,
        help="Maximum CPU memory to use (e.g., '200GiB'). "
             "Useful to leave headroom for other processes.",
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
    parser.add_argument(
        "--list_configs",
        action="store_true",
        help="List available ModelOpt quantization configs and exit",
    )
    parser.add_argument(
        "--clean_offload",
        action="store_true",
        help="Remove offload folder after successful quantization",
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        default=True,
        help="Load model with low CPU memory usage (default: True)",
    )
    
    args = parser.parse_args()
    
    # ========================================================================
    # List Configs Mode
    # ========================================================================
    if args.list_configs:
        print("\nAvailable ModelOpt Quantization Configs:")
        print("=" * 50)
        configs = list_available_quant_configs()
        for cfg in sorted(configs):
            print(f"  - mtq.{cfg}")
        print("=" * 50)
        print("\nTo use a predefined config, ModelOpt will auto-detect:")
        print("  NVFP4_DEFAULT_CFG, FP4_DEFAULT_CFG, NVFP4_AWQ_CFG, etc.")
        print("\nIf none found, a custom dynamic FP4 config will be used.")
        sys.exit(0)
    
    # ========================================================================
    # Setup
    # ========================================================================
    print("\n" + "=" * 70)
    print("NVFP4 Quantization with CPU Offloading")
    print("=" * 70)
    print(f"Input model:     {args.input_model}")
    print(f"Output model:    {args.output_model}")
    print(f"Dataset YAML:    {args.dataset_yaml}")
    print(f"Offload folder:  {args.offload_folder}")
    print(f"Mode:            {'W4A8 (FP4 weights + FP8 activations)' if args.w4a8 else 'W4A16 (FP4 weights + FP16 activations)'}")
    if args.max_gpu_memory:
        print(f"Max GPU memory:  {args.max_gpu_memory}")
    if args.max_cpu_memory:
        print(f"Max CPU memory:  {args.max_cpu_memory}")
    print("=" * 70 + "\n")
    
    # Create offload folder
    offload_path = Path(args.offload_folder)
    offload_path.mkdir(parents=True, exist_ok=True)
    print(f"Offload folder created: {offload_path.absolute()}")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. This will be VERY slow.")
    else:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"CUDA Device {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
    
    print_memory_status("Initial memory status:")
    
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
    # Load Model with Offloading
    # ========================================================================
    print("\nLoading model with CPU/disk offloading...")
    print("This may take a while for large models...")
    
    # Determine dtype
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    
    # Build memory configuration for device_map="auto"
    max_memory = {}
    
    if args.max_gpu_memory:
        # Parse the memory string (e.g., "80GiB" or "80GB")
        for i in range(torch.cuda.device_count()):
            max_memory[i] = args.max_gpu_memory
    
    if args.max_cpu_memory:
        max_memory["cpu"] = args.max_cpu_memory
    
    # If no memory limits specified, let accelerate figure it out
    if not max_memory:
        max_memory = None
    
    print(f"Loading with device_map='auto' and offloading to: {offload_path}")
    if max_memory:
        print(f"Memory limits: {max_memory}")
    
    # Load model with offloading
    model = AutoModelForCausalLM.from_pretrained(
        args.input_model,
        torch_dtype=torch_dtype,
        device_map="auto",
        offload_folder=str(offload_path),
        offload_state_dict=True,  # Offload state dict during loading
        max_memory=max_memory,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
    )
    model.eval()
    
    # Print device map
    print(f"\nModel loaded: {model.config.model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    # Show device distribution
    if hasattr(model, "hf_device_map"):
        device_map = model.hf_device_map
        device_counts = {}
        for layer, device in device_map.items():
            device_str = str(device)
            device_counts[device_str] = device_counts.get(device_str, 0) + 1
        
        print("\nDevice distribution:")
        for device, count in sorted(device_counts.items()):
            print(f"  {device}: {count} layers")
    
    print_memory_status("Memory after model loading:")
    
    # ========================================================================
    # Prepare Calibration Data
    # ========================================================================
    print("\nPreparing calibration dataset...")
    calib_dataset = load_calibration_dataset(
        args.dataset_yaml,
        tokenizer,
        max_samples_override=args.max_samples,
        max_seq_len_override=args.max_seq_len,
        oversample_ratio_override=args.oversample_ratio,
    )
    
    forward_loop = create_calibration_dataloader(calib_dataset, args.batch_size)
    
    # ========================================================================
    # Configure Quantization
    # ========================================================================
    print("\nConfiguring NVFP4 quantization...")
    
    # List available configs for debugging
    available_configs = list_available_quant_configs()
    print(f"Available ModelOpt configs: {available_configs}")
    
    # Select configuration
    if args.w4a8:
        quant_config = get_nvfp4_w4a8_config(skip_layers=args.skip_layers)
        print("Using W4A8 configuration (FP4 weights + FP8 activations)")
    else:
        quant_config = get_nvfp4_config(skip_layers=args.skip_layers)
        print("Using W4A16 configuration (FP4 weights + FP16 activations)")
    
    # Log skip layers
    if args.skip_layers:
        for pattern in args.skip_layers:
            print(f"  Skipping quantization for: *{pattern}*")
    
    # ========================================================================
    # Run Quantization
    # ========================================================================
    print("\n" + "=" * 70)
    print("Running calibration and quantization...")
    print("This may take a long time for large offloaded models.")
    print("=" * 70 + "\n")
    
    # Clear cache before quantization
    clear_memory()
    print_memory_status("Memory before quantization:")
    
    # Run ModelOpt quantization
    model = mtq.quantize(model, quant_config, forward_loop)
    
    # Print quantization summary
    print("\nQuantization Summary:")
    mtq.print_quant_summary(model)
    
    print_memory_status("Memory after quantization:")
    
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
    # Cleanup
    # ========================================================================
    if args.clean_offload and offload_path.exists():
        print(f"\nCleaning up offload folder: {offload_path}")
        try:
            shutil.rmtree(offload_path)
            print("  ✓ Offload folder removed")
        except Exception as e:
            print(f"  ✗ Failed to remove offload folder: {e}")
    else:
        # Report offload folder size
        if offload_path.exists():
            offload_size = sum(f.stat().st_size for f in offload_path.rglob("*") if f.is_file())
            print(f"\nOffload folder size: {offload_size / 1e9:.2f} GB")
            print(f"  Use --clean_offload to remove after quantization")
            print(f"  Or manually: rm -rf {offload_path}")
    
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
    
    print_memory_status("Final memory status:")


if __name__ == "__main__":
    main()
