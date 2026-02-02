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
    import modelopt
    
    # Check ModelOpt version
    modelopt_version = getattr(modelopt, "__version__", "unknown")
    print(f"ModelOpt version: {modelopt_version}")
    
    # Warn if version is older than NVIDIA's recommended
    try:
        from packaging import version
        if modelopt_version != "unknown":
            current = version.parse(modelopt_version.split('+')[0].split('.dev')[0])
            recommended = version.parse("0.35.0")
            if current < recommended:
                print(f"WARNING: ModelOpt version {modelopt_version} may be outdated.")
                print(f"         NVIDIA uses v0.35.0+ for their official NVFP4 checkpoints.")
                print(f"         Consider upgrading: pip install --upgrade nvidia-modelopt")
    except Exception:
        pass  # packaging might not be installed
        
except ImportError as e:
    print("ERROR: nvidia-modelopt is not installed.")
    print("Install with: pip install nvidia-modelopt")
    print(f"Import error: {e}")
    sys.exit(1)


# ============================================================================
# NVFP4 Configuration
# ============================================================================
# NVFP4 uses FP4 (E2M1) format with block-16 quantization for BOTH weights
# and activations. This provides ~4x compression vs FP16.
#
# Key characteristics (from NVIDIA's official config):
#   - Weights: FP4 (E2M1) with per-block-16 scaling, FP8 (E4M3) scale factors
#   - Activations: FP4 (E2M1) with per-block-16 dynamic quantization
#   - scale_bits: (4, 3) = FP8 E4M3 format for scale factors
#   - IMPORTANT: Both weights AND activations must be quantized for quality
#
# Reference: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference
# ============================================================================

# Default layers that should NOT be quantized (from NVIDIA's ModelOpt)
_default_disabled_quantizer_cfg = {
    "nn.BatchNorm1d": {"*": {"enable": False}},
    "nn.BatchNorm2d": {"*": {"enable": False}},
    "nn.BatchNorm3d": {"*": {"enable": False}},
    "nn.LeakyReLU": {"*": {"enable": False}},
    "*lm_head*": {"enable": False},
    "*proj_out.*": {"enable": False},  # In Whisper model, lm_head has key name proj_out
    "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
    "*router*": {"enable": False},  # Skip the MOE router
    "*mlp.gate.*": {"enable": False},  # Skip the MOE router
    "*mlp.shared_expert_gate.*": {"enable": False},  # Skip the MOE router
    "*linear_attn.conv1d*": {"enable": False},
    "*mixer.conv1d*": {"enable": False},
    "*output_layer*": {"enable": False},
    "output.*": {"enable": False},
    "default": {"enable": False},
}


def get_nvfp4_config(skip_layers: list[str] = None):
    """
    Get NVFP4 quantization config (W4A4 - FP4 weights + FP4 activations).
    
    This matches NVIDIA's official NVFP4_DEFAULT_CFG exactly.
    Both weights AND activations are quantized to FP4 for best quality.
    """
    # Try to use predefined NVFP4 config if available (preferred)
    predefined_configs = [
        "NVFP4_DEFAULT_CFG",
        "NVFP4_AWQ_LITE_CFG",
    ]
    
    for config_name in predefined_configs:
        if hasattr(mtq, config_name):
            print(f"Using predefined config: mtq.{config_name}")
            import copy
            config = copy.deepcopy(getattr(mtq, config_name))
            # Add skip layers if specified
            if skip_layers:
                for pattern in skip_layers:
                    config["quant_cfg"][f"*{pattern}*"] = {"enable": False}
            return config
    
    # Fall back to custom config that EXACTLY matches NVIDIA's NVFP4_DEFAULT_CFG
    print("Using custom NVFP4 config (matching NVIDIA's NVFP4_DEFAULT_CFG)")
    
    config = {
        "quant_cfg": {
            # Weight quantization: FP4 with dynamic block-16 quantization + FP8 scale factors
            "*weight_quantizer": {
                "num_bits": (2, 1),  # FP4 = E2M1 format
                "block_sizes": {
                    -1: 16,  # INTEGER key (not string!) - block size 16 along last dimension
                    "type": "dynamic",  # FP4 requires dynamic quantization
                    "scale_bits": (4, 3),  # CRITICAL: FP8 E4M3 format for scale factors
                },
                "axis": None,  # Required for block quantization
                "enable": True,
            },
            # Input/activation quantization: ALSO FP4 (W4A4 configuration)
            # This is CRITICAL for NVFP4 quality - both weights AND activations must be quantized
            "*input_quantizer": {
                "num_bits": (2, 1),  # FP4 = E2M1 format
                "block_sizes": {
                    -1: 16,  # INTEGER key
                    "type": "dynamic",
                    "scale_bits": (4, 3),  # FP8 E4M3 for scale factors
                },
                "axis": None,
                "enable": True,
            },
            # Include all default disabled layers
            **_default_disabled_quantizer_cfg,
        },
        "algorithm": "max",  # Simple string, not dict
    }
    
    # Add additional skip layers
    if skip_layers:
        for pattern in skip_layers:
            config["quant_cfg"][f"*{pattern}*"] = {"enable": False}
    
    return config


def get_nvfp4_w4a8_config(skip_layers: list[str] = None):
    """
    Get NVFP4 W4A8 config (FP4 weights + FP8 activations).
    
    This matches NVIDIA's W4A8_NVFP4_FP8_CFG - uses FP4 for weights, FP8 for activations.
    Use this for slightly better accuracy at the cost of larger activation memory.
    """
    # Try to use predefined config if available
    if hasattr(mtq, "W4A8_NVFP4_FP8_CFG"):
        print("Using predefined config: mtq.W4A8_NVFP4_FP8_CFG")
        import copy
        config = copy.deepcopy(mtq.W4A8_NVFP4_FP8_CFG)
        if skip_layers:
            for pattern in skip_layers:
                config["quant_cfg"][f"*{pattern}*"] = {"enable": False}
        return config
    
    config = {
        "quant_cfg": {
            "*weight_quantizer": {
                "num_bits": (2, 1),  # FP4 = E2M1
                "block_sizes": {
                    -1: 32,  # Note: W4A8 uses block size 32, not 16
                    "type": "dynamic",
                    "scale_bits": (4, 3),  # FP8 E4M3 scale factors
                },
                "axis": None,
                "enable": True,
            },
            "*input_quantizer": {
                "num_bits": (4, 3),  # FP8 = E4M3 (not FP4)
                "axis": None,
                "enable": True,
            },
            **_default_disabled_quantizer_cfg,
        },
        "algorithm": "max",
    }
    
    if skip_layers:
        for pattern in skip_layers:
            config["quant_cfg"][f"*{pattern}*"] = {"enable": False}
    
    return config


def get_nvfp4_awq_config(skip_layers: list[str] = None):
    """
    Get NVFP4 with AWQ optimization (NVFP4_AWQ_LITE_CFG).
    
    AWQ (Activation-aware Weight Quantization) provides better accuracy
    by finding optimal scaling factors that minimize quantization error.
    This is recommended for best quality NVFP4 quantization.
    """
    # Try to use predefined config if available
    if hasattr(mtq, "NVFP4_AWQ_LITE_CFG"):
        print("Using predefined config: mtq.NVFP4_AWQ_LITE_CFG")
        import copy
        config = copy.deepcopy(mtq.NVFP4_AWQ_LITE_CFG)
        if skip_layers:
            for pattern in skip_layers:
                config["quant_cfg"][f"*{pattern}*"] = {"enable": False}
        return config
    
    # Fallback: same as NVFP4_DEFAULT but with AWQ algorithm
    config = {
        "quant_cfg": {
            "*weight_quantizer": {
                "num_bits": (2, 1),
                "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                "axis": None,
                "enable": True,
            },
            "*input_quantizer": {
                "num_bits": (2, 1),
                "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                "axis": None,
                "enable": True,
            },
            **_default_disabled_quantizer_cfg,
        },
        "algorithm": "awq_lite",  # Use AWQ for better accuracy
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
        "--mode",
        type=str,
        default="nvfp4",
        choices=["nvfp4", "nvfp4_awq", "w4a8"],
        help=(
            "Quantization mode:\n"
            "  nvfp4     - NVFP4 W4A4 (FP4 weights + FP4 activations) - matches NVIDIA's default\n"
            "  nvfp4_awq - NVFP4 with AWQ optimization - best quality, recommended\n"
            "  w4a8      - W4A8 (FP4 weights + FP8 activations) - slightly better accuracy"
        ),
    )
    parser.add_argument(
        "--w4a8",
        action="store_true",
        help="DEPRECATED: Use --mode w4a8 instead. Kept for backward compatibility.",
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
    print("NVFP4 Quantization with NVIDIA Model Optimizer")
    print("=" * 70)
    print(f"Input model:  {args.input_model}")
    print(f"Output model: {args.output_model}")
    print(f"Dataset YAML: {args.dataset_yaml}")
    # Determine mode for display
    display_mode = args.mode if not args.w4a8 else "w4a8"
    mode_descriptions = {
        "nvfp4": "W4A4 (FP4 weights + FP4 activations) - NVIDIA default",
        "nvfp4_awq": "W4A4 + AWQ (FP4 weights + FP4 activations + AWQ optimization)",
        "w4a8": "W4A8 (FP4 weights + FP8 activations)",
    }
    print(f"Mode:         {mode_descriptions.get(display_mode, display_mode)}")
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
    
    # Handle backward compatibility with --w4a8 flag
    mode = args.mode
    if args.w4a8:
        print("WARNING: --w4a8 is deprecated. Use --mode w4a8 instead.")
        mode = "w4a8"
    
    # Select configuration based on mode
    if mode == "w4a8":
        quant_config = get_nvfp4_w4a8_config(skip_layers=args.skip_layers)
        print("Using W4A8 configuration (FP4 weights + FP8 activations)")
    elif mode == "nvfp4_awq":
        quant_config = get_nvfp4_awq_config(skip_layers=args.skip_layers)
        print("Using NVFP4 + AWQ configuration (FP4 weights + FP4 activations + AWQ optimization)")
        print("  NOTE: AWQ calibration takes longer but provides better accuracy")
    else:  # nvfp4 (default)
        quant_config = get_nvfp4_config(skip_layers=args.skip_layers)
        print("Using NVFP4 W4A4 configuration (FP4 weights + FP4 activations)")
    
    # DIAGNOSTIC: Print the actual config being used
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: Actual quantization config being applied:")
    print("=" * 70)
    import json
    # Convert config to JSON-serializable format for printing
    def config_to_str(cfg):
        """Convert config to string, handling non-serializable types."""
        result = {}
        for k, v in cfg.items():
            if isinstance(v, dict):
                result[k] = config_to_str(v)
            else:
                result[str(k)] = str(v)
        return result
    
    print(f"Algorithm: {quant_config.get('algorithm', 'NOT SET')}")
    print("\nWeight quantizer config:")
    weight_cfg = quant_config.get("quant_cfg", {}).get("*weight_quantizer", {})
    print(f"  num_bits: {weight_cfg.get('num_bits', 'NOT SET')}")
    print(f"  block_sizes: {weight_cfg.get('block_sizes', 'NOT SET')}")
    print(f"  axis: {weight_cfg.get('axis', 'NOT SET')}")
    print(f"  enable: {weight_cfg.get('enable', 'NOT SET')}")
    
    print("\nInput quantizer config:")
    input_cfg = quant_config.get("quant_cfg", {}).get("*input_quantizer", {})
    print(f"  num_bits: {input_cfg.get('num_bits', 'NOT SET')}")
    print(f"  block_sizes: {input_cfg.get('block_sizes', 'NOT SET')}")
    print(f"  axis: {input_cfg.get('axis', 'NOT SET')}")
    print(f"  enable: {input_cfg.get('enable', 'NOT SET')}")
    print("=" * 70 + "\n")
        print("  This matches NVIDIA's official NVFP4_DEFAULT_CFG")
    
    # Log skip layers
    if args.skip_layers:
        for pattern in args.skip_layers:
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
