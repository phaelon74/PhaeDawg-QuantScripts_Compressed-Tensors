#!/bin/bash
# =============================================================================
# RunPod NVFP4 Quantization Script for Behemoth-R1-123B-v2
# =============================================================================
# 
# This script sets up the environment and quantizes TheDrummer/Behemoth-R1-123B-v2
# to NVFP4 format using NVIDIA ModelOpt on a 4x B200 RunPod instance.
#
# Usage:
#   chmod +x runpod_behemoth_nvfp4.sh
#   ./runpod_behemoth_nvfp4.sh
#
# Prerequisites:
#   - Set HF_TOKEN environment variable with your HuggingFace token
#   - Token must have write access to TheHouseOfTheDude organization
#
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

# Paths
WORKSPACE="/workspace"
VENV_NAME="Nvidia-ModelOPT"
VENV_PATH="${WORKSPACE}/${VENV_NAME}"
MODELS_DIR="${WORKSPACE}/models"

# Model configuration
INPUT_MODEL_REPO="TheDrummer/Behemoth-R1-123B-v2"
INPUT_MODEL_PATH="${MODELS_DIR}/Behemoth-R1-123B-v2"
OUTPUT_MODEL_PATH="${MODELS_DIR}/TheHouseOfTheDude/Behemoth-R1-V2_ModelOpt/NVFP4"
OUTPUT_HF_REPO="TheHouseOfTheDude/Behemoth-R1-V2_ModelOpt-NVFP4"

# Quantization settings
MAX_SAMPLES=512
MAX_SEQ_LEN=4096
OVERSAMPLE_RATIO=1.5

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_banner() {
    echo ""
    echo "============================================================================="
    echo "$1"
    echo "============================================================================="
    echo ""
}

# =============================================================================
# Step 0: Pre-flight Checks
# =============================================================================

print_banner "Step 0: Pre-flight Checks"

# Check HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    log_error "HF_TOKEN environment variable is not set!"
    echo ""
    echo "Please set your HuggingFace token before running this script:"
    echo "  export HF_TOKEN='hf_xxxxxxxxxxxxxxxxxxxx'"
    echo ""
    echo "The token must have write access to the TheHouseOfTheDude organization."
    exit 1
fi
log_success "HF_TOKEN is set"

# Check NVIDIA drivers
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found. NVIDIA drivers may not be installed."
    exit 1
fi

log_info "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
log_info "Detected ${GPU_COUNT} GPU(s)"

# Check disk space
AVAILABLE_SPACE=$(df -BG ${WORKSPACE} | tail -1 | awk '{print $4}' | tr -d 'G')
log_info "Available disk space: ${AVAILABLE_SPACE}GB"

if [ "$AVAILABLE_SPACE" -lt 500 ]; then
    log_warning "Low disk space! Recommended: 500GB+ for 123B model"
fi

# =============================================================================
# Step 1: Navigate to Workspace
# =============================================================================

print_banner "Step 1: Setting Up Workspace"

cd ${WORKSPACE}
log_success "Changed to workspace: $(pwd)"

# =============================================================================
# Step 2: Create Virtual Environment
# =============================================================================

print_banner "Step 2: Creating Virtual Environment"

if [ -d "${VENV_PATH}" ]; then
    log_warning "Virtual environment already exists at ${VENV_PATH}"
    read -p "Do you want to recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Removing existing virtual environment..."
        rm -rf "${VENV_PATH}"
    fi
fi

if [ ! -d "${VENV_PATH}" ]; then
    log_info "Creating virtual environment..."
    python3 -m venv "${VENV_PATH}"
    log_success "Virtual environment created at ${VENV_PATH}"
fi

# Activate virtual environment
log_info "Activating virtual environment..."
source "${VENV_PATH}/bin/activate"
log_success "Virtual environment activated: $(which python)"

# Upgrade pip
log_info "Upgrading pip..."
pip install --upgrade pip

# =============================================================================
# Step 3: Install Requirements
# =============================================================================

print_banner "Step 3: Installing Requirements"

log_info "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

log_info "Installing NVIDIA ModelOpt with HuggingFace extras..."
pip install "nvidia-modelopt[hf]"

log_info "Installing additional dependencies..."
pip install transformers datasets accelerate pyyaml python-dotenv huggingface_hub

log_info "Installing hf_transfer for faster downloads..."
pip install hf_transfer

# Verify installations
log_info "Verifying installations..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
python -c "import modelopt; print(f'ModelOpt: {modelopt.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

log_success "All requirements installed successfully"

# =============================================================================
# Step 4: Create Models Directory
# =============================================================================

print_banner "Step 4: Creating Models Directory"

mkdir -p "${MODELS_DIR}"
log_success "Models directory created: ${MODELS_DIR}"

# =============================================================================
# Step 5: Download Model from HuggingFace
# =============================================================================

print_banner "Step 5: Downloading Model from HuggingFace"

# Enable hf_transfer for faster downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

if [ -d "${INPUT_MODEL_PATH}" ] && [ -f "${INPUT_MODEL_PATH}/config.json" ]; then
    log_warning "Model already exists at ${INPUT_MODEL_PATH}"
    read -p "Do you want to re-download it? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Skipping download, using existing model"
    else
        rm -rf "${INPUT_MODEL_PATH}"
    fi
fi

if [ ! -d "${INPUT_MODEL_PATH}" ] || [ ! -f "${INPUT_MODEL_PATH}/config.json" ]; then
    log_info "Downloading ${INPUT_MODEL_REPO}..."
    log_info "This is a 123B model (~246GB), this will take a while..."
    
    python -c "
from huggingface_hub import snapshot_download
import os

snapshot_download(
    repo_id='${INPUT_MODEL_REPO}',
    local_dir='${INPUT_MODEL_PATH}',
    token=os.environ.get('HF_TOKEN'),
    resume_download=True,
    max_workers=8,
)
print('Download complete!')
"
    log_success "Model downloaded to ${INPUT_MODEL_PATH}"
fi

# Verify download
if [ -f "${INPUT_MODEL_PATH}/config.json" ]; then
    log_success "Model download verified"
else
    log_error "Model download failed - config.json not found"
    exit 1
fi

# =============================================================================
# Step 6: Create Output Directory
# =============================================================================

print_banner "Step 6: Creating Output Directory"

mkdir -p "${OUTPUT_MODEL_PATH}"
log_success "Output directory created: ${OUTPUT_MODEL_PATH}"

# =============================================================================
# Step 7: Create Dataset Configuration
# =============================================================================

print_banner "Step 7: Creating Dataset Configuration"

DATASET_YAML="${WORKSPACE}/calibration_dataset.yaml"

cat > "${DATASET_YAML}" << 'EOF'
# =============================================================================
# Calibration Dataset Configuration for Behemoth-R1-123B-v2 NVFP4 Quantization
# =============================================================================
# Balanced mix of creative writing, roleplay, and general conversation datasets
# optimized for the Behemoth model's creative writing strengths.
# =============================================================================

calibration:
  max_samples: 512           # Target samples for calibration
  max_seq_len: 4096          # Max sequence length (Behemoth supports long context)
  seed: 42                   # Random seed for reproducibility
  batch_size: 1              # Batch size for calibration
  min_token_length: 64       # Filter samples shorter than this
  oversample_ratio: 1.5      # Request 50% more to account for filtering

datasets:
  # ============================================================================
  # Creative Writing & Roleplay (Primary focus for Behemoth)
  # ============================================================================
  
  - name: opus_writingprompts
    path: Gryphe/Opus-WritingPrompts
    split: train
    weight: 1.5
    formatter: sharegpt
    columns: [conversations]

  - name: stheno_filtered
    path: anthracite-org/stheno-filtered-v1.1
    split: train
    weight: 1.5
    formatter: sharegpt
    columns: [conversations]

  - name: kalo_opus_instruct
    path: Gryphe/Kalo-Opus-Instruct-22k-no-refusal
    split: train
    weight: 1.0
    formatter: sharegpt
    columns: [conversations]

  # ============================================================================
  # General Instruction Following
  # ============================================================================
  
  - name: ultrachat
    path: HuggingFaceH4/ultrachat_200k
    split: train_sft
    weight: 1.0
    formatter: chat_completion
    columns: [messages]

  - name: open_assistant
    path: OpenAssistant/oasst1
    split: train
    weight: 0.5
    formatter: chat_completion
    columns: [messages]

  # ============================================================================
  # Reasoning & Long-form Content (for R1 reasoning capabilities)
  # ============================================================================
  
  - name: openorca
    path: Open-Orca/OpenOrca
    split: train
    weight: 0.5
    formatter: prompt_answer
    columns: [question, response]
EOF

log_success "Dataset configuration created: ${DATASET_YAML}"
cat "${DATASET_YAML}"

# =============================================================================
# Step 8: Create Quantization Script
# =============================================================================

print_banner "Step 8: Creating Quantization Script"

QUANT_SCRIPT="${WORKSPACE}/quantize_nvfp4.py"

cat > "${QUANT_SCRIPT}" << 'PYTHON_EOF'
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
            if skip_layers:
                for pattern in skip_layers:
                    config["quant_cfg"][f"*{pattern}*"] = {"enable": False}
            return config
    
    print("Using custom NVFP4 config (dynamic block quantization)")
    
    config = {
        "quant_cfg": {
            "*weight_quantizer": {
                "num_bits": (2, 1),
                "block_sizes": {
                    "-1": 16,
                    "type": "dynamic",
                },
                "enable": True,
            },
            "*input_quantizer": {
                "enable": False,
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
# Dataset Loading and Formatting
# ============================================================================

def format_raw_text(example: dict, columns: list[str]) -> str:
    for col in columns:
        if col in example and example[col]:
            return str(example[col])
    for col in ["text", "content", "body"]:
        if col in example and example[col]:
            return str(example[col])
    return ""


def format_prompt_answer(example: dict, columns: list[str]) -> str:
    if len(columns) >= 2:
        prompt = example.get(columns[0], "")
        answer = example.get(columns[1], "")
        return f"Human: {prompt}\n\nAssistant: {answer}"
    return format_raw_text(example, columns)


def format_sharegpt(example: dict, columns: list[str]) -> str:
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
    msg_col = columns[0] if columns else "messages"
    messages = example.get(msg_col, example.get("data", []))
    
    if not messages:
        return ""
    
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
    """Load and prepare calibration dataset from YAML configuration."""
    print(f"\n{'='*70}")
    print("Loading calibration dataset configuration...")
    print(f"{'='*70}")
    
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    calib_cfg = cfg.get("calibration", {})
    max_samples = max_samples_override or calib_cfg.get("max_samples", 512)
    max_seq_len = max_seq_len_override or calib_cfg.get("max_seq_len", 2048)
    seed = calib_cfg.get("seed", 42)
    min_token_length = calib_cfg.get("min_token_length", 32)
    oversample_ratio = oversample_ratio_override or calib_cfg.get("oversample_ratio", 1.5)
    
    oversample_target = int(max_samples * oversample_ratio)
    
    print(f"Target samples: {max_samples}")
    print(f"Oversample ratio: {oversample_ratio}x (requesting {oversample_target} samples)")
    print(f"Max sequence length: {max_seq_len}")
    print(f"Min token length filter: {min_token_length}")
    print(f"Random seed: {seed}")
    
    datasets_cfg = cfg.get("datasets", [])
    if not datasets_cfg:
        raise ValueError("No datasets specified in configuration")
    
    all_samples = []
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
        
        n_samples = max(1, int(oversample_target * weight))
        
        print(f"  - {name}: {n_samples} samples (weight={weight:.2%})")
        
        try:
            if streaming:
                ds = load_dataset(path, split=split, streaming=True)
                samples = []
                for i, item in enumerate(ds):
                    if i >= n_samples:
                        break
                    samples.append(item)
                ds = Dataset.from_list(samples)
            else:
                ds = load_dataset(path, split=split)
                ds = ds.shuffle(seed=seed)
                ds = ds.select(range(min(len(ds), n_samples)))
            
            formatter = FORMATTERS.get(formatter_name, format_raw_text)
            
            def format_example(example):
                text = formatter(example, columns)
                return {"text": text}
            
            ds = ds.map(format_example, remove_columns=ds.column_names)
            ds = ds.filter(lambda x: len(x["text"].strip()) > 0)
            
            all_samples.append(ds)
            
        except Exception as e:
            print(f"    WARNING: Failed to load {name}: {e}")
            continue
    
    if not all_samples:
        raise RuntimeError("No datasets were successfully loaded")
    
    combined = concatenate_datasets(all_samples)
    print(f"\nTotal samples loaded (before filtering): {len(combined)}")
    
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
    
    combined = combined.filter(lambda x: len(x["input_ids"]) >= min_token_length)
    
    samples_after_filter = len(combined)
    print(f"Samples after filtering (>={min_token_length} tokens): {samples_after_filter}")
    
    if samples_after_filter >= max_samples:
        combined = combined.shuffle(seed=seed)
        combined = combined.select(range(max_samples))
        print(f"Trimmed to target: {max_samples} samples")
    elif samples_after_filter < max_samples:
        shortfall = max_samples - samples_after_filter
        shortfall_pct = (shortfall / max_samples) * 100
        print(f"WARNING: Only {samples_after_filter} samples available "
              f"(target: {max_samples}, shortfall: {shortfall} / {shortfall_pct:.1f}%)")
    
    print(f"\nFinal calibration dataset: {len(combined)} samples")
    print(f"{'='*70}\n")
    
    return combined


def create_calibration_dataloader(dataset: Dataset, batch_size: int = 1) -> Callable:
    """Create a forward loop function for ModelOpt calibration."""
    def forward_loop(model):
        device = next(model.parameters()).device
        
        for i, example in enumerate(dataset):
            input_ids = torch.tensor([example["input_ids"]], device=device)
            
            with torch.no_grad():
                try:
                    model(input_ids)
                except Exception as e:
                    print(f"Warning: Calibration step {i} failed: {e}")
                    continue
            
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
    )
    
    parser.add_argument("--input_model", required=True, help="Path or HuggingFace model ID")
    parser.add_argument("--output_model", required=True, help="Output directory for NVFP4 checkpoint")
    parser.add_argument("--dataset_yaml", required=True, help="Path to YAML dataset config")
    parser.add_argument("--max_samples", type=int, default=None, help="Override max calibration samples")
    parser.add_argument("--max_seq_len", type=int, default=None, help="Override max sequence length")
    parser.add_argument("--oversample_ratio", type=float, default=None, help="Override oversample ratio")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for calibration")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", type=str, default="auto", help="Device map for model loading")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code")
    parser.add_argument("--skip_layers", type=str, nargs="*", default=None, help="Layer patterns to skip")
    parser.add_argument("--list_configs", action="store_true", help="List available configs and exit")
    
    args = parser.parse_args()
    
    if args.list_configs:
        print("\nAvailable ModelOpt Quantization Configs:")
        print("=" * 50)
        configs = list_available_quant_configs()
        for cfg in sorted(configs):
            print(f"  - mtq.{cfg}")
        sys.exit(0)
    
    print("\n" + "=" * 70)
    print("NVFP4 Quantization with NVIDIA Model Optimizer")
    print("=" * 70)
    print(f"Input model:  {args.input_model}")
    print(f"Output model: {args.output_model}")
    print(f"Dataset YAML: {args.dataset_yaml}")
    print("=" * 70 + "\n")
    
    if torch.cuda.is_available():
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.input_model,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
    
    print("\nLoading model...")
    
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
    
    if hasattr(model, "hf_device_map"):
        device_counts = {}
        for layer, device in model.hf_device_map.items():
            device_str = str(device)
            device_counts[device_str] = device_counts.get(device_str, 0) + 1
        print("\nDevice distribution:")
        for device, count in sorted(device_counts.items()):
            print(f"  {device}: {count} layers")
    
    print("\nPreparing calibration dataset...")
    calib_dataset = load_calibration_dataset(
        args.dataset_yaml,
        tokenizer,
        max_samples_override=args.max_samples,
        max_seq_len_override=args.max_seq_len,
        oversample_ratio_override=args.oversample_ratio,
    )
    
    forward_loop = create_calibration_dataloader(calib_dataset, args.batch_size)
    
    print("\nConfiguring NVFP4 quantization...")
    available_configs = list_available_quant_configs()
    print(f"Available ModelOpt configs: {available_configs}")
    
    quant_config = get_nvfp4_config(skip_layers=args.skip_layers)
    
    print("\n" + "=" * 70)
    print("Running calibration and quantization...")
    print("=" * 70 + "\n")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    model = mtq.quantize(model, quant_config, forward_loop)
    
    print("\nQuantization Summary:")
    mtq.print_quant_summary(model)
    
    print("\n" + "=" * 70)
    print("Exporting NVFP4 checkpoint...")
    print("=" * 70 + "\n")
    
    output_path = Path(args.output_model)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with torch.inference_mode():
        export_hf_checkpoint(
            model,
            export_dir=str(output_path),
        )
    
    tokenizer.save_pretrained(output_path)
    
    print("\nVerifying output checkpoint...")
    
    required_files = ["config.json", "hf_quant_config.json"]
    for f in required_files:
        if (output_path / f).exists():
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} (missing)")
    
    weight_files = list(output_path.glob("*.safetensors")) + list(output_path.glob("*.bin"))
    if weight_files:
        total_size = sum(f.stat().st_size for f in weight_files)
        print(f"  ✓ Model weights: {len(weight_files)} files, {total_size / 1e9:.2f} GB")
    
    print("\n" + "=" * 70)
    print("NVFP4 quantization complete!")
    print("=" * 70)
    print(f"\nOutput saved to: {output_path.absolute()}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
PYTHON_EOF

chmod +x "${QUANT_SCRIPT}"
log_success "Quantization script created: ${QUANT_SCRIPT}"

# =============================================================================
# Step 9: Run Quantization
# =============================================================================

print_banner "Step 9: Running NVFP4 Quantization"

log_info "Starting quantization of ${INPUT_MODEL_REPO}..."
log_info "This will take several hours for a 123B model on 4x B200s..."
log_info ""
log_info "Configuration:"
log_info "  Input:  ${INPUT_MODEL_PATH}"
log_info "  Output: ${OUTPUT_MODEL_PATH}"
log_info "  Max Samples: ${MAX_SAMPLES}"
log_info "  Max Seq Len: ${MAX_SEQ_LEN}"
log_info "  Oversample Ratio: ${OVERSAMPLE_RATIO}"
log_info ""

# Run quantization
python "${QUANT_SCRIPT}" \
    --input_model "${INPUT_MODEL_PATH}" \
    --output_model "${OUTPUT_MODEL_PATH}" \
    --dataset_yaml "${DATASET_YAML}" \
    --max_samples ${MAX_SAMPLES} \
    --max_seq_len ${MAX_SEQ_LEN} \
    --oversample_ratio ${OVERSAMPLE_RATIO} \
    --device_map auto \
    --trust_remote_code

if [ $? -eq 0 ]; then
    log_success "Quantization completed successfully!"
else
    log_error "Quantization failed!"
    exit 1
fi

# =============================================================================
# Step 10: Upload to HuggingFace
# =============================================================================

print_banner "Step 10: Uploading to HuggingFace"

log_info "Uploading to ${OUTPUT_HF_REPO}..."

# Create model card
MODEL_CARD="${OUTPUT_MODEL_PATH}/README.md"
cat > "${MODEL_CARD}" << EOF
---
license: other
base_model: TheDrummer/Behemoth-R1-123B-v2
tags:
  - nvfp4
  - modelopt
  - quantized
  - blackwell
  - b200
library_name: transformers
---

# Behemoth-R1-V2 ModelOpt NVFP4

NVFP4 quantized version of [TheDrummer/Behemoth-R1-123B-v2](https://huggingface.co/TheDrummer/Behemoth-R1-123B-v2) using NVIDIA Model Optimizer.

## Quantization Details

| Property | Value |
|----------|-------|
| **Original Model** | TheDrummer/Behemoth-R1-123B-v2 |
| **Quantization** | NVFP4 (FP4 weights, FP16 activations) |
| **Method** | NVIDIA ModelOpt PTQ |
| **Calibration Samples** | ${MAX_SAMPLES} |
| **Max Sequence Length** | ${MAX_SEQ_LEN} |

## Hardware Requirements

- **Optimal**: NVIDIA Blackwell GPUs (B100, B200, RTX PRO 6000 Blackwell)
- **Compatible**: Hopper/Ampere (will use weight-only mode)

## Usage with vLLM

\`\`\`python
from vllm import LLM, SamplingParams

llm = LLM(
    model="TheHouseOfTheDude/Behemoth-R1-V2_ModelOpt-NVFP4",
    quantization="modelopt",
    trust_remote_code=True,
)

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)
outputs = llm.generate(["Write a story about..."], sampling_params)
print(outputs[0].outputs[0].text)
\`\`\`

## Chat Template

Uses Mistral v7 (Non-Tekken) format. See the original model card for usage details.

## Credits

- Original Model: [TheDrummer](https://huggingface.co/TheDrummer)
- Quantization: TheHouseOfTheDude
- Quantization Framework: NVIDIA ModelOpt
EOF

log_success "Model card created"

# Upload to HuggingFace
python << EOF
from huggingface_hub import HfApi, create_repo
import os

api = HfApi()
token = os.environ.get("HF_TOKEN")
repo_id = "${OUTPUT_HF_REPO}"
local_path = "${OUTPUT_MODEL_PATH}"

print(f"Creating/checking repository: {repo_id}")

try:
    # Create repo if it doesn't exist
    create_repo(
        repo_id=repo_id,
        token=token,
        private=False,
        exist_ok=True,
        repo_type="model",
    )
    print(f"Repository ready: {repo_id}")
except Exception as e:
    print(f"Note: {e}")

print(f"Uploading model from {local_path}...")
print("This may take a while for large models...")

api.upload_folder(
    folder_path=local_path,
    repo_id=repo_id,
    token=token,
    commit_message="Upload NVFP4 quantized Behemoth-R1-123B-v2",
)

print(f"\n✓ Upload complete!")
print(f"Model available at: https://huggingface.co/{repo_id}")
EOF

if [ $? -eq 0 ]; then
    log_success "Upload completed successfully!"
else
    log_error "Upload failed!"
    exit 1
fi

# =============================================================================
# Done!
# =============================================================================

print_banner "All Done!"

log_success "NVFP4 quantization and upload complete!"
echo ""
echo "Summary:"
echo "  Input Model:  ${INPUT_MODEL_REPO}"
echo "  Output Model: ${OUTPUT_HF_REPO}"
echo "  Local Path:   ${OUTPUT_MODEL_PATH}"
echo ""
echo "Model URL: https://huggingface.co/${OUTPUT_HF_REPO}"
echo ""
echo "To use with vLLM:"
echo "  from vllm import LLM"
echo "  llm = LLM(model='${OUTPUT_HF_REPO}', quantization='modelopt')"
echo ""
