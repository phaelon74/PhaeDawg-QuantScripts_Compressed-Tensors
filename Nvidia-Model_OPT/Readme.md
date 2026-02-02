# NVFP4 Quantization with NVIDIA Model Optimizer

This repository provides a minimal, reproducible workflow for quantizing
Hugging Face LLMs into NVIDIA NVFP4 format using **NVIDIA TensorRT Model Optimizer
(ModelOpt)**, suitable for Blackwell (SM12.0+) GPUs.

The resulting checkpoint is compatible with:
- **TensorRT-LLM** (Blackwell builds)
- **vLLM** with `quantization="modelopt"` or `quantization="modelopt_fp4"`
- Other runtimes that support ModelOpt unified HF checkpoints

---

## Table of Contents

- [What is NVFP4?](#what-is-nvfp4)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Installation](#installation)
- [Dataset Configuration](#dataset-configuration)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Advanced Options](#advanced-options)
  - [CPU Offloading for Large Models](#cpu-offloading-for-large-models)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## What is NVFP4?

NVFP4 (NVIDIA FP4) is a 4-bit floating-point quantization format introduced with
NVIDIA Blackwell GPUs that provides:

| Benefit | Details |
|---------|---------|
| **~3.5x Memory Reduction** | Compared to FP16 (~1.8x vs FP8) |
| **2-3x Inference Speedup** | Optimized for Blackwell Tensor Cores |
| **Minimal Accuracy Loss** | Usually <1% degradation vs FP8 |
| **Fine-grained Scaling** | Block size 16 with FP8 scale factors |

### NVFP4 vs Other Formats

| Format | Weights | Activations | Best Use Case |
|--------|---------|-------------|---------------|
| **NVFP4** | FP4 (block-16) | FP16 or FP8 | Maximum compression on Blackwell |
| **FP8** | FP8 | FP8 | Balanced speed/accuracy on Hopper+ |
| **INT4 AWQ** | INT4 | FP16 | Weight-only quantization |
| **INT8 SmoothQuant** | INT8 | INT8 | Full quantization on Ampere+ |

---

## Hardware Requirements

| GPU | Compute Capability | NVFP4 Support |
|-----|-------------------|---------------|
| **RTX PRO 6000 Blackwell** | SM 12.0 | ✅ Native |
| **B100 / B200** | SM 12.0 | ✅ Native |
| **H100 / H200** | SM 9.0 | ⚠️ Calibration only* |
| **A100 / RTX 4090** | SM 8.0+ | ⚠️ Calibration only* |

> \* On pre-Blackwell GPUs, you can **create** NVFP4 checkpoints during calibration,
> but inference with full FP4 acceleration requires Blackwell hardware.
> vLLM will fall back to weight-only mode on older GPUs.

### Minimum VRAM

| Model Size | Standard Script | With CPU Offloading |
|------------|-----------------|---------------------|
| **8B** | 24 GB | Not needed |
| **70B** | 160 GB | ~40-60 GB |
| **123B** | 280-300 GB | ~60-80 GB |
| **405B** | 900 GB | ~80-100 GB |

> **Tip:** Use `quantize_nvfp4_offload.py` when your GPU VRAM is insufficient.
> See [CPU Offloading for Large Models](#cpu-offloading-for-large-models) below.

---

## Software Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| **Python** | 3.10 - 3.12 | Required by nvidia-modelopt |
| **CUDA** | 13.0+ | Required for Blackwell native support |
| **PyTorch** | 2.10.0+ | With CUDA 13.0 support |
| **nvidia-modelopt** | 0.40.0+ | Use `nvidia-modelopt[hf]` for HF compatibility |

> **Note:** CUDA 13.0 is required for native Blackwell (SM 12.0) support. 
> CUDA 12.x can be used for calibration on Hopper/Ampere GPUs.

---

## Installation

We recommend using [uv](https://docs.astral.sh/uv/) for fast, reproducible
Python environments. uv is significantly faster than pip and handles
dependency resolution better.

### Option 1: Using uv (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv --python 3.11
source .venv/bin/activate  # Linux/macOS
# OR: .venv\Scripts\activate  # Windows

# Install PyTorch with CUDA 13.0 support
# Option A: Use nightly builds (recommended for CUDA 13.0)
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# Option B: If stable cu130 wheels are available
# uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Install nvidia-modelopt with HuggingFace extras (recommended)
# The [hf] extra installs tested-compatible versions of transformers
uv pip install "nvidia-modelopt[hf]"

# Install additional dependencies
uv pip install datasets accelerate pyyaml python-dotenv huggingface_hub safetensors
```

### Option 2: Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# OR: venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 13.0 support
# Option A: Use nightly builds (recommended for CUDA 13.0)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# Option B: If stable cu130 wheels are available
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Install nvidia-modelopt with HuggingFace extras (recommended)
# The [hf] extra installs tested-compatible versions of transformers
pip install "nvidia-modelopt[hf]"

# Install other dependencies
pip install datasets accelerate pyyaml python-dotenv huggingface_hub safetensors
```

### Option 3: Using Docker (NVIDIA Container)

For production-ready environment with all dependencies pre-configured:

```bash
docker run --rm -it --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$(pwd)/output_models:/workspace/output_models" \
  -e HF_TOKEN=$HF_TOKEN \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev
```

### Verify Installation

```bash
# Check CUDA version
nvcc --version

# Verify PyTorch CUDA support
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA Version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Verify ModelOpt installation
python -c "import modelopt.torch.quantization as mtq; print('ModelOpt: OK')"
```

Expected output for CUDA 13.0 on Blackwell:
```
PyTorch: 2.10.0+cu130
CUDA Available: True
CUDA Version: 13.0
GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
ModelOpt: OK
```

> **Known Warnings (Safe to Ignore):**
> - `Can't initialize NVML` - This is a known PyTorch issue on some systems and does not affect functionality.
> - `transformers version X.X.X is not tested with nvidia-modelopt` - Install with `nvidia-modelopt[hf]` for tested compatibility (see below).

---

## Dataset Configuration

Calibration datasets are specified via YAML configuration files. The script
expects a specific format to load and sample from multiple HuggingFace datasets.

### YAML Format

```yaml
# Dataset configuration for NVFP4 calibration
datasets:
  - name: ultrachat        # Human-readable name (optional)
    path: HuggingFaceH4/ultrachat_200k
    split: train_sft
    weight: 0.4            # Sampling weight (normalized automatically)
    text_column: text      # Column containing text (auto-detected if not specified)
    streaming: false       # Set true for very large datasets

  - name: writing_prompts
    path: Gryphe/Opus-WritingPrompts
    split: train
    weight: 0.3
    formatter: sharegpt    # Use ShareGPT conversation format
    columns: [conversations]

calibration:
  max_samples: 512         # Target samples for calibration (after filtering)
  max_seq_len: 2048        # Maximum sequence length for tokenization
  seed: 42                 # Random seed for reproducibility
  min_token_length: 32     # Filter samples shorter than this (default: 32)
  oversample_ratio: 1.5    # Request 50% more samples, trim after filtering
```

### Oversampling to Guarantee Sample Count

The script uses **oversampling** to ensure you always get the target number of
calibration samples, even after filtering out short/empty samples.

**How it works:**
1. Request `max_samples * oversample_ratio` samples from datasets (proportionally)
2. Tokenize and filter out samples shorter than `min_token_length`
3. Trim back to exactly `max_samples` (maintaining random order)

**Example with default settings:**
```
Target: 512 samples
Oversample ratio: 1.5x
Requested: 768 samples
After filtering: ~600 samples
Final (trimmed): 512 samples
```

If filtering removes too many samples (more than oversample allows), the script
will warn but continue with whatever samples are available.

### HuggingFace Dataset Cache

HuggingFace datasets are cached **system-wide** (not per-venv) in a shared location.
Once downloaded, they're available to all Python environments.

**Default cache locations:**

| OS | Default Cache Path |
|----|-------------------|
| Linux | `~/.cache/huggingface/datasets/` |
| macOS | `~/.cache/huggingface/datasets/` |
| Windows | `C:\Users\<username>\.cache\huggingface\datasets\` |

**To find your cache location:**
```bash
# Check current cache directory
python -c "from datasets import config; print(config.HF_DATASETS_CACHE)"

# List all cached datasets
ls -la ~/.cache/huggingface/datasets/

# See size of each cached dataset
du -sh ~/.cache/huggingface/datasets/*
```

**Example output:**
```
1.2G  ~/.cache/huggingface/datasets/HuggingFaceH4___ultrachat_200k
450M  ~/.cache/huggingface/datasets/Gryphe___opus-writingprompts
2.1G  ~/.cache/huggingface/datasets/anthracite-org___stheno-filtered-v1.1
```

### Copying Cache Between Environments

Since the cache is system-wide, you typically don't need to copy anything.
However, if you have datasets cached in a different location:

```bash
# Copy entire HuggingFace cache from one location to another
cp -r /path/to/old/venv/.cache/huggingface ~/.cache/huggingface

# Or copy just the datasets folder
cp -r /path/to/source/.cache/huggingface/datasets/* ~/.cache/huggingface/datasets/

# Verify the copy worked
ls ~/.cache/huggingface/datasets/
```

**To use a custom cache location:**
```bash
# Set environment variable before running
export HF_DATASETS_CACHE="/path/to/your/cache"
python quantize_nvfp4.py ...

# Or add to .env file
echo 'HF_DATASETS_CACHE=/path/to/your/cache' >> .env
```

### Using Local/Offline Datasets

If you want to explicitly use datasets from a specific path:

```yaml
datasets:
  # Use HuggingFace ID (will use cache automatically)
  - name: ultrachat
    path: HuggingFaceH4/ultrachat_200k
    split: train_sft
    weight: 0.5

  # Or specify absolute path to local arrow files
  - name: my_local_dataset
    path: /absolute/path/to/dataset/folder
    split: train
    weight: 0.5
```

**To pre-download datasets for offline use:**
```bash
python -c "
from datasets import load_dataset

# This downloads and caches the dataset
datasets_to_cache = [
    ('HuggingFaceH4/ultrachat_200k', 'train_sft'),
    ('Gryphe/Opus-WritingPrompts', 'train'),
    ('anthracite-org/stheno-filtered-v1.1', 'train'),
]

for ds_name, split in datasets_to_cache:
    print(f'Downloading {ds_name}...')
    ds = load_dataset(ds_name, split=split)
    print(f'  Cached {len(ds)} samples')
"
```

### Formatter Types

| Formatter | Description | Expected Columns |
|-----------|-------------|------------------|
| `raw_text` | Plain text content | `text`, `content`, or specified |
| `prompt_answer` | Question/answer pairs | `[prompt, response]` or `[instruction, output]` |
| `sharegpt` | ShareGPT conversation format | `conversations` |
| `chat_completion` | OpenAI-style messages | `messages` |

See `Datasets/SWRP_Default.yaml` for a comprehensive example with multiple
dataset sources balanced for creative writing and roleplay tasks.

---

## Usage

### Basic Usage

```bash
python quantize_nvfp4.py \
  --input_model ./fp16_model \
  --output_model ./nvfp4_model \
  --dataset_yaml Datasets/SWRP_Default.yaml
```

### Advanced Options

```bash
python quantize_nvfp4.py \
  --input_model meta-llama/Llama-3.3-70B-Instruct \
  --output_model ./Llama-3.3-70B-Instruct-NVFP4 \
  --dataset_yaml Datasets/SWRP_Default.yaml \
  --max_samples 1024 \
  --max_seq_len 4096 \
  --oversample_ratio 1.5 \
  --batch_size 1 \
  --trust_remote_code
```

| Option | Default | Description |
|--------|---------|-------------|
| `--max_samples` | 512 | Target calibration samples |
| `--max_seq_len` | 2048 | Max tokens per sample |
| `--oversample_ratio` | 1.5 | Request N times more samples to account for filtering |
| `--batch_size` | 1 | Calibration batch size |
| `--dtype` | auto | Model dtype (auto/float16/bfloat16) |
| `--trust_remote_code` | false | Trust remote code for custom models |
| `--w4a8` | false | Use W4A8 mode (FP4 weights + FP8 activations) |
| `--skip_layers` | none | Layer patterns to skip quantizing |
| `--list_configs` | false | List available ModelOpt configs and exit |

### CPU Offloading for Large Models

For models larger than your available VRAM (e.g., 123B+ models on a single GPU),
use the offloading version of the script:

```bash
python quantize_nvfp4_offload.py \
  --input_model ./Behemoth-R1-123B-v2 \
  --output_model ./Behemoth-R1-123B-v2-NVFP4 \
  --dataset_yaml Datasets/SWRP_Default.yaml \
  --offload_folder ./offload_cache
```

#### Offload Script Options

| Option | Default | Description |
|--------|---------|-------------|
| `--offload_folder` | **required** | Directory for CPU/disk offloading (use fast SSD) |
| `--max_gpu_memory` | auto | Limit GPU memory per device (e.g., "80GiB") |
| `--max_cpu_memory` | auto | Limit CPU memory (e.g., "200GiB") |
| `--clean_offload` | false | Delete offload folder after quantization |

#### Advanced Offloading Examples

```bash
# 123B model with limited GPU memory
python quantize_nvfp4_offload.py \
  --input_model ./123B_model \
  --output_model ./123B_nvfp4 \
  --dataset_yaml Datasets/SWRP_Default.yaml \
  --offload_folder /fast_nvme/offload \
  --max_gpu_memory 80GiB

# Very large model (405B) - aggressive offloading
python quantize_nvfp4_offload.py \
  --input_model meta-llama/Llama-3.1-405B-Instruct \
  --output_model ./405B_nvfp4 \
  --dataset_yaml Datasets/SWRP_Default.yaml \
  --offload_folder /fast_nvme/offload \
  --max_gpu_memory 60GiB \
  --max_cpu_memory 300GiB \
  --clean_offload
```

#### Offload Performance Tips

1. **Use fast storage**: NVMe SSD is strongly recommended for `--offload_folder`
2. **Ensure sufficient disk space**: ~50-100 GB for 123B+ models
3. **More RAM = faster**: CPU offloading uses system RAM before disk
4. **Lower GPU limit = slower**: More offloading means more data transfer
5. **Expect longer runtime**: Offloading adds significant overhead (2-5x slower)

#### Estimated Quantization Time (123B model)

| Configuration | Approximate Time |
|---------------|------------------|
| Full GPU (300GB+ VRAM) | ~30-60 min |
| CPU offload (100GB VRAM) | ~2-4 hours |
| Heavy offload (60GB VRAM) | ~6-12 hours |

### Environment Variables

Create a `.env` file for convenience:

```bash
# .env
HF_TOKEN=hf_xxxxxxxxxxxx
INPUT_MODEL=/path/to/fp16/model
OUTPUT_MODEL=/path/to/nvfp4/model
```

### Multi-GPU Calibration

For very large models, use tensor parallelism during calibration:

```bash
# Using accelerate for multi-GPU
accelerate launch --num_processes 4 quantize_nvfp4.py \
  --input_model ./huge_model \
  --output_model ./nvfp4_model \
  --dataset_yaml Datasets/SWRP_Default.yaml
```

---

## Deployment

### vLLM

Deploy the quantized model with vLLM:

```python
from vllm import LLM, SamplingParams

# Load ModelOpt NVFP4 checkpoint
llm = LLM(
    model="./nvfp4_model",
    quantization="modelopt",  # or "modelopt_fp4" for explicit FP4
    trust_remote_code=True,
    tensor_parallel_size=1,
)

# Generate
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)
outputs = llm.generate(["Hello, how are you?"], sampling_params)
print(outputs[0].outputs[0].text)
```

### vLLM Server

Start an OpenAI-compatible API server:

```bash
vllm serve ./nvfp4_model \
  --quantization modelopt \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1
```

### TensorRT-LLM

For TensorRT-LLM deployment:

```bash
# Convert to TensorRT-LLM format
python convert_checkpoint.py \
  --model_dir ./nvfp4_model \
  --output_dir ./trt_model \
  --dtype float16

# Build TensorRT engine
trtllm-build \
  --checkpoint_dir ./trt_model \
  --output_dir ./trt_engine \
  --gemm_plugin float16
```

---

## Output Checkpoint Structure

The exported checkpoint follows ModelOpt's unified Hugging Face format:

```
nvfp4_model/
├── config.json              # Model configuration
├── hf_quant_config.json     # Quantization metadata (detected by vLLM)
├── model.safetensors        # Quantized weights (or sharded)
├── tokenizer.json           # Tokenizer
├── tokenizer_config.json
├── special_tokens_map.json
└── generation_config.json
```

The `hf_quant_config.json` contains:
```json
{
  "quantization": {
    "quant_algo": "NVFP4",
    "kv_cache_quant_algo": null
  }
}
```

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
--batch_size 1

# Use gradient checkpointing (if supported)
--gradient_checkpointing

# Use CPU offloading for calibration data
--offload_calibration_data
```

### Slow Calibration

- Reduce `max_samples` (512 is usually sufficient for NVFP4)
- Reduce `max_seq_len` (2048 is typical)
- Use faster datasets (avoid `streaming=true` for small calibrations)

### Model Loading Errors

```bash
# For models requiring remote code execution
--trust_remote_code

# For gated models
export HF_TOKEN=your_token_here
```

### vLLM Compatibility Issues

Ensure your vLLM version supports ModelOpt checkpoints:
```bash
pip install vllm>=0.6.0
```

---

## References

- [NVIDIA Model Optimizer GitHub](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
- [NVIDIA ModelOpt Documentation](https://nvidia.github.io/TensorRT-Model-Optimizer/)
- [Introducing NVFP4 Blog Post](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference)
- [vLLM ModelOpt Integration](https://docs.vllm.ai/en/stable/features/quantization/modelopt/)
- [NVIDIA Inference Optimized Checkpoints Collection](https://huggingface.co/collections/nvidia/inference-optimized-checkpoints-with-model-optimizer)

---

## License

This project is provided under the same terms as the upstream dependencies.
See NVIDIA's Model Optimizer license for details on the quantization framework.
