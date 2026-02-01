# NVFP4 Quantization with NVIDIA Model Optimizer

This repository provides a minimal, reproducible workflow for quantizing
a Hugging Face LLM into NVIDIA NVFP4 format using NVIDIA Model Optimizer
(ModelOpt), suitable for Blackwell (SM12.0+) GPUs.

The resulting checkpoint is compatible with:
- TensorRT-LLM (Blackwell builds)
- vLLM with ModelOpt backend
- Other runtimes that support ModelOpt unified HF checkpoints

---

## Hardware Requirements

- NVIDIA Blackwell GPU (SM12.0+)
  - RTX PRO 6000 Blackwell
  - B100 / B200
- CUDA 12.4+ recommended

> NVFP4 is **not** recommended on pre-Blackwell GPUs.

---

## Python Environment

Create a fresh virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip

