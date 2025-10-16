#!/bin/bash

# System Information Gathering Script for llm-compressor Bug Report
# Run this script and paste the output into your GitHub issue

echo "================================================================================"
echo "llm-compressor NVFP4 Bug - System Information"
echo "================================================================================"
echo ""

echo "---------------- Date/Time ----------------"
date
echo ""

echo "---------------- System Information ----------------"
uname -a
echo ""

echo "---------------- Memory Information ----------------"
free -h
echo ""

echo "---------------- GPU Information ----------------"
nvidia-smi
echo ""

echo "---------------- Python Version ----------------"
python --version
echo ""

echo "---------------- PyTorch + CUDA Info ----------------"
python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Version: {torch.version.cuda}')
print(f'cuDNN Version: {torch.backends.cudnn.version()}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Device: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'CUDA Memory: {props.total_memory / 1024**3:.2f} GB')
    print(f'CUDA Compute Capability: {props.major}.{props.minor}')
"
echo ""

echo "---------------- Key Package Versions ----------------"
pip list | grep -E "llmcompressor|llm-compressor|torch|transformers|compressed-tensors|accelerate|datasets"
echo ""

echo "---------------- Python Environment ----------------"
which python
echo ""

echo "---------------- Virtual Environment ----------------"
echo "VIRTUAL_ENV: $VIRTUAL_ENV"
echo ""

echo "---------------- CUDA Environment Variables ----------------"
env | grep -i cuda
echo ""

echo "================================================================================"
echo "End of System Information"
echo "================================================================================"
echo ""
echo "Copy the above output and paste it into your GitHub issue."
echo ""

