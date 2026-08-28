"""EXL3 dense constants shared by the plugin and CPU tests."""

from __future__ import annotations

MCG_SENTINEL = 0xCBAC1FED
MUL1_SENTINEL = 0x83DCD12D


def int32_sentinel(value: int) -> "torch.Tensor":
    """Pack a uint32 EXL3 marker into signed int32 without Python overflow."""
    import torch

    wrapped = value - (1 << 32) if value >= (1 << 31) else value
    return torch.tensor([wrapped], dtype=torch.int32)


HADAMARD_BLOCK = 128
TRELLIS_TILE = 16
MIN_BITRATE = 1
MAX_BITRATE = 8

# Upstream ExLlamaV3 default. SM86 Behemoth shapes must overwrite this via
# VLLM_EXL3_RECONSTRUCT_M or manifests/sm86_crossover.json after microbench.
DEFAULT_RECONSTRUCT_M = 144
FUSED_RECONSTRUCT_M = 1024
MAX_RECONSTRUCT_SLICE_N = 32768

PLUGIN_NAME = "vllm_exl3_sm86"
QUANTIZATION_NAME = "exl3"
REQUIRED_VLLM_VERSION = "0.1.dev12995+g1f369db5d"
REQUIRED_TORCH_PREFIX = "2.9.1"
REQUIRED_CUDA = "13.0"
PINNED_EXLLAMAV3_COMMIT = "0c49587a7c235e6303a6bbedc8b665272ad3a2ea"
PINNED_VLLM_GIT_SHA = "1f369db5d5680355e8909df56e77592c55ebdbf9"

# Behemoth-R1-123B-v2 architecture (dense Mistral).
BEHEMOTH_LAYERS = 88
BEHEMOTH_HIDDEN = 12288
BEHEMOTH_INTERMEDIATE = 28672
BEHEMOTH_Q_HEADS = 96
BEHEMOTH_KV_HEADS = 8
BEHEMOTH_HEAD_DIM = 128
BEHEMOTH_VOCAB = 32768
BEHEMOTH_DECODER_LINEARS = BEHEMOTH_LAYERS * 7  # 616

# Local TP4 matrix shapes (K, N) after column/row slicing.
BEHEMOTH_TP4_SHAPES = {
    "q_proj": (12288, 3072),
    "k_proj": (12288, 256),
    "v_proj": (12288, 256),
    "o_proj": (3072, 12288),
    "gate_proj": (12288, 7168),
    "up_proj": (12288, 7168),
    "down_proj": (7168, 12288),
    "lm_head": (12288, 8192),
}

DECODER_BITRATES = (3, 4)
HEAD_BITRATE = 6
MICROBENCH_M = (1, 2, 4, 8, 16, 32, 64, 128, 144, 256, 512, 1024, 4096)
GRAPH_CAPTURE_SIZES = (1, 2, 4)
