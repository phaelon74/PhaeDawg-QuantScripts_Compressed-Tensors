from __future__ import annotations

import pytest
import torch

from vllm_exl3_sm86.constants import MUL1_SENTINEL
from vllm_exl3_sm86.slicing import (
    qkv_output_start,
    slice_trellis,
    slice_vector,
    unpack_signs,
)


def test_unpack_signs_roundtrip_pattern():
    bits = torch.tensor([0x0001], dtype=torch.int16)
    signs = unpack_signs(bits)
    assert signs.dtype == torch.float16
    assert signs.numel() == 16
    assert signs[0].item() == -1.0
    assert signs[1].item() == 1.0


def test_column_slice_n_and_svh():
    trellis = torch.arange(8 * 16 * 48, dtype=torch.int16).reshape(8, 16, 48)
    svh = torch.arange(256, dtype=torch.float16)
    sliced = slice_trellis(trellis, dim=1, start=0, size=128)
    assert sliced.shape == (8, 8, 48)
    rank1 = slice_trellis(trellis, dim=1, start=128, size=128)
    assert rank1.shape == (8, 8, 48)
    vec = slice_vector(svh, 128, 128)
    assert vec.numel() == 128
    with pytest.raises(ValueError, match="128-aligned"):
        slice_trellis(trellis, dim=1, start=64, size=128)



def test_row_slice_k_and_suh():
    trellis = torch.zeros((32, 8, 64), dtype=torch.int16)
    suh = torch.ones(512, dtype=torch.float16)
    sliced = slice_trellis(trellis, dim=0, start=128, size=128)
    assert sliced.shape == (8, 8, 64)
    assert slice_vector(suh, 128, 128).shape == (128,)


def test_qkv_kv_head_replication():
    # TP4, 8 KV heads, 4 ranks => 2 replicas, K/V shard rank = tp_rank // 2.
    assert qkv_output_start("q", 3072, tp_rank=3, num_kv_head_replicas=1) == 9216
    assert qkv_output_start("k", 256, tp_rank=3, num_kv_head_replicas=2) == 256
    assert qkv_output_start("v", 256, tp_rank=0, num_kv_head_replicas=2) == 0


def test_markers_are_independent_scalars():
    mul1 = torch.tensor([MUL1_SENTINEL], dtype=torch.int32)
    assert mul1.numel() == 1
    assert int(mul1.item()) & 0xFFFFFFFF == MUL1_SENTINEL


def test_slice_on_load_drops_nonlocal_column_payload():
    from types import SimpleNamespace
    from vllm_exl3_sm86.slicing import slice_on_load

    trellis = torch.zeros((8, 16, 48), dtype=torch.int16)
    svh = torch.arange(256, dtype=torch.float16)
    suh = torch.ones(128, dtype=torch.float16)
    layer = SimpleNamespace(
        exl3_tp_size=4,
        exl3_tp_rank=1,
        exl3_parallel_mode="column",
        exl3_output_partition_sizes=[64],
        exl3_shard_ids=[None],
        num_kv_head_replicas=1,
    )
    # N=256, 4 ranks => 64 per rank, but must be 128-aligned. Use 128.
    layer.exl3_output_partition_sizes = [128]
    local_t = slice_on_load(layer, "trellis", trellis, None)
    local_v = slice_on_load(layer, "svh", svh, None)
    local_u = slice_on_load(layer, "suh", suh, None)
    assert local_t.shape == (8, 8, 48)
    assert local_v.numel() == 128
    assert local_u.numel() == 128  # column-parallel keeps full suh

