"""Hadamard-aligned TP slicing for dense EXL3 payloads."""

from __future__ import annotations

import torch

from .constants import HADAMARD_BLOCK, TRELLIS_TILE

ShardId = str | int | tuple[int, ...] | None


def unpack_signs(bitfield: torch.Tensor) -> torch.Tensor:
    """Legacy packed su/sv bitfields -> FP16 ±1 Hadamard signs."""
    words = bitfield.contiguous().view(torch.uint16).to(torch.int32)
    masks = 1 << torch.arange(16, device=words.device, dtype=torch.int32)
    negative = (words.unsqueeze(-1) & masks) != 0
    return (1.0 - negative.to(torch.float16) * 2.0).flatten().contiguous()


def require_hadamard_aligned(start: int, size: int, *, axis: str) -> None:
    if start % HADAMARD_BLOCK or size % HADAMARD_BLOCK:
        raise ValueError(
            f"EXL3 TP {axis} slice must be {HADAMARD_BLOCK}-aligned, "
            f"got start={start}, size={size}"
        )


def slice_trellis(tensor: torch.Tensor, *, dim: int, start: int, size: int) -> torch.Tensor:
    axis = "output" if dim == 1 else "input"
    require_hadamard_aligned(start, size, axis=axis)
    return tensor.narrow(dim, start // TRELLIS_TILE, size // TRELLIS_TILE).contiguous()


def slice_vector(tensor: torch.Tensor, start: int, size: int) -> torch.Tensor:
    require_hadamard_aligned(start, size, axis="vector")
    return tensor.narrow(0, start, size).contiguous()


def qkv_output_start(
    shard_id: ShardId,
    shard_size: int,
    tp_rank: int,
    num_kv_head_replicas: int,
) -> int:
    if shard_id in ("k", "v"):
        shard_rank = tp_rank // max(num_kv_head_replicas, 1)
    else:
        shard_rank = tp_rank
    return shard_rank * shard_size


def output_shard_size(layer: object, shard_id: ShardId) -> int:
    sizes = getattr(layer, "exl3_output_partition_sizes")
    if shard_id is None:
        return sizes[0]
    if isinstance(shard_id, str) and shard_id in ("q", "k", "v"):
        return sizes[{"q": 0, "k": 1, "v": 2}[shard_id]]
    if isinstance(shard_id, int):
        return sizes[shard_id]
    return sizes[list(getattr(layer, "exl3_shard_ids")).index(shard_id)]


def slice_on_load(
    layer: object,
    field: str,
    tensor: torch.Tensor,
    shard_id: ShardId,
) -> torch.Tensor:
    """Drop non-local TP payload immediately so workers do not keep full copies."""
    tp_size = int(getattr(layer, "exl3_tp_size", 1))
    if tp_size == 1 or field in {"mcg", "mul1", "su", "sv"}:
        return tensor.contiguous()
    mode = getattr(layer, "exl3_parallel_mode")
    rank = int(getattr(layer, "exl3_tp_rank", 0))
    if mode == "row":
        start = rank * int(layer.exl3_input_size_per_partition)
        size = int(layer.exl3_input_size_per_partition)
        if field == "suh":
            return slice_vector(tensor, start, size)
        if field == "trellis":
            return slice_trellis(tensor, dim=0, start=start, size=size)
        return tensor.contiguous()
    size = output_shard_size(layer, shard_id)
    replicas = int(getattr(layer, "num_kv_head_replicas", 1))
    start = qkv_output_start(shard_id, size, rank, replicas)
    if field == "svh":
        return slice_vector(tensor, start, size)
    if field == "trellis":
        return slice_trellis(tensor, dim=1, start=start, size=size)
    return tensor.contiguous()

