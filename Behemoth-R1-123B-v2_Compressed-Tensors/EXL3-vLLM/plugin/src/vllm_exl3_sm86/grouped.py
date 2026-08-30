"""Graph-safe grouped GEMM (exl3_mgemm) for matching packed pairs.

ArtusDev 4.25 gate/up are all K=4, same TP4 width, implicit 3inst. K/V are
mixed K=5/K=6 per layer, so they stay two GEMMs unless a later layer matches.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from .constants import GRAPH_CAPTURE_SIZES, TRELLIS_TILE
from .slicing import ShardId, output_shard_size

_WARMED_KEYS: set[tuple[Any, ...]] = set()
_ENABLED_LAYERS = 0
_WARMUP_FAILED = False


def mgemm_disabled() -> bool:
    return os.environ.get("VLLM_EXL3_DISABLE_MGEMM", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def matching_pair_shards(layer: torch.nn.Module) -> list[ShardId] | None:
    """Return two shard ids that can share one mgemm, else None."""
    shard_ids = list(getattr(layer, "exl3_shard_ids", ()))
    if len(shard_ids) != 2:
        return None
    left, right = shard_ids
    if output_shard_size(layer, left) != output_shard_size(layer, right):
        return None
    trellis = layer.trellis.exl3_tensors
    if left not in trellis or right not in trellis:
        return None
    if tuple(trellis[left].shape) != tuple(trellis[right].shape):
        return None
    mcg_left = left in layer.mcg.exl3_tensors
    mcg_right = right in layer.mcg.exl3_tensors
    mul1_left = left in layer.mul1.exl3_tensors
    mul1_right = right in layer.mul1.exl3_tensors
    if mcg_left != mcg_right or mul1_left != mul1_right:
        return None
    if mcg_left and mul1_left:
        return None
    return [left, right]


def _ptr_table(tensors: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [int(tensor.data_ptr()) for tensor in tensors],
        dtype=torch.int64,
        device=device,
    )


def enable_mgemm_if_eligible(layer: torch.nn.Module) -> bool:
    """Build pointer tables and per-M workspaces. Returns True if decode will fuse."""
    global _ENABLED_LAYERS, _WARMUP_FAILED
    layer.exl3_use_mgemm = False
    if _WARMUP_FAILED or mgemm_disabled():
        return False
    pair = matching_pair_shards(layer)
    if pair is None:
        return False
    from .ops import ext_has_mgemm

    if not ext_has_mgemm():
        return False

    left, right = pair
    trellis = layer.trellis.exl3_tensors[left]
    k = int(trellis.shape[0] * TRELLIS_TILE)
    n = int(output_shard_size(layer, left))
    packed_n = int(trellis.shape[1] * TRELLIS_TILE)
    n = max(n, packed_n)
    bitrate = int(trellis.shape[2] // TRELLIS_TILE)
    device = trellis.device
    layer.exl3_mgemm_shards = pair
    layer.exl3_mgemm_k = k
    layer.exl3_mgemm_n = n
    layer.exl3_mgemm_bitrate = bitrate
    layer.exl3_mgemm_mcg = left in layer.mcg.exl3_tensors
    layer.exl3_mgemm_mul1 = left in layer.mul1.exl3_tensors
    layer.exl3_mgemm_ptrs_trellis = _ptr_table(
        [layer.trellis.exl3_tensors[shard] for shard in pair], device
    )
    layer.exl3_mgemm_ptrs_suh = _ptr_table(
        [layer.suh.exl3_tensors[shard] for shard in pair], device
    )
    layer.exl3_mgemm_ptrs_svh = _ptr_table(
        [layer.svh.exl3_tensors[shard] for shard in pair], device
    )
    layer.exl3_mgemm_out_ws = {}
    layer.exl3_mgemm_xhad_ws = {}
    layer.exl3_mgemm_packed_ws = {}
    layer.exl3_mgemm_x_ws = {}
    for m in GRAPH_CAPTURE_SIZES:
        layer.exl3_mgemm_out_ws[m] = torch.empty(
            (2, m, n), dtype=torch.float16, device=device
        )
        layer.exl3_mgemm_xhad_ws[m] = torch.empty(
            (2, m, k), dtype=torch.float16, device=device
        )
        layer.exl3_mgemm_packed_ws[m] = torch.empty(
            (m, 2 * n), dtype=torch.float16, device=device
        )
        layer.exl3_mgemm_x_ws[m] = torch.empty(
            (m, k), dtype=torch.float16, device=device
        )
    layer.exl3_use_mgemm = True
    _warmup_mgemm(layer)
    if not layer.exl3_use_mgemm:
        return False
    _ENABLED_LAYERS += 1
    if _ENABLED_LAYERS == 1:
        print(
            "vllm_exl3_sm86: fused exl3_mgemm decode enabled for packed "
            f"pairs (bitrate={bitrate}, N={n}, mcg={int(layer.exl3_mgemm_mcg)}, "
            f"mul1={int(layer.exl3_mgemm_mul1)})",
            flush=True,
        )
    return True


def _warmup_mgemm(layer: torch.nn.Module) -> None:
    """Lock autotune for this (K, N, bitrate, codebook, M) before CUDA graphs."""
    device = layer.trellis.exl3_tensors[layer.exl3_mgemm_shards[0]].device
    if device.type != "cuda":
        return
    key = (
        int(layer.exl3_mgemm_k),
        int(layer.exl3_mgemm_n),
        int(layer.exl3_mgemm_bitrate),
        bool(layer.exl3_mgemm_mcg),
        bool(layer.exl3_mgemm_mul1),
        str(device),
    )
    if key in _WARMED_KEYS:
        return
    from .ops import call_exl3_mgemm

    try:
        for m, out in layer.exl3_mgemm_out_ws.items():
            x = torch.zeros(
                (1, m, layer.exl3_mgemm_k), dtype=torch.float16, device=device
            )
            call_exl3_mgemm(
                x,
                layer.exl3_mgemm_ptrs_trellis,
                layer.exl3_mgemm_ptrs_suh,
                layer.exl3_mgemm_ptrs_svh,
                layer.exl3_mgemm_bitrate,
                layer.exl3_mgemm_mcg,
                layer.exl3_mgemm_mul1,
                out,
                layer.exl3_mgemm_xhad_ws[m],
            )
        torch.cuda.synchronize(device)
    except Exception as exc:
        global _WARMUP_FAILED
        _WARMUP_FAILED = True
        layer.exl3_use_mgemm = False
        print(
            f"vllm_exl3_sm86: mgemm warmup failed, falling back to two GEMMs: {exc}",
            flush=True,
        )
        return
    _WARMED_KEYS.add(key)
