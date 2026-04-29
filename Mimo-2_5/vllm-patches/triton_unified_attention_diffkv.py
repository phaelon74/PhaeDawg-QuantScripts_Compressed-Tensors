# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from vllm/v1/attention/ops/triton_unified_attention.py
# Original authors:
#   Burkhard Ringlein, Jan van Lunteren, Chih-Chieh Yang, Thomas Parnell
#
# Modifications: Add DiffKV support (head_size_q != head_size_v).
# Target hardware: Blackwell consumer (sm_120) and any GPU with the
# standard mma.sync tensor cores. No FA3-specific instructions are used,
# so this kernel runs everywhere unified_attention runs.
#
# Differences from the original kernel:
#   1. HEAD_SIZE / HEAD_SIZE_PADDED are split into HEAD_SIZE_Q (used for
#      Q and K loads) and HEAD_SIZE_V (used for V loads, the accumulator,
#      and the output write).
#   2. Two separate `offs_d_*` ranges and `dim_mask_*` masks gate the Q/K
#      vs V dimensions independently.
#   3. The Python wrapper auto-detects head dims from `q.shape[-1]` and
#      `v.shape[-1]`. The `out` tensor must be allocated with the V dim.
#
# Everything else (softmax + sinks + sliding window + alibi + softcap +
# paged KV + GQA + 2D/3D segmented decode + per-tensor FP8) is unchanged
# because none of those code paths depend on Q-vs-V dim equality.
#
# 2D mode is the only mode currently supported by this DiffKV variant.
# 3D / segmented-decode requires a parallel reduce kernel; once the
# baseline is validated, that path can be added by mirroring the
# changes onto reduce_segments.

from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_attention_helpers import (
    apply_alibi_to_score,
    apply_softcap,
    cdiv_fn,
    compute_kv_seq_mask,
    compute_tile_loop_bounds,
    init_softmax_M,
    load_qq_bias_tile,
    resolve_seq_and_query_len,
    softmax_step,
)
from vllm.v1.kv_cache_interface import KVQuantMode

logger = init_logger(__name__)
is_batch_invariant = envs.VLLM_BATCH_INVARIANT
float8_info = torch.finfo(current_platform.fp8_dtype())


@triton.jit
def _cast_kv_tile(data, Q, tensor_scale, KV_QUANT_MODE: tl.constexpr):
    """Cast a loaded KV tile to Q's dtype, dequantizing if needed.

    Mirrors the behaviour of the helper in triton_unified_attention.py.
    """
    if KV_QUANT_MODE == 1:
        if Q.dtype.is_fp8():
            return data.to(Q.dtype)
        return (data.to(tl.float32) * tl.load(tensor_scale)).to(Q.dtype)
    return data.to(Q.dtype)


@triton.jit
def kernel_unified_attention_diffkv(
    # ---- Output destinations ----
    output_ptr,              # [num_tokens, num_query_heads, HEAD_SIZE_V]
    # ---- Inputs ----
    query_ptr,               # [num_tokens, num_query_heads, HEAD_SIZE_Q]
    key_cache_ptr,           # [num_blocks, block_size, num_kv_heads,
                             #  HEAD_SIZE_Q + HEAD_SIZE_V]  (DiffKV pack)
    value_cache_ptr,         # same buffer as key_cache, V slice begins
                             # at offset HEAD_SIZE_Q in the last dim
    sink_ptr,                # [num_query_heads]  or  None (USE_SINKS)
    block_tables_ptr,
    seq_lens_ptr,
    alibi_slopes_ptr,
    qq_bias_ptr,
    # ---- Scalars ----
    scale,                   # float (softmax scale)
    k_scale,                 # float ptr (per-tensor FP8 dequant), unused
                             # in BF16 path
    v_scale,                 # float ptr
    softcap,                 # float
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,    # should be HEAD_SIZE_Q
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,   # should be HEAD_SIZE_V
    qq_bias_stride_0: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE_Q: tl.constexpr,           # NEW: Q/K logical dim
    HEAD_SIZE_Q_PADDED: tl.constexpr,    # NEW: Q/K dim padded to power-of-2
    HEAD_SIZE_V: tl.constexpr,           # NEW: V/output logical dim
    HEAD_SIZE_V_PADDED: tl.constexpr,    # NEW: V dim padded to power-of-2
    USE_ALIBI_SLOPES: tl.constexpr,
    USE_ALIBI_SQRT: tl.constexpr,
    USE_QQ_BIAS: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    USE_SINKS: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    USE_MM_PREFIX: tl.constexpr,
    MAX_MM_RANGES: tl.constexpr,
    mm_prefix_range_ptr,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.constexpr,      # = element size in elements
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.constexpr,
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    USE_FP8: tl.constexpr,
    KV_QUANT_MODE: tl.constexpr = 0,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
    out_scale=None,                      # FP8 output rescale ptr (unused)
):
    # ---- Program-id bookkeeping (identical to unified_attention) ----
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    (
        seq_idx,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        cur_batch_query_len,
        seq_len,
    ) = resolve_seq_and_query_len(
        query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    # ---- Address ranges ----
    offs_m = tl.arange(0, BLOCK_M)
    offs_d_qk = tl.arange(0, HEAD_SIZE_Q_PADDED)   # for Q & K
    offs_d_v = tl.arange(0, HEAD_SIZE_V_PADDED)    # for V & accumulator & output
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = (
        kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    )
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d_qk[None, :]
    )

    dim_mask_qk = tl.where(offs_d_qk < HEAD_SIZE_Q, 1, 0).to(tl.int1)
    dim_mask_v = tl.where(offs_d_v < HEAD_SIZE_V, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # ---- Load Q ----
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask_qk[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    M = init_softmax_M(
        sink_ptr,
        query_offset_1,
        query_mask_1,
        0,            # segm_idx; 2D path only
        BLOCK_M,
        USE_SINKS,
        False,        # IS_3D
    )
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    # Accumulator is sized to the V dim because the final output dim = V dim.
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_V_PADDED], dtype=tl.float32)

    context_len = seq_len - cur_batch_query_len

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    if USE_QQ_BIAS:
        qq_bias_row_ptrs = qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0

    loop_lo, loop_hi, max_seq_prefix_len = compute_tile_loop_bounds(
        context_len,
        seq_len,
        cur_batch_query_len,
        q_block_local_idx,
        0,                   # segm_idx
        0,                   # tiles_per_segment
        TILE_SIZE,
        BLOCK_M,
        BLOCK_Q,
        num_queries_per_kv,
        SLIDING_WINDOW,
        USE_MM_PREFIX,
        False,               # IS_3D
        -1,                  # CHUNK_LOOKBACK
        -1,                  # CHUNK_SIZE
    )

    # ---- Main K/V tile loop ----
    for j in range(loop_lo, loop_hi):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        # K offset: last dim uses HEAD_SIZE_Q
        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d_qk[:, None] * stride_k_cache_3
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )
        # V offset: last dim uses HEAD_SIZE_V (and the V cache pointer
        # already points HEAD_SIZE_Q elements into the packed buffer).
        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d_v[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )

        # K : (HEAD_SIZE_Q_PADDED, TILE_SIZE)
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask_qk[:, None] & tile_mask[None, :],
            other=0.0,
        )
        K = _cast_kv_tile(K_load, Q, k_scale, KV_QUANT_MODE)

        # V : (TILE_SIZE, HEAD_SIZE_V_PADDED)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask_v[None, :] & tile_mask[:, None],
            other=0.0,
        )
        V = _cast_kv_tile(V_load, Q, v_scale, KV_QUANT_MODE)

        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = compute_kv_seq_mask(
            query_abs_pos,
            seq_offset,
            seq_idx,
            mm_prefix_range_ptr,
            SLIDING_WINDOW,
            USE_MM_PREFIX,
            MAX_MM_RANGES,
            -1,                # CHUNK_LOOKBACK
            -1,                # CHUNK_SIZE
        )

        # S : (BLOCK_M, TILE_SIZE) -- fp32 logits
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask,
            S,
            float("-inf"),
        )

        if USE_ALIBI_SLOPES:
            S = apply_alibi_to_score(
                S, alibi_slope, seq_offset, context_len, query_pos, USE_ALIBI_SQRT
            )

        if USE_QQ_BIAS:
            S += load_qq_bias_tile(
                qq_bias_row_ptrs, seq_offset, context_len, qq_bias_stride_0
            )

        M, L, P, alpha = softmax_step(S, M, L)
        acc = acc * alpha[:, None]

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW,
                V,
                0.0,
            )

        # acc : (BLOCK_M, HEAD_SIZE_V_PADDED)
        acc += tl.dot(P.to(V.dtype), V)

    # ---- Epilogue (2D path) ----
    acc = acc / L[:, None]
    if USE_FP8:
        acc = acc * tl.load(out_scale)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d_v[None, :]
    )
    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask_v[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


def _get_tile_size(
    max_head_size: int,
    element_size: int,
    is_prefill: bool,
) -> int:
    """Pick a TILE_SIZE that fits sm_120's 99 KB SMEM with safe margin.

    The tightest constraint is the K tile: ``HEAD_SIZE_Q * TILE_SIZE *
    element_size``.  Empirically, TILE_SIZE=32 prefill / 16 decode
    matches the upstream defaults and fits comfortably for any
    head_size <= 256 in BF16 (16 KB K + 8 KB V + 2 KB S + accumulator).
    """
    if is_prefill:
        return 32
    return 16 if element_size >= 2 else 32


def unified_attention_diffkv(
    q: torch.Tensor,                     # [num_tokens, num_q_heads, head_size_q]
    k: torch.Tensor,                     # [num_blocks, block_size, num_kv_heads,
                                         #  head_size_q]   (sliced view)
    v: torch.Tensor,                     # [num_blocks, block_size, num_kv_heads,
                                         #  head_size_v]   (sliced view)
    out: torch.Tensor,                   # [num_tokens, num_q_heads, head_size_v]
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size,                         # tuple(int, int): (left, right);
                                         # right is ignored for causal
    block_table: torch.Tensor,
    softcap: float,
    q_descale,                           # must be None
    k_descale,                           # per-tensor FP8 scale, or None
    v_descale,                           # per-tensor FP8 scale, or None
    alibi_slopes: torch.Tensor | None = None,
    output_scale: torch.Tensor | None = None,
    qq_bias: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,   # [num_q_heads], the s_aux bias
    use_alibi_sqrt: bool = False,
    kv_quant_mode: KVQuantMode = KVQuantMode.NONE,
):
    """Drop-in DiffKV-aware replacement for ``unified_attention``.

    Differences from the upstream signature:
      * `out` must be pre-allocated with shape
        ``[num_tokens, num_q_heads, head_size_v]``.
      * The 3D / segmented-decode mode and per-token-head KV quant are
        not yet supported (assert below).
      * Chunked attention (chunk_lookback) not supported.
    """
    assert causal, "Only causal attention is supported in DiffKV path"
    assert q_descale is None, "Q scales not supported"
    assert kv_quant_mode in (
        KVQuantMode.NONE,
        KVQuantMode.FP8_PER_TENSOR,
    ), (
        "DiffKV path supports KV_QUANT_MODE in {NONE, FP8_PER_TENSOR}; "
        f"got {kv_quant_mode.name}."
    )

    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], (
            "Sinks must have num_query_heads entries; "
            f"got {sinks.shape[0]} vs {q.shape[1]}"
        )

    head_size_q = q.shape[2]
    head_size_v = out.shape[2]
    assert k.shape[3] == head_size_q, (
        f"k last-dim {k.shape[3]} != q head_size_q {head_size_q}"
    )
    assert v.shape[3] == head_size_v, (
        f"v last-dim {v.shape[3]} != out head_size_v {head_size_v}"
    )

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    # Upper bound for total Q blocks across the batch (matches upstream).
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0

    max_head = max(head_size_q, head_size_v)
    tile_size = _get_tile_size(max_head, q.element_size(), is_prefill=True)

    grid = (total_num_q_blocks, num_kv_heads)

    kernel_unified_attention_diffkv[grid](
        output_ptr=out,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        sink_ptr=sinks,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=alibi_slopes,
        qq_bias_ptr=qq_bias,
        scale=softmax_scale,
        k_scale=k_descale,
        v_scale=v_descale,
        softcap=softcap,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        qq_bias_stride_0=qq_bias.stride(0) if qq_bias is not None else 0,
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        HEAD_SIZE_Q=head_size_q,
        HEAD_SIZE_Q_PADDED=triton.next_power_of_2(head_size_q),
        HEAD_SIZE_V=head_size_v,
        HEAD_SIZE_V_PADDED=triton.next_power_of_2(head_size_v),
        USE_ALIBI_SLOPES=alibi_slopes is not None,
        USE_ALIBI_SQRT=use_alibi_sqrt,
        USE_QQ_BIAS=qq_bias is not None,
        USE_SOFTCAP=(softcap > 0),
        USE_SINKS=(sinks is not None),
        USE_MM_PREFIX=False,
        MAX_MM_RANGES=0,
        mm_prefix_range_ptr=None,
        SLIDING_WINDOW=sliding_window_val,
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        USE_FP8=output_scale is not None,
        KV_QUANT_MODE=int(kv_quant_mode),
        out_scale=output_scale,
    )
