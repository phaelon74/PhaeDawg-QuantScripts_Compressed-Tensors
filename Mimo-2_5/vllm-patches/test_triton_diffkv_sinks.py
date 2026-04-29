# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for ``unified_attention_diffkv``.

Compares the Triton kernel output against a pure-PyTorch reference for a
range of shapes covering MiMo-V2.5's actual attention configuration:

  * ``head_size_q = 192, head_size_v = 128`` (MiMo qk vs v dim)
  * ``num_q_heads = 64, num_kv_heads = 4`` (GQA group = 16)
  * Sliding window = 128 with attention sink bias

Run from the vLLM repo root with::

    pytest -xvs tests/kernels/attention/test_triton_diffkv_sinks.py
"""

import math

import pytest
import torch

from vllm.v1.attention.ops.triton_unified_attention_diffkv import (
    unified_attention_diffkv,
)


def _ref_attention_diffkv(
    q: torch.Tensor,                     # [T_q, num_q_heads, hd_q]
    k: torch.Tensor,                     # [T_k, num_kv_heads, hd_q]
    v: torch.Tensor,                     # [T_k, num_kv_heads, hd_v]
    softmax_scale: float,
    sliding_window: int | None,
    sinks: torch.Tensor | None,          # [num_q_heads]
    causal: bool = True,
) -> torch.Tensor:
    """Pure-PyTorch single-sequence reference, unbatched.

    Computes attention in fp32, returns bf16 to match the kernel's IO.
    """
    T_q, num_q_heads, hd_q = q.shape
    T_k, num_kv_heads, _ = k.shape
    hd_v = v.shape[-1]
    num_q_per_kv = num_q_heads // num_kv_heads

    qf = q.float()
    kf = k.float().repeat_interleave(num_q_per_kv, dim=1)   # broadcast GQA
    vf = v.float().repeat_interleave(num_q_per_kv, dim=1)

    # logits: [T_q, num_q_heads, T_k]
    logits = torch.einsum("qhd,khd->qhk", qf, kf) * softmax_scale

    # Causal + sliding window mask.
    q_abs = torch.arange(T_q, device=q.device) + (T_k - T_q)
    k_abs = torch.arange(T_k, device=q.device)
    rel = q_abs[:, None] - k_abs[None, :]                   # [T_q, T_k]
    valid = rel >= 0 if causal else torch.ones_like(rel, dtype=torch.bool)
    if sliding_window is not None:
        valid = valid & (rel < sliding_window)
    logits = logits.masked_fill(~valid[:, None, :], float("-inf"))

    # Softmax with optional sink bias in the denominator.
    if sinks is not None:
        sink_logit = sinks.float().view(1, num_q_heads, 1)
        m = torch.max(
            logits.amax(dim=-1, keepdim=True),
            sink_logit,
        )
        unnorm = torch.exp(logits - m)
        unnorm = torch.where(torch.isnan(unnorm), torch.zeros_like(unnorm), unnorm)
        denom = unnorm.sum(dim=-1, keepdim=True) + torch.exp(sink_logit - m)
    else:
        m = logits.amax(dim=-1, keepdim=True)
        unnorm = torch.exp(logits - m)
        unnorm = torch.where(torch.isnan(unnorm), torch.zeros_like(unnorm), unnorm)
        denom = unnorm.sum(dim=-1, keepdim=True)

    p = unnorm / torch.clamp(denom, min=1e-30)              # [T_q, h, T_k]
    out = torch.einsum("qhk,khd->qhd", p, vf)               # [T_q, h, hd_v]
    return out.to(q.dtype)


def _build_paged_kv_cache(
    k_flat: torch.Tensor,                # [T_k, num_kv_heads, hd_q]
    v_flat: torch.Tensor,                # [T_k, num_kv_heads, hd_v]
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack K and V into the DiffKV cache layout used by vLLM:

        kv_cache : [num_blocks, block_size, num_kv_heads, hd_q + hd_v]

    Returns ``(kv_cache, block_table)``.  block_table has shape
    ``[num_seqs, max_num_blocks]`` and lists physical block indices in
    logical order.  We build a single-sequence layout.
    """
    T_k, num_kv_heads, hd_q = k_flat.shape
    hd_v = v_flat.shape[-1]
    num_blocks = (T_k + block_size - 1) // block_size

    kv_cache = torch.zeros(
        num_blocks,
        block_size,
        num_kv_heads,
        hd_q + hd_v,
        dtype=k_flat.dtype,
        device=k_flat.device,
    )

    # Fill block-by-block.
    for blk in range(num_blocks):
        lo = blk * block_size
        hi = min(lo + block_size, T_k)
        n = hi - lo
        kv_cache[blk, :n, :, :hd_q] = k_flat[lo:hi]
        kv_cache[blk, :n, :, hd_q:] = v_flat[lo:hi]

    block_table = torch.arange(
        num_blocks, dtype=torch.int32, device=k_flat.device
    ).unsqueeze(0)
    return kv_cache, block_table


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seq_q", [1, 17, 64])
@pytest.mark.parametrize("seq_k", [128, 257, 512])
@pytest.mark.parametrize(
    "hd_q,hd_v",
    [
        (192, 128),    # MiMo-V2.5 exact
        (128, 128),    # equal-dim sanity check
        (256, 128),    # stress test for BLOCK budget
    ],
)
@pytest.mark.parametrize("use_sinks", [False, True])
@pytest.mark.parametrize("sliding_window", [None, 128])
def test_unified_attention_diffkv_correctness(
    seq_q: int,
    seq_k: int,
    hd_q: int,
    hd_v: int,
    use_sinks: bool,
    sliding_window: int | None,
):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if seq_q > seq_k:
        pytest.skip("query length cannot exceed key length")

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    num_q_heads = 64
    num_kv_heads = 4
    block_size = 16
    softmax_scale = 1.0 / math.sqrt(hd_q)

    # Random Q/K/V.
    q = torch.randn(seq_q, num_q_heads, hd_q, dtype=dtype, device=device) * 0.02
    k = torch.randn(seq_k, num_kv_heads, hd_q, dtype=dtype, device=device) * 0.02
    v = torch.randn(seq_k, num_kv_heads, hd_v, dtype=dtype, device=device) * 0.02

    sinks = (
        torch.randn(num_q_heads, dtype=dtype, device=device)
        if use_sinks
        else None
    )

    # ---- Reference ----
    ref = _ref_attention_diffkv(
        q, k, v,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        sinks=sinks,
        causal=True,
    )

    # ---- Triton ----
    kv_cache, block_table = _build_paged_kv_cache(k, v, block_size)
    key_cache = kv_cache[..., :hd_q]
    value_cache = kv_cache[..., hd_q:]

    out = torch.zeros_like(ref)

    cu_seqlens_q = torch.tensor(
        [0, seq_q], dtype=torch.int32, device=device
    )
    seqused_k = torch.tensor([seq_k], dtype=torch.int32, device=device)

    window = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)

    unified_attention_diffkv(
        q=q,
        k=key_cache,
        v=value_cache,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=seq_q,
        seqused_k=seqused_k,
        max_seqlen_k=seq_k,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=window,
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        alibi_slopes=None,
        sinks=sinks,
    )

    # BF16 tolerance: kernel uses fp32 accum but inputs are bf16, and
    # the GQA broadcast in the reference goes through fp32 too.
    abs_diff = (out.float() - ref.float()).abs()
    max_diff = abs_diff.max().item()
    rel_diff = (abs_diff / ref.float().abs().clamp(min=1e-3)).max().item()

    print(
        f"[diffkv] seq_q={seq_q} seq_k={seq_k} hd_q={hd_q} hd_v={hd_v} "
        f"sinks={use_sinks} sw={sliding_window}  "
        f"max_abs={max_diff:.4f}  max_rel={rel_diff:.4f}"
    )

    assert max_diff < 5e-2, (
        f"Triton DiffKV output diverges from reference: "
        f"max_abs={max_diff:.4f}"
    )
