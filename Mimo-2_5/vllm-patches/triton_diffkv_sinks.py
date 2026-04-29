# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton-based DiffKV attention backend with sinks support.

Drop-in replacement for ``FlashAttentionDiffKVBackend`` on devices that
do not support FlashAttention 3 (notably Blackwell consumer GPUs,
sm_120, where wgmma / TMA / TMEM are unavailable).

Inherits the K+V packed KV-cache layout, metadata format, and cache
update path from ``FlashAttentionDiffKVBackend`` / ``FlashAttentionDiffKVImpl``.
Only the attention compute (forward) is replaced: instead of calling
``flash_attn_varlen_func`` we call ``unified_attention_diffkv``.

Limitations of the first iteration (will assert if hit):
  * KV cache must be BF16/FP16 (no FP8).
  * No DCP (decode context parallel).
  * No cascade attention.
  * 2D launch path only (no 3D segmented decode).
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.attention.backends.flash_attn_diffkv import (
    FlashAttentionDiffKVBackend,
    FlashAttentionDiffKVImpl,
)
from vllm.v1.attention.ops.triton_unified_attention_diffkv import (
    unified_attention_diffkv,
)
from vllm.v1.kv_cache_interface import KVQuantMode

logger = init_logger(__name__)


class TritonDiffKVSinksBackend(FlashAttentionDiffKVBackend):
    """vLLM attention backend: Triton-based DiffKV + sinks.

    Inherits everything from ``FlashAttentionDiffKVBackend`` (cache shape,
    stride order, metadata builder, cache update kernel). Only the impl
    class differs.
    """

    @staticmethod
    def get_name() -> str:
        return "TRITON_DIFFKV_SINKS"

    @staticmethod
    def get_impl_cls() -> type["TritonDiffKVSinksImpl"]:
        return TritonDiffKVSinksImpl


class TritonDiffKVSinksImpl(FlashAttentionDiffKVImpl):
    """Attention impl that calls ``unified_attention_diffkv`` instead of
    ``flash_attn_varlen_func``.

    The constructor deliberately does NOT call
    ``get_flash_attn_version(...)`` because:
      * On sm_120 there is no usable FA3, and
      * Our kernel needs no FA-version dispatch.
    """

    def __init__(self, *args, **kwargs) -> None:
        # Skip FlashAttentionDiffKVImpl.__init__ (which probes FA version);
        # call the grandparent FlashAttentionImpl directly.
        super(FlashAttentionDiffKVImpl, self).__init__(*args, **kwargs)
        # Sentinel so any code that introspects this attribute does not
        # trip over None checks.
        self.vllm_flash_attn_version = -1

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """DiffKV+sinks forward via Triton.

        Args:
            query: [num_tokens, num_heads, head_size_q]
            key:   [num_tokens, num_kv_heads, head_size_q]
            value: [num_tokens, num_kv_heads, head_size_v]
            kv_cache: [num_blocks, block_size, num_kv_heads,
                       head_size_q + head_size_v]
            output: [num_tokens, num_heads * head_size_v]
        """
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "Fused output quantization not supported by "
                "TritonDiffKVSinksImpl."
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        attn_type = self.attn_type
        num_actual_tokens = attn_metadata.num_actual_tokens

        # Encoder attention has no KV cache; delegate to the parent's
        # standard path (FlashAttention-2 works fine for plain QKV with
        # equal head dims, and MiMo's vision/audio encoders use it).
        if attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "TritonDiffKVSinksImpl does not yet support quantized KV "
                "cache; got kv_cache_dtype="
                f"{self.kv_cache_dtype!r}. Use BF16/FP16 for now."
            )

        if attn_metadata.use_cascade:
            raise NotImplementedError(
                "Cascade attention is not yet supported by "
                "TritonDiffKVSinksImpl."
            )

        if self.dcp_world_size > 1:
            raise NotImplementedError(
                "Decode context parallel is not yet supported by "
                "TritonDiffKVSinksImpl."
            )

        # K and V are packed in the last dim of kv_cache.
        head_size_q = self.head_size
        head_size_v = FlashAttentionDiffKVBackend.head_size_v
        key_cache = kv_cache[..., :head_size_q]
        value_cache = kv_cache[..., head_size_q:]

        # Convert ``self.sliding_window`` to the (left, right) tuple
        # accepted by ``unified_attention_diffkv``.
        if self.sliding_window is None:
            window_size = (-1, -1)
        else:
            window_size = tuple(self.sliding_window)

        # Output for the kernel must be viewed as
        # [num_tokens, num_heads, head_size_v] (last dim is V dim,
        # not Q dim).
        out_view = output[:num_actual_tokens].view(
            num_actual_tokens, self.num_heads, head_size_v
        )

        unified_attention_diffkv(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=out_view,
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            window_size=window_size,
            block_table=attn_metadata.block_table,
            softcap=self.logits_soft_cap or 0.0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            alibi_slopes=self.alibi_slopes,
            output_scale=None,
            qq_bias=None,
            sinks=self.sinks,
            use_alibi_sqrt=False,
            kv_quant_mode=KVQuantMode.NONE,
        )

        return output
