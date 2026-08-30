from __future__ import annotations

import torch

from vllm_exl3_sm86.ops import exl3_gemm_fake, register_custom_op


def test_fake_impl_matches_packed_n():
    x = torch.empty(4, 256, dtype=torch.float16)
    trellis = torch.empty(16, 8, 64, dtype=torch.int16)
    suh = torch.empty(256, dtype=torch.float16)
    svh = torch.empty(128, dtype=torch.float16)
    out = exl3_gemm_fake(x, trellis, suh, svh, False, True)
    assert tuple(out.shape) == (4, 128)
    assert out.dtype == torch.float16


def test_custom_op_registers_without_cuda_extension():
    register_custom_op()
    assert hasattr(torch.ops, "vllm")
    assert hasattr(torch.ops.vllm, "exl3_gemm")
    assert hasattr(torch.ops.vllm, "exl3_gemm_out")
    assert hasattr(torch.ops.vllm, "exl3_mgemm_out")
