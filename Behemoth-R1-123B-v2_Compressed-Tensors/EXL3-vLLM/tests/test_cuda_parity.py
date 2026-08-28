"""CUDA parity: plugin op vs native ExLlamaV3 reconstruct for each bitrate/M."""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="SM86 CUDA tests need a GPU"
)


def _ext():
    try:
        from vllm_exl3_sm86.ops import _load_exl3_ext

        return _load_exl3_ext()
    except Exception as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize("bitrate", [3, 4, 6])
@pytest.mark.parametrize("m", [1, 8, 32, 128, 1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_plugin_matches_reconstructed_fp16(bitrate, m, dtype):
    ext = _ext()
    from vllm_exl3_sm86.ops import call_exl3_gemm

    k, n = 256, 256
    device = torch.device("cuda")
    torch.manual_seed(0)
    trellis = torch.randint(
        -2**15, 2**15, (k // 16, n // 16, 16 * bitrate), dtype=torch.int16, device=device
    )
    suh = torch.randn(k, dtype=torch.float16, device=device)
    svh = torch.randn(n, dtype=torch.float16, device=device)
    x_caller = torch.randn(m, k, dtype=dtype, device=device)
    x = x_caller.to(torch.float16).contiguous()
    got = call_exl3_gemm(x, trellis, suh, svh, mcg=False, mul1=True)
    w = torch.empty((k, n), dtype=torch.float16, device=device)
    ext.reconstruct(w, trellis, bitrate, False, True)
    xh = torch.empty_like(x)
    ext.had_r_128(x, xh, suh, None, 1.0)
    ref = torch.empty((m, n), dtype=torch.float16, device=device)
    ext.hgemm(xh, w, ref)
    ext.had_r_128(ref, ref, None, svh, 1.0)
    torch.cuda.synchronize()
    if not torch.allclose(got, ref, rtol=1e-3, atol=1e-2):
        max_err = (got - ref).abs().max().item()
        pytest.fail(f"parity failed bitrate={bitrate} M={m} dtype={dtype}: max_err={max_err}")
