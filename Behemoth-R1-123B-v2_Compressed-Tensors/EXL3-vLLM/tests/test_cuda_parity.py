"""CUDA parity: plugin wrappers vs native exllamav3_ext.

Random trellis bits are valid packed indices, but compressed GEMM and
reconstruct are different kernels. Compare:

- plugin compressed path vs raw ext.exl3_gemm (same kernel, tight)
- plugin reconstruct path vs reconstruct_hgemm (same kernel, tight)
- compressed vs reconstruct with microbench-style tiles (fp16 noise)
"""

from __future__ import annotations

import pytest
import torch

from vllm_exl3_sm86.constants import FUSED_RECONSTRUCT_M, TRELLIS_TILE

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="SM86 CUDA tests need a GPU"
)


def _ext():
    try:
        from vllm_exl3_sm86.ops import _load_exl3_ext

        return _load_exl3_ext()
    except Exception as exc:
        pytest.skip(str(exc))


def _payloads(bitrate: int, m: int, dtype: torch.dtype, device: torch.device, seed: int = 0):
    k, n = 256, 256
    torch.manual_seed(seed)
    # Match kernel_microbench: bounded codes, not the full int16 range.
    trellis = torch.randint(
        -1024,
        1024,
        (k // TRELLIS_TILE, n // TRELLIS_TILE, TRELLIS_TILE * bitrate),
        dtype=torch.int16,
        device=device,
    )
    suh = torch.randn(k, dtype=torch.float16, device=device)
    svh = torch.randn(n, dtype=torch.float16, device=device)
    x = torch.randn(m, k, dtype=dtype, device=device).to(torch.float16).contiguous()
    return x, trellis, suh, svh


def _native_compressed(ext, x, trellis, suh, svh, mcg: bool, mul1: bool):
    output = torch.empty(
        (x.shape[0], trellis.shape[1] * TRELLIS_TILE),
        dtype=torch.float16,
        device=x.device,
    )
    x_had = torch.empty_like(x)
    ext.exl3_gemm(x, trellis, output, suh, x_had, svh, -1, mcg, mul1, 0)
    return output


def _reconstruct_ref(
    ext,
    x,
    trellis,
    suh,
    svh,
    bitrate: int,
    *,
    mcg: bool = False,
    mul1: bool = True,
):
    k = trellis.shape[0] * TRELLIS_TILE
    n = trellis.shape[1] * TRELLIS_TILE
    w = torch.empty((k, n), dtype=torch.float16, device=x.device)
    ext.reconstruct(w, trellis, bitrate, mcg, mul1)
    xh = torch.empty_like(x)
    ext.had_r_128(x, xh, suh, None, 1.0)
    ref = torch.empty((x.shape[0], n), dtype=torch.float16, device=x.device)
    ext.hgemm(xh, w, ref)
    ext.had_r_128(ref, ref, None, svh, 1.0)
    return ref


@pytest.mark.parametrize("bitrate", [5, 6])
@pytest.mark.parametrize("mode", [1, 2])
def test_k56_3inst_gemv_matches_reconstruct(bitrate, mode, monkeypatch):
    monkeypatch.setenv("EXL3_GEMV_K56", str(mode))
    monkeypatch.setenv("VLLM_EXL3_FORCE_COMPRESSED", "1")
    monkeypatch.delenv("VLLM_EXL3_FORCE_RECONSTRUCT", raising=False)
    ext = _ext()
    from vllm_exl3_sm86.ops import call_exl3_gemm

    device = torch.device("cuda")
    x, trellis, suh, svh = _payloads(
        bitrate, 1, torch.float16, device
    )
    got = call_exl3_gemm(
        x, trellis, suh, svh, mcg=False, mul1=False
    )
    ref = _reconstruct_ref(
        ext,
        x,
        trellis,
        suh,
        svh,
        bitrate,
        mcg=False,
        mul1=False,
    )
    torch.cuda.synchronize()
    if not torch.allclose(got, ref, rtol=5e-2, atol=0.75):
        max_err = (got.float() - ref.float()).abs().max().item()
        pytest.fail(
            f"K{bitrate} mode={mode} 3inst GEMV parity failed: "
            f"max_err={max_err}"
        )


@pytest.mark.parametrize("mode", [1, 2])
def test_k4_arithmetic_gemv_matches_reconstruct(mode, monkeypatch):
    monkeypatch.setenv("EXL3_GEMV_K4_ARITH", str(mode))
    monkeypatch.setenv("VLLM_EXL3_FORCE_COMPRESSED", "1")
    monkeypatch.delenv("VLLM_EXL3_FORCE_RECONSTRUCT", raising=False)
    ext = _ext()
    from vllm_exl3_sm86.ops import call_exl3_gemm

    device = torch.device("cuda")
    x, trellis, suh, svh = _payloads(4, 1, torch.float16, device)
    got = call_exl3_gemm(
        x, trellis, suh, svh, mcg=False, mul1=False
    )
    ref = _reconstruct_ref(
        ext,
        x,
        trellis,
        suh,
        svh,
        4,
        mcg=False,
        mul1=False,
    )
    torch.cuda.synchronize()
    if not torch.allclose(got, ref, rtol=5e-2, atol=0.75):
        max_err = (got.float() - ref.float()).abs().max().item()
        pytest.fail(
            f"K4 arithmetic mode={mode} parity failed: max_err={max_err}"
        )


def test_k4_slim_gemv_matches_reconstruct(monkeypatch):
    monkeypatch.setenv("EXL3_GEMV_K4_SLIM", "1")
    monkeypatch.delenv("EXL3_GEMV_K4_ARITH", raising=False)
    monkeypatch.setenv("VLLM_EXL3_FORCE_COMPRESSED", "1")
    monkeypatch.delenv("VLLM_EXL3_FORCE_RECONSTRUCT", raising=False)
    ext = _ext()
    from vllm_exl3_sm86.ops import call_exl3_gemm

    device = torch.device("cuda")
    x, trellis, suh, svh = _payloads(4, 1, torch.float16, device)
    got = call_exl3_gemm(
        x, trellis, suh, svh, mcg=False, mul1=False
    )
    ref = _reconstruct_ref(
        ext,
        x,
        trellis,
        suh,
        svh,
        4,
        mcg=False,
        mul1=False,
    )
    torch.cuda.synchronize()
    if not torch.allclose(got, ref, rtol=5e-2, atol=0.75):
        max_err = (got.float() - ref.float()).abs().max().item()
        pytest.fail(f"K4 slim parity failed: max_err={max_err}")


def test_k4_tcfold_gemv_matches_reconstruct(monkeypatch):
    monkeypatch.setenv("EXL3_GEMV_K4_TCFOLD", "1")
    monkeypatch.delenv("EXL3_GEMV_K4_ARITH", raising=False)
    monkeypatch.delenv("EXL3_GEMV_K4_SLIM", raising=False)
    monkeypatch.setenv("VLLM_EXL3_FORCE_COMPRESSED", "1")
    monkeypatch.delenv("VLLM_EXL3_FORCE_RECONSTRUCT", raising=False)
    ext = _ext()
    from vllm_exl3_sm86.ops import call_exl3_gemm

    device = torch.device("cuda")
    x, trellis, suh, svh = _payloads(4, 1, torch.float16, device)
    got = call_exl3_gemm(
        x, trellis, suh, svh, mcg=False, mul1=False
    )
    ref = _reconstruct_ref(
        ext,
        x,
        trellis,
        suh,
        svh,
        4,
        mcg=False,
        mul1=False,
    )
    torch.cuda.synchronize()
    # The fold replaces a*(lo+hi) with a*lo + a*hi, so the fp16 accumulator
    # takes twice as many terms per k-slice before the fp32 fold.
    if not torch.allclose(got, ref, rtol=5e-2, atol=0.75):
        max_err = (got.float() - ref.float()).abs().max().item()
        pytest.fail(f"K4 tensor-core fold parity failed: max_err={max_err}")


def test_k4_tcfold_is_opt_in(monkeypatch):
    """Default dispatch must be byte-identical to the unfolded kernel."""
    monkeypatch.delenv("EXL3_GEMV_K4_TCFOLD", raising=False)
    monkeypatch.delenv("EXL3_GEMV_K4_ARITH", raising=False)
    monkeypatch.delenv("EXL3_GEMV_K4_SLIM", raising=False)
    monkeypatch.setenv("VLLM_EXL3_FORCE_COMPRESSED", "1")
    ext = _ext()
    from vllm_exl3_sm86.ops import call_exl3_gemm

    device = torch.device("cuda")
    x, trellis, suh, svh = _payloads(4, 1, torch.float16, device)
    got = call_exl3_gemm(x, trellis, suh, svh, mcg=False, mul1=False)
    ref = _native_compressed(ext, x, trellis, suh, svh, False, False)
    torch.cuda.synchronize()
    if not torch.equal(got, ref):
        max_err = (got.float() - ref.float()).abs().max().item()
        pytest.fail(f"unfolded K4 default drifted: max_err={max_err}")


@pytest.mark.parametrize("bitrate", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("mcg,mul1", [(False, False), (True, False), (False, True)])
def test_plugin_compressed_matches_ext_gemm_all_codebooks(bitrate, mcg, mul1, monkeypatch):
    monkeypatch.setenv("VLLM_EXL3_FORCE_COMPRESSED", "1")
    monkeypatch.delenv("VLLM_EXL3_FORCE_RECONSTRUCT", raising=False)
    ext = _ext()
    from vllm_exl3_sm86.ops import call_exl3_gemm

    device = torch.device("cuda")
    x, trellis, suh, svh = _payloads(bitrate, 1, torch.float16, device)
    got = call_exl3_gemm(x, trellis, suh, svh, mcg=mcg, mul1=mul1)
    ref = _native_compressed(ext, x, trellis, suh, svh, mcg, mul1)
    torch.cuda.synchronize()
    if not torch.equal(got, ref):
        max_err = (got.float() - ref.float()).abs().max().item()
        pytest.fail(
            f"plugin vs ext bitrate={bitrate} mcg={mcg} mul1={mul1} M=1 max_err={max_err}"
        )


@pytest.mark.parametrize("bitrate", [3, 4, 6])
@pytest.mark.parametrize("m", [1, 8, 32, 128, 1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_plugin_compressed_matches_ext_gemm(bitrate, m, dtype, monkeypatch):
    monkeypatch.setenv("VLLM_EXL3_FORCE_COMPRESSED", "1")
    monkeypatch.delenv("VLLM_EXL3_FORCE_RECONSTRUCT", raising=False)
    ext = _ext()
    from vllm_exl3_sm86.ops import call_exl3_gemm

    device = torch.device("cuda")
    x, trellis, suh, svh = _payloads(bitrate, m, dtype, device)
    got = call_exl3_gemm(x, trellis, suh, svh, mcg=False, mul1=True)
    ref = _native_compressed(ext, x, trellis, suh, svh, False, True)
    torch.cuda.synchronize()
    if not torch.equal(got, ref) and not torch.allclose(got, ref, rtol=0, atol=0):
        max_err = (got - ref).abs().max().item()
        pytest.fail(
            f"plugin vs ext.exl3_gemm bitrate={bitrate} M={m} dtype={dtype}: "
            f"max_err={max_err}"
        )


@pytest.mark.parametrize("bitrate", [3, 4, 6])
@pytest.mark.parametrize("m", [1, 8, 32, 128, 1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_plugin_reconstruct_matches_hgemm(bitrate, m, dtype, monkeypatch):
    monkeypatch.setenv("VLLM_EXL3_FORCE_RECONSTRUCT", "1")
    monkeypatch.delenv("VLLM_EXL3_FORCE_COMPRESSED", raising=False)
    _ext()
    from vllm_exl3_sm86.ops import call_exl3_gemm
    from vllm_exl3_sm86.prefill import reconstruct_hgemm

    device = torch.device("cuda")
    x, trellis, suh, svh = _payloads(bitrate, m, dtype, device)
    got = call_exl3_gemm(x, trellis, suh, svh, mcg=False, mul1=True)
    ref = reconstruct_hgemm(
        x,
        trellis,
        suh,
        svh,
        mcg=False,
        mul1=True,
        fused=m >= FUSED_RECONSTRUCT_M,
    )
    torch.cuda.synchronize()
    if not torch.allclose(got, ref, rtol=1e-5, atol=1e-5):
        max_err = (got - ref).abs().max().item()
        pytest.fail(
            f"plugin vs reconstruct_hgemm bitrate={bitrate} M={m} dtype={dtype}: "
            f"max_err={max_err}"
        )


@pytest.mark.parametrize("bitrate", [3, 4, 6])
@pytest.mark.parametrize("m", [1, 8, 32, 128, 1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_plugin_matches_reconstructed_fp16(bitrate, m, dtype, monkeypatch):
    monkeypatch.setenv("VLLM_EXL3_FORCE_COMPRESSED", "1")
    monkeypatch.delenv("VLLM_EXL3_FORCE_RECONSTRUCT", raising=False)
    ext = _ext()
    from vllm_exl3_sm86.ops import call_exl3_gemm

    device = torch.device("cuda")
    x, trellis, suh, svh = _payloads(bitrate, m, dtype, device)
    got = call_exl3_gemm(x, trellis, suh, svh, mcg=False, mul1=True)
    ref = _reconstruct_ref(ext, x, trellis, suh, svh, bitrate)
    torch.cuda.synchronize()
    # GEMV (M=1) and fp16 GEMM vs reconstruct differ in reduction order.
    atol = 0.75 if m == 1 else 0.25
    if not torch.allclose(got, ref, rtol=5e-2, atol=atol):
        max_err = (got.float() - ref.float()).abs().max().item()
        pytest.fail(
            f"parity failed bitrate={bitrate} M={m} dtype={dtype}: max_err={max_err}"
        )


@pytest.mark.parametrize("m", [1, 2, 4])
def test_mgemm_matches_two_gemms(m, monkeypatch):
    monkeypatch.setenv("VLLM_EXL3_FORCE_COMPRESSED", "1")
    monkeypatch.delenv("VLLM_EXL3_FORCE_RECONSTRUCT", raising=False)
    ext = _ext()
    from vllm_exl3_sm86.ops import call_exl3_gemm, call_exl3_mgemm, ext_has_mgemm

    if not ext_has_mgemm():
        pytest.skip("exllamav3_ext does not export exl3_mgemm")
    device = torch.device("cuda")
    x, trellis0, suh0, svh0 = _payloads(4, m, torch.float16, device, seed=0)
    _, trellis1, suh1, svh1 = _payloads(4, m, torch.float16, device, seed=1)
    n = trellis0.shape[1] * TRELLIS_TILE
    k = x.shape[-1]
    ref = torch.cat(
        [
            call_exl3_gemm(x, trellis0, suh0, svh0, False, False),
            call_exl3_gemm(x, trellis1, suh1, svh1, False, False),
        ],
        dim=-1,
    )
    ptrs_trellis = torch.tensor(
        [int(trellis0.data_ptr()), int(trellis1.data_ptr())],
        dtype=torch.int64,
        device=device,
    )
    ptrs_suh = torch.tensor(
        [int(suh0.data_ptr()), int(suh1.data_ptr())],
        dtype=torch.int64,
        device=device,
    )
    ptrs_svh = torch.tensor(
        [int(svh0.data_ptr()), int(svh1.data_ptr())],
        dtype=torch.int64,
        device=device,
    )
    out = torch.empty((2, m, n), dtype=torch.float16, device=device)
    x_had = torch.empty((2, m, k), dtype=torch.float16, device=device)
    call_exl3_mgemm(
        x.view(1, m, k),
        ptrs_trellis,
        ptrs_suh,
        ptrs_svh,
        4,
        False,
        False,
        out,
        x_had,
    )
    got = torch.cat((out[0], out[1]), dim=-1)
    torch.cuda.synchronize()
    # mgemm autotunes a batched grid; two sequential gemms autotune
    # independently. Reduction-order noise is the same class as compressed
    # vs reconstruct (0.066 at M=1/2, 0.125 at M=4 on 3inst).
    if not torch.allclose(got, ref, rtol=5e-2, atol=0.25):
        max_err = (got.float() - ref.float()).abs().max().item()
        pytest.fail(f"mgemm vs two gemms M={m} max_err={max_err}")
