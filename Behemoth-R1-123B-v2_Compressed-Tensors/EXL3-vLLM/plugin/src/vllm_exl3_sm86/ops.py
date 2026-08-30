"""Opaque vLLM custom op around exllamav3_ext.exl3_gemm / reconstruct+hgemm."""

from __future__ import annotations

import ctypes
import importlib
import os
import sys
from typing import Any

import torch

from .constants import DEFAULT_RECONSTRUCT_M, FUSED_RECONSTRUCT_M, TRELLIS_TILE
from .nvtx import nvtx_range
from .prefill import reconstruct_hgemm, reconstruct_threshold_for_shape

_EXL3_EXT: Any | None = None
_OP_REGISTERED = False


def _load_exl3_ext() -> Any:
    global _EXL3_EXT
    if _EXL3_EXT is not None:
        return _EXL3_EXT

    shim = os.environ.get("VLLM_EXL3_ABI_SHIM")
    if shim:
        ctypes.CDLL(shim, mode=ctypes.RTLD_GLOBAL)

    ext_path = os.environ.get("VLLM_EXL3_EXT_PATH")
    if ext_path:
        search_dir = ext_path if os.path.isdir(ext_path) else os.path.dirname(ext_path)
        if search_dir and search_dir not in sys.path:
            sys.path.insert(0, search_dir)

    try:
        ext = importlib.import_module("exllamav3_ext")
    except Exception as exc:
        hint = (
            "Set VLLM_EXL3_EXT_PATH to the directory containing "
            "exllamav3_ext*.so (and VLLM_EXL3_ABI_SHIM when the local "
            "PyTorch ABI shim is required)."
        )
        raise RuntimeError(f"Unable to import exllamav3_ext. {hint}") from exc

    if not hasattr(ext, "exl3_gemm"):
        raise RuntimeError(
            "The imported exllamav3_ext does not export exl3_gemm; rebuild "
            "with scripts/build_exllamav3_ext.sh against the vLLM ABI."
        )
    _EXL3_EXT = ext
    return ext


def _force_compressed() -> bool:
    return os.environ.get("VLLM_EXL3_FORCE_COMPRESSED", "").strip() in {
        "1",
        "true",
        "yes",
    }


def _force_reconstruct() -> bool:
    return os.environ.get("VLLM_EXL3_FORCE_RECONSTRUCT", "").strip() in {
        "1",
        "true",
        "yes",
    }


def _should_reconstruct(m: int, k: int, n: int, bitrate: int) -> bool:
    if _force_compressed():
        return False
    if _force_reconstruct():
        return True
    threshold = reconstruct_threshold_for_shape(k, n, bitrate)
    if threshold is None:
        threshold = int(os.environ.get("VLLM_EXL3_RECONSTRUCT_M", DEFAULT_RECONSTRUCT_M))
    return m > threshold


def _compressed_gemm(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: bool,
    mul1: bool,
    out: torch.Tensor | None = None,
    x_had: torch.Tensor | None = None,
) -> torch.Tensor:
    ext = _load_exl3_ext()
    n = int(trellis.shape[1] * TRELLIS_TILE)
    if out is None:
        out = torch.empty((x.shape[0], n), dtype=torch.float16, device=x.device)
    if x_had is None:
        x_had = torch.empty_like(x)
    with nvtx_range("exl3.gemm"):
        ext.exl3_gemm(
            x,
            trellis,
            out,
            suh,
            x_had,
            svh,
            -1,
            mcg,
            mul1,
            0,
        )
    return out


def exl3_gemm_impl(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: bool,
    mul1: bool,
    out: torch.Tensor | None = None,
    x_had: torch.Tensor | None = None,
) -> torch.Tensor:
    """M-dependent dispatch lives inside the opaque op so torch.compile cannot
    specialize a prefill branch and reuse it for decode."""
    m = int(x.shape[0])
    k = int(trellis.shape[0] * TRELLIS_TILE)
    n = int(trellis.shape[1] * TRELLIS_TILE)
    bitrate = int(trellis.shape[2] // TRELLIS_TILE)
    if _should_reconstruct(m, k, n, bitrate):
        fused = m >= FUSED_RECONSTRUCT_M
        with nvtx_range("exl3.reconstruct"):
            y = reconstruct_hgemm(
                x, trellis, suh, svh, mcg=mcg, mul1=mul1, fused=fused
            )
            if out is not None:
                out.copy_(y)
                return out
            return y
    return _compressed_gemm(x, trellis, suh, svh, mcg, mul1, out=out, x_had=x_had)


def exl3_gemm_fake(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: bool,
    mul1: bool,
) -> torch.Tensor:
    del suh, svh, mcg, mul1
    return torch.empty(
        (x.shape[0], trellis.shape[1] * TRELLIS_TILE),
        dtype=torch.float16,
        device=x.device,
    )


def ext_has_mgemm() -> bool:
    try:
        return hasattr(_load_exl3_ext(), "exl3_mgemm")
    except Exception:
        return False


def exl3_mgemm_impl(
    x: torch.Tensor,
    ptrs_trellis: torch.Tensor,
    ptrs_suh: torch.Tensor,
    ptrs_svh: torch.Tensor,
    bitrate: int,
    mcg: bool,
    mul1: bool,
    out: torch.Tensor,
    x_had: torch.Tensor,
) -> None:
    """One input, two outputs: A is (1, M, K), C is (2, M, N), A_had is (2, M, K)."""
    ext = _load_exl3_ext()
    if x.dim() == 2:
        x = x.unsqueeze(0)
    with nvtx_range("exl3.mgemm"):
        ext.exl3_mgemm(
            x,
            ptrs_trellis,
            out,
            ptrs_suh,
            x_had,
            ptrs_svh,
            None,
            None,
            int(bitrate),
            -1,
            bool(mcg),
            bool(mul1),
            -1,
            -1,
            0,
            1,
            None,
            None,
        )


def _packed_ws_for_m(
    m: int,
    outs: list[torch.Tensor],
    xhads: list[torch.Tensor],
    packeds: list[torch.Tensor],
    xpads: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    for out, xhad, packed, xpad in zip(outs, xhads, packeds, xpads):
        if int(out.shape[1]) == m:
            return out, xhad, packed, xpad
    return None


def exl3_packed_pair_impl(
    x: torch.Tensor,
    trellis0: torch.Tensor,
    suh0: torch.Tensor,
    svh0: torch.Tensor,
    trellis1: torch.Tensor,
    suh1: torch.Tensor,
    svh1: torch.Tensor,
    mcg: bool,
    mul1: bool,
    bitrate: int,
    n0: int,
    n1: int,
    ptrs_trellis: torch.Tensor,
    ptrs_suh: torch.Tensor,
    ptrs_svh: torch.Tensor,
    outs: list[torch.Tensor],
    xhads: list[torch.Tensor],
    packeds: list[torch.Tensor],
    xpads: list[torch.Tensor],
) -> torch.Tensor:
    """Fuse a packed pair at capture sizes; two GEMMs otherwise.

    Token-count dispatch stays inside this opaque impl so torch.compile cannot
    specialize the model to the profile-run length (8192).
    """
    m = int(x.shape[0])
    packed_k = int(trellis0.shape[0] * TRELLIS_TILE)
    if x.shape[-1] > packed_k:
        raise ValueError(
            f"EXL3 input width {x.shape[-1]} exceeds packed K={packed_k}"
        )
    decode_ws = _packed_ws_for_m(m, outs, xhads, packeds, xpads)
    if decode_ws is not None:
        out, xhad, packed, xpad = decode_ws
        xin = x
        if x.shape[-1] < packed_k:
            xpad.zero_()
            xpad[:, : x.shape[-1]].copy_(x)
            xin = xpad
        exl3_mgemm_impl(
            xin.view(1, m, packed_k),
            ptrs_trellis,
            ptrs_suh,
            ptrs_svh,
            bitrate,
            mcg,
            mul1,
            out,
            xhad,
        )
        packed[:, :n0].copy_(out[0, :, :n0])
        packed[:, n0 : n0 + n1].copy_(out[1, :, :n1])
        return packed[:, : n0 + n1].clone()
    if x.shape[-1] < packed_k:
        x = torch.nn.functional.pad(x, (0, packed_k - x.shape[-1]))
    y0 = exl3_gemm_impl(x, trellis0, suh0, svh0, mcg, mul1)
    y1 = exl3_gemm_impl(x, trellis1, suh1, svh1, mcg, mul1)
    return torch.cat((y0[..., :n0], y1[..., :n1]), dim=-1)


def exl3_packed_pair_fake(
    x: torch.Tensor,
    trellis0: torch.Tensor,
    suh0: torch.Tensor,
    svh0: torch.Tensor,
    trellis1: torch.Tensor,
    suh1: torch.Tensor,
    svh1: torch.Tensor,
    mcg: bool,
    mul1: bool,
    bitrate: int,
    n0: int,
    n1: int,
    ptrs_trellis: torch.Tensor,
    ptrs_suh: torch.Tensor,
    ptrs_svh: torch.Tensor,
    outs: list[torch.Tensor],
    xhads: list[torch.Tensor],
    packeds: list[torch.Tensor],
    xpads: list[torch.Tensor],
) -> torch.Tensor:
    del (
        trellis0,
        suh0,
        svh0,
        trellis1,
        suh1,
        svh1,
        mcg,
        mul1,
        bitrate,
        ptrs_trellis,
        ptrs_suh,
        ptrs_svh,
        outs,
        xhads,
        packeds,
        xpads,
    )
    return torch.empty(
        (x.shape[0], int(n0) + int(n1)),
        dtype=torch.float16,
        device=x.device,
    )


def register_custom_op() -> None:
    """Register vllm::exl3_gemm, exl3_gemm_out, and exl3_mgemm_out once."""
    global _OP_REGISTERED
    has_vllm = hasattr(torch.ops, "vllm")
    has_gemm = has_vllm and hasattr(torch.ops.vllm, "exl3_gemm")
    has_out = has_vllm and hasattr(torch.ops.vllm, "exl3_gemm_out")
    has_mgemm = has_vllm and hasattr(torch.ops.vllm, "exl3_mgemm_out")
    has_packed = has_vllm and hasattr(torch.ops.vllm, "exl3_packed_pair")
    if _OP_REGISTERED and has_gemm and has_out and has_mgemm and has_packed:
        return

    if not has_gemm:

        @torch.library.custom_op(
            "vllm::exl3_gemm",
            mutates_args=(),
            device_types="cuda",
        )
        def _exl3_gemm(
            x: torch.Tensor,
            trellis: torch.Tensor,
            suh: torch.Tensor,
            svh: torch.Tensor,
            mcg: bool,
            mul1: bool,
        ) -> torch.Tensor:
            return exl3_gemm_impl(x, trellis, suh, svh, mcg, mul1)

        @_exl3_gemm.register_fake
        def _exl3_gemm_fake(
            x: torch.Tensor,
            trellis: torch.Tensor,
            suh: torch.Tensor,
            svh: torch.Tensor,
            mcg: bool,
            mul1: bool,
        ) -> torch.Tensor:
            return exl3_gemm_fake(x, trellis, suh, svh, mcg, mul1)

    if not has_out:

        @torch.library.custom_op(
            "vllm::exl3_gemm_out",
            mutates_args=("out", "x_had"),
            device_types="cuda",
        )
        def _exl3_gemm_out(
            x: torch.Tensor,
            trellis: torch.Tensor,
            suh: torch.Tensor,
            svh: torch.Tensor,
            mcg: bool,
            mul1: bool,
            out: torch.Tensor,
            x_had: torch.Tensor,
        ) -> None:
            exl3_gemm_impl(x, trellis, suh, svh, mcg, mul1, out=out, x_had=x_had)

        @_exl3_gemm_out.register_fake
        def _exl3_gemm_out_fake(
            x: torch.Tensor,
            trellis: torch.Tensor,
            suh: torch.Tensor,
            svh: torch.Tensor,
            mcg: bool,
            mul1: bool,
            out: torch.Tensor,
            x_had: torch.Tensor,
        ) -> None:
            del x, trellis, suh, svh, mcg, mul1, out, x_had

    if not has_mgemm:

        @torch.library.custom_op(
            "vllm::exl3_mgemm_out",
            mutates_args=("out", "x_had"),
            device_types="cuda",
        )
        def _exl3_mgemm_out(
            x: torch.Tensor,
            ptrs_trellis: torch.Tensor,
            ptrs_suh: torch.Tensor,
            ptrs_svh: torch.Tensor,
            bitrate: int,
            mcg: bool,
            mul1: bool,
            out: torch.Tensor,
            x_had: torch.Tensor,
        ) -> None:
            exl3_mgemm_impl(
                x,
                ptrs_trellis,
                ptrs_suh,
                ptrs_svh,
                bitrate,
                mcg,
                mul1,
                out,
                x_had,
            )

        @_exl3_mgemm_out.register_fake
        def _exl3_mgemm_out_fake(
            x: torch.Tensor,
            ptrs_trellis: torch.Tensor,
            ptrs_suh: torch.Tensor,
            ptrs_svh: torch.Tensor,
            bitrate: int,
            mcg: bool,
            mul1: bool,
            out: torch.Tensor,
            x_had: torch.Tensor,
        ) -> None:
            del (
                x,
                ptrs_trellis,
                ptrs_suh,
                ptrs_svh,
                bitrate,
                mcg,
                mul1,
                out,
                x_had,
            )

    if not has_packed:

        @torch.library.custom_op(
            "vllm::exl3_packed_pair",
            mutates_args=("outs", "xhads", "packeds", "xpads"),
            device_types="cuda",
        )
        def _exl3_packed_pair(
            x: torch.Tensor,
            trellis0: torch.Tensor,
            suh0: torch.Tensor,
            svh0: torch.Tensor,
            trellis1: torch.Tensor,
            suh1: torch.Tensor,
            svh1: torch.Tensor,
            mcg: bool,
            mul1: bool,
            bitrate: int,
            n0: int,
            n1: int,
            ptrs_trellis: torch.Tensor,
            ptrs_suh: torch.Tensor,
            ptrs_svh: torch.Tensor,
            outs: list[torch.Tensor],
            xhads: list[torch.Tensor],
            packeds: list[torch.Tensor],
            xpads: list[torch.Tensor],
        ) -> torch.Tensor:
            return exl3_packed_pair_impl(
                x,
                trellis0,
                suh0,
                svh0,
                trellis1,
                suh1,
                svh1,
                mcg,
                mul1,
                bitrate,
                n0,
                n1,
                ptrs_trellis,
                ptrs_suh,
                ptrs_svh,
                outs,
                xhads,
                packeds,
                xpads,
            )

        @_exl3_packed_pair.register_fake
        def _exl3_packed_pair_fake(
            x: torch.Tensor,
            trellis0: torch.Tensor,
            suh0: torch.Tensor,
            svh0: torch.Tensor,
            trellis1: torch.Tensor,
            suh1: torch.Tensor,
            svh1: torch.Tensor,
            mcg: bool,
            mul1: bool,
            bitrate: int,
            n0: int,
            n1: int,
            ptrs_trellis: torch.Tensor,
            ptrs_suh: torch.Tensor,
            ptrs_svh: torch.Tensor,
            outs: list[torch.Tensor],
            xhads: list[torch.Tensor],
            packeds: list[torch.Tensor],
            xpads: list[torch.Tensor],
        ) -> torch.Tensor:
            return exl3_packed_pair_fake(
                x,
                trellis0,
                suh0,
                svh0,
                trellis1,
                suh1,
                svh1,
                mcg,
                mul1,
                bitrate,
                n0,
                n1,
                ptrs_trellis,
                ptrs_suh,
                ptrs_svh,
                outs,
                xhads,
                packeds,
                xpads,
            )

    _OP_REGISTERED = True



def call_exl3_gemm(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: bool,
    mul1: bool,
    out: torch.Tensor | None = None,
    x_had: torch.Tensor | None = None,
) -> torch.Tensor:
    register_custom_op()
    if out is not None and x_had is not None:
        if hasattr(torch.ops, "vllm") and hasattr(torch.ops.vllm, "exl3_gemm_out"):
            torch.ops.vllm.exl3_gemm_out(
                x, trellis, suh, svh, mcg, mul1, out, x_had
            )
            return out
        return exl3_gemm_impl(
            x, trellis, suh, svh, mcg, mul1, out=out, x_had=x_had
        )
    if hasattr(torch.ops, "vllm") and hasattr(torch.ops.vllm, "exl3_gemm"):
        return torch.ops.vllm.exl3_gemm(x, trellis, suh, svh, mcg, mul1)
    return exl3_gemm_impl(x, trellis, suh, svh, mcg, mul1)


def call_exl3_mgemm(
    x: torch.Tensor,
    ptrs_trellis: torch.Tensor,
    ptrs_suh: torch.Tensor,
    ptrs_svh: torch.Tensor,
    bitrate: int,
    mcg: bool,
    mul1: bool,
    out: torch.Tensor,
    x_had: torch.Tensor,
) -> torch.Tensor:
    register_custom_op()
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if hasattr(torch.ops, "vllm") and hasattr(torch.ops.vllm, "exl3_mgemm_out"):
        torch.ops.vllm.exl3_mgemm_out(
            x,
            ptrs_trellis,
            ptrs_suh,
            ptrs_svh,
            int(bitrate),
            mcg,
            mul1,
            out,
            x_had,
        )
        return out
    exl3_mgemm_impl(
        x,
        ptrs_trellis,
        ptrs_suh,
        ptrs_svh,
        bitrate,
        mcg,
        mul1,
        out,
        x_had,
    )
    return out


def call_exl3_packed_pair(
    x: torch.Tensor,
    trellis0: torch.Tensor,
    suh0: torch.Tensor,
    svh0: torch.Tensor,
    trellis1: torch.Tensor,
    suh1: torch.Tensor,
    svh1: torch.Tensor,
    mcg: bool,
    mul1: bool,
    bitrate: int,
    n0: int,
    n1: int,
    ptrs_trellis: torch.Tensor,
    ptrs_suh: torch.Tensor,
    ptrs_svh: torch.Tensor,
    outs: list[torch.Tensor],
    xhads: list[torch.Tensor],
    packeds: list[torch.Tensor],
    xpads: list[torch.Tensor],
) -> torch.Tensor:
    register_custom_op()
    args = (
        x,
        trellis0,
        suh0,
        svh0,
        trellis1,
        suh1,
        svh1,
        mcg,
        mul1,
        int(bitrate),
        int(n0),
        int(n1),
        ptrs_trellis,
        ptrs_suh,
        ptrs_svh,
        outs,
        xhads,
        packeds,
        xpads,
    )
    if hasattr(torch.ops, "vllm") and hasattr(torch.ops.vllm, "exl3_packed_pair"):
        return torch.ops.vllm.exl3_packed_pair(*args)
    return exl3_packed_pair_impl(*args)
