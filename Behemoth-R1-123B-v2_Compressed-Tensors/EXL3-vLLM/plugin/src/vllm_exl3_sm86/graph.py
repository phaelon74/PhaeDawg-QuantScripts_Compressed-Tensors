"""CUDA-graph prewarm and workspace pinning for capture sizes 1/2/4.

Graphs stay disabled until:
- eager TP4 continuous batching is correct,
- every (M, K, N, bitrate, codebook, dtype) used by capture sizes is prewarmed,
- INT8 / reconstruct / cuBLAS workspaces have stable pointers,
- 100K capture/replay stress passes.
"""

from __future__ import annotations

import os
from typing import Iterable

import torch

from .constants import (
    BEHEMOTH_TP4_SHAPES,
    CODEBOOK_FLAGS,
    DECODER_BITRATES,
    GRAPH_CAPTURE_SIZES,
    HEAD_BITRATE,
)
from .ops import call_exl3_gemm, call_exl3_mgemm, ext_has_mgemm
from .prefill import preallocate_workspaces


def graphs_allowed() -> bool:
    return os.environ.get("VLLM_EXL3_ALLOW_GRAPHS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _dummy_payloads(
    device: torch.device,
    k: int,
    n: int,
    bitrate: int,
    m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    trellis = torch.zeros(
        (k // 16, n // 16, 16 * bitrate), dtype=torch.int16, device=device
    )
    suh = torch.ones(k, dtype=torch.float16, device=device)
    svh = torch.ones(n, dtype=torch.float16, device=device)
    x = torch.zeros((m, k), dtype=torch.float16, device=device)
    return x, trellis, suh, svh


def prewarm_behemoth_tp4(
    device: torch.device | None = None,
    *,
    capture_sizes: Iterable[int] = GRAPH_CAPTURE_SIZES,
    bitrates: Iterable[int] = (*DECODER_BITRATES, HEAD_BITRATE),
    codebooks: Iterable[tuple[bool, bool]] = CODEBOOK_FLAGS,
) -> list[dict[str, int | bool]]:
    """Touch every production shape so autotune cannot run during capture.

    Autotune keys include codebook flags. ArtusDev 4.25 is implicit 3inst
    (`mcg=False`, `mul1=False`); prewarming only `mul1=True` leaves the
    serving kernels cold.
    """
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to prewarm EXL3 kernels")
        device = torch.device("cuda", torch.cuda.current_device())
    shapes = list(BEHEMOTH_TP4_SHAPES.values())
    preallocate_workspaces(device, shapes)
    receipts: list[dict[str, int | bool]] = []
    for name, (k, n) in BEHEMOTH_TP4_SHAPES.items():
        if name == "lm_head":
            shape_bitrates = (HEAD_BITRATE,)
        else:
            requested = tuple(dict.fromkeys(int(b) for b in bitrates))
            shape_bitrates = requested or tuple(DECODER_BITRATES)
        for bitrate in shape_bitrates:
            for mcg, mul1 in codebooks:
                for m in capture_sizes:
                    x, trellis, suh, svh = _dummy_payloads(device, k, n, bitrate, m)
                    call_exl3_gemm(x, trellis, suh, svh, mcg=mcg, mul1=mul1)
                    receipts.append(
                        {
                            "name": name,
                            "k": k,
                            "n": n,
                            "bitrate": bitrate,
                            "m": int(m),
                            "mcg": bool(mcg),
                            "mul1": bool(mul1),
                        }
                    )
    if ext_has_mgemm():
        k, n = BEHEMOTH_TP4_SHAPES["gate_proj"]
        requested = tuple(dict.fromkeys(int(b) for b in bitrates)) or (4,)
        for bitrate in requested:
            for mcg, mul1 in codebooks:
                for m in capture_sizes:
                    x, trellis0, suh0, svh0 = _dummy_payloads(device, k, n, bitrate, m)
                    _, trellis1, suh1, svh1 = _dummy_payloads(device, k, n, bitrate, m)
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
                    out = torch.empty((2, int(m), n), dtype=torch.float16, device=device)
                    x_had = torch.empty((2, int(m), k), dtype=torch.float16, device=device)
                    call_exl3_mgemm(
                        x.view(1, int(m), k),
                        ptrs_trellis,
                        ptrs_suh,
                        ptrs_svh,
                        int(bitrate),
                        bool(mcg),
                        bool(mul1),
                        out,
                        x_had,
                    )
                    receipts.append(
                        {
                            "name": "gate_up_proj",
                            "k": k,
                            "n": n,
                            "bitrate": bitrate,
                            "m": int(m),
                            "mcg": bool(mcg),
                            "mul1": bool(mul1),
                            "mgemm": True,
                        }
                    )
    torch.cuda.synchronize(device)
    return receipts
