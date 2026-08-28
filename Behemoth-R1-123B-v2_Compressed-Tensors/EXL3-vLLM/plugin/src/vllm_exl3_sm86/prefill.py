"""Synchronous reconstruct-one-weight + hgemm/cuBLAS for large-M prefill.

Never cache reconstructed model weights. One bounded workspace is reused per
device and grown to the largest local (K, N) seen.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from .constants import (
    DEFAULT_RECONSTRUCT_M,
    HADAMARD_BLOCK,
    MAX_RECONSTRUCT_SLICE_N,
    TRELLIS_TILE,
)

_WORKSPACES: dict[tuple[str, int], torch.Tensor] = {}
_HAD_SCRATCH: dict[tuple[str, int], torch.Tensor] = {}
_THRESHOLDS: dict[tuple[int, int, int], int] | None = None


def _ext() -> Any:
    from .ops import _load_exl3_ext

    return _load_exl3_ext()


def load_crossover_table() -> dict[tuple[int, int, int], int]:
    global _THRESHOLDS
    if _THRESHOLDS is not None:
        return _THRESHOLDS
    table: dict[tuple[int, int, int], int] = {}
    path = os.environ.get("VLLM_EXL3_CROSSOVER_JSON", "")
    if not path:
        here = Path(__file__).resolve()
        default = here.parents[3] / "manifests" / "sm86_crossover.json"
        path = str(default) if default.is_file() else ""
    if path and Path(path).is_file():
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        for row in raw.get("thresholds", []):
            m = row.get("m")
            if m is None:
                continue
            table[(int(row["k"]), int(row["n"]), int(row["bitrate"]))] = int(m)
    _THRESHOLDS = table
    return table


def reconstruct_threshold_for_shape(k: int, n: int, bitrate: int) -> int | None:
    table = load_crossover_table()
    return table.get((k, n, bitrate))


def _workspace(device: torch.device, rows: int, cols: int) -> torch.Tensor:
    key = (str(device), int(device.index or 0))
    need = rows * cols
    buf = _WORKSPACES.get(key)
    if buf is None or buf.numel() < need:
        buf = torch.empty(need, dtype=torch.float16, device=device)
        _WORKSPACES[key] = buf
    return buf[:need].view(rows, cols)


def _had_scratch(x: torch.Tensor) -> torch.Tensor:
    key = (str(x.device), int(x.device.index or 0))
    buf = _HAD_SCRATCH.get(key)
    if buf is None or buf.shape != x.shape or buf.dtype != x.dtype:
        buf = torch.empty_like(x)
        _HAD_SCRATCH[key] = buf
    return buf


def preallocate_workspaces(
    device: torch.device, shapes: list[tuple[int, int]]
) -> None:
    """Allocate the max reconstruct workspace before CUDA-graph capture."""
    max_k = max(k for k, _n in shapes)
    max_n = max(n for _k, n in shapes)
    _workspace(device, max_k, min(max_n, MAX_RECONSTRUCT_SLICE_N))


def reconstruct_hgemm(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    *,
    mcg: bool,
    mul1: bool,
    fused: bool,
) -> torch.Tensor:
    """Reconstruct one local weight into a reusable workspace, then hgemm.

    Auxiliary-stream prefetch stays disabled unless profiling proves a benefit
    without pointer/event hazards.
    """
    ext = _ext()
    rows = int(x.shape[0])
    k = int(trellis.shape[0] * TRELLIS_TILE)
    n = int(trellis.shape[1] * TRELLIS_TILE)
    bitrate = int(trellis.shape[2] // TRELLIS_TILE)
    if k % HADAMARD_BLOCK or n % HADAMARD_BLOCK:
        raise ValueError(f"EXL3 reconstruct requires 128-aligned K/N, got {k}x{n}")

    y = torch.empty((rows, n), dtype=torch.float16, device=x.device)
    use_fused = bool(fused) and k % 128 == 0 and n % 128 == 0
    if use_fused:
        xh = x
    else:
        xh = _had_scratch(x)
        ext.had_r_128(x, xh, suh, None, 1.0)

    if n <= MAX_RECONSTRUCT_SLICE_N:
        w = _workspace(x.device, k, n)
        if use_fused:
            ext.reconstruct_had_slice(w, trellis, suh, svh, bitrate, mcg, mul1, 0)
        else:
            ext.reconstruct(w, trellis, bitrate, mcg, mul1)
        ext.hgemm(xh, w, y)
    else:
        w = _workspace(x.device, k, MAX_RECONSTRUCT_SLICE_N)
        for n_start in range(0, n, MAX_RECONSTRUCT_SLICE_N):
            n_end = min(n_start + MAX_RECONSTRUCT_SLICE_N, n)
            view = w[:, : n_end - n_start]
            if use_fused:
                ext.reconstruct_had_slice(
                    view, trellis, suh, svh[n_start:], bitrate, mcg, mul1, n_start
                )
            else:
                ext.reconstruct_slice(view, trellis, bitrate, mcg, mul1, n_start)
            ext.hgemm(xh, view, y[:, n_start:n_end])

    if not use_fused:
        ext.had_r_128(y, y, None, svh, 1.0)
    return y


def default_threshold() -> int:
    return int(os.environ.get("VLLM_EXL3_RECONSTRUCT_M", DEFAULT_RECONSTRUCT_M))
