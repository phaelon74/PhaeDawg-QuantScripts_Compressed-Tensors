"""Optional NVTX ranges for nsys. Off unless VLLM_EXL3_NVTX=1."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import torch


def nvtx_enabled() -> bool:
    return os.environ.get("VLLM_EXL3_NVTX", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _is_compiling() -> bool:
    is_compiling = getattr(torch.compiler, "is_compiling", None)
    return bool(is_compiling()) if callable(is_compiling) else False


@contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    # range_push returns an int. Dynamo/AOT cannot put that in the FX graph,
    # so NVTX must stay out of compiled apply() even when the env is on.
    # Check compiling first so Dynamo dead-strips the push/pop branch.
    if _is_compiling() or not nvtx_enabled() or not torch.cuda.is_available():
        yield
        return
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()
