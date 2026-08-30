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


@contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    if not nvtx_enabled() or not torch.cuda.is_available():
        yield
        return
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()
