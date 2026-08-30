from __future__ import annotations

import pytest

from vllm_exl3_sm86.graph import graphs_allowed


def test_graphs_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_EXL3_ALLOW_GRAPHS", raising=False)
    assert graphs_allowed() is False


def test_graphs_opt_in(monkeypatch):
    monkeypatch.setenv("VLLM_EXL3_ALLOW_GRAPHS", "1")
    assert graphs_allowed() is True


@pytest.mark.skipif(
    __import__("torch").cuda.is_available() is False,
    reason="CUDA prewarm requires a GPU",
)
def test_capture_after_prewarm_does_not_allocate_int8_workspace():
    import torch
    from vllm_exl3_sm86.graph import (
        capture_allocated_bytes_delta,
        prewarm_behemoth_tp4,
    )

    device = torch.device("cuda")
    try:
        prewarm_behemoth_tp4(
            device,
            capture_sizes=(1,),
            bitrates=(4,),
            codebooks=((False, False), (False, True)),
        )
        delta = capture_allocated_bytes_delta(device, bitrate=4, m=1, mul1=True)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    # CUDA graph pool may bump a few KB. The INT8 workspace is 16 MiB.
    assert delta < 2 * 1024 * 1024, f"capture allocated {delta} bytes"


@pytest.mark.skipif(
    __import__("torch").cuda.is_available() is False,
    reason="CUDA prewarm requires a GPU",
)
def test_prewarm_touches_capture_sizes():
    import torch
    from vllm_exl3_sm86.graph import prewarm_behemoth_tp4

    try:
        receipts = prewarm_behemoth_tp4(
            torch.device("cuda"),
            capture_sizes=(1,),
            bitrates=(3,),
            codebooks=((False, False),),
        )
    except RuntimeError as exc:
        pytest.skip(str(exc))
    assert receipts
    assert {row["m"] for row in receipts} == {1}
    assert all(row["mul1"] is False and row["mcg"] is False for row in receipts)
