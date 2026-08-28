from __future__ import annotations

from types import SimpleNamespace

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
def test_prewarm_touches_capture_sizes():
    import torch
    from vllm_exl3_sm86.graph import prewarm_behemoth_tp4

    try:
        receipts = prewarm_behemoth_tp4(torch.device("cuda"), capture_sizes=(1,), bitrates=(3,))
    except RuntimeError as exc:
        pytest.skip(str(exc))
    assert receipts
    assert {row["m"] for row in receipts} == {1}
