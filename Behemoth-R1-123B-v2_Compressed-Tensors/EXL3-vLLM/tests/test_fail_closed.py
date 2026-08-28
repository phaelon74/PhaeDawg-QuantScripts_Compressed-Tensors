from __future__ import annotations

import pytest

from vllm_exl3_sm86.graph import graphs_allowed


def test_graphs_fail_closed_without_opt_in(monkeypatch):
    monkeypatch.delenv("VLLM_EXL3_ALLOW_GRAPHS", raising=False)
    assert graphs_allowed() is False


def test_empty_storage_without_fallback_is_documented():
    pytest.importorskip("vllm")
    from vllm_exl3_sm86.config import Exl3Config

    cfg = Exl3Config()
    cfg._eager_checked = True
    with pytest.raises(ValueError, match="tensor_storage is empty"):
        cfg.get_quant_method(object(), "x")
