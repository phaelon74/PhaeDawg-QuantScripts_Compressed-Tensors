from __future__ import annotations

from vllm_exl3_sm86.nvtx import nvtx_enabled, nvtx_range


def test_nvtx_off_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_EXL3_NVTX", raising=False)
    assert nvtx_enabled() is False


def test_nvtx_on(monkeypatch):
    monkeypatch.setenv("VLLM_EXL3_NVTX", "1")
    assert nvtx_enabled() is True


def test_nvtx_range_noops_when_disabled(monkeypatch):
    monkeypatch.delenv("VLLM_EXL3_NVTX", raising=False)
    with nvtx_range("exl3.test"):
        pass
