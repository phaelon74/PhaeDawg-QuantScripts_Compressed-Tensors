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


def test_nvtx_range_noops_while_compiling(monkeypatch):
    monkeypatch.setenv("VLLM_EXL3_NVTX", "1")

    def fake_push(_name: str) -> int:
        raise AssertionError("range_push must not run under Dynamo")

    monkeypatch.setattr("vllm_exl3_sm86.nvtx._is_compiling", lambda: True)
    monkeypatch.setattr("torch.cuda.nvtx.range_push", fake_push)
    with nvtx_range("exl3.apply"):
        pass
