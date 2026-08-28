from __future__ import annotations

import pytest

from vllm_exl3_sm86.constants import REQUIRED_TORCH_PREFIX, REQUIRED_VLLM_VERSION
from vllm_exl3_sm86.version_guard import check_runtime


def test_skip_guard(monkeypatch):
    monkeypatch.setenv("VLLM_EXL3_SKIP_VERSION_GUARD", "1")
    check_runtime(require_cuda=False)


def test_mismatch_raises(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.delenv("VLLM_EXL3_SKIP_VERSION_GUARD", raising=False)
    monkeypatch.delenv("VLLM_EXL3_ALLOW_VLLM_DRIFT", raising=False)

    import vllm_exl3_sm86.version_guard as vg

    class FakeVllm:
        __version__ = "not-the-pin"

    monkeypatch.setattr(vg, "check_runtime", check_runtime)
    import sys
    from types import ModuleType

    fake = ModuleType("vllm")
    fake.__version__ = "0.0.0"
    monkeypatch.setitem(sys.modules, "vllm", fake)
    with pytest.raises(RuntimeError, match="ABI mismatch"):
        check_runtime(require_cuda=False)

    assert REQUIRED_VLLM_VERSION.startswith("0.1.dev")
    assert REQUIRED_TORCH_PREFIX == "2.9.1"
