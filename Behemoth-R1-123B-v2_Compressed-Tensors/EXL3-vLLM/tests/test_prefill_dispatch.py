from __future__ import annotations

import json

import pytest

from vllm_exl3_sm86.constants import DEFAULT_RECONSTRUCT_M
from vllm_exl3_sm86.prefill import load_crossover_table, reconstruct_threshold_for_shape


def test_default_threshold_without_table(monkeypatch, tmp_path):
    monkeypatch.delenv("VLLM_EXL3_CROSSOVER_JSON", raising=False)
    monkeypatch.delenv("VLLM_EXL3_RECONSTRUCT_M", raising=False)
    import vllm_exl3_sm86.prefill as prefill

    prefill._THRESHOLDS = None
    missing = tmp_path / "missing.json"
    monkeypatch.setenv("VLLM_EXL3_CROSSOVER_JSON", str(missing))
    assert reconstruct_threshold_for_shape(12288, 3072, 3) is None
    from vllm_exl3_sm86.prefill import default_threshold

    assert default_threshold() == DEFAULT_RECONSTRUCT_M


def test_per_shape_sm86_table(monkeypatch, tmp_path):
    table = {
        "thresholds": [
            {"k": 12288, "n": 3072, "bitrate": 3, "m": 96},
            {"k": 12288, "n": 8192, "bitrate": 6, "m": 256},
        ]
    }
    path = tmp_path / "crossover.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    monkeypatch.setenv("VLLM_EXL3_CROSSOVER_JSON", str(path))
    import vllm_exl3_sm86.prefill as prefill

    prefill._THRESHOLDS = None
    load_crossover_table()
    assert reconstruct_threshold_for_shape(12288, 3072, 3) == 96
    assert reconstruct_threshold_for_shape(12288, 8192, 6) == 256
    assert reconstruct_threshold_for_shape(7168, 12288, 4) is None
