from __future__ import annotations

import pytest

from vllm_exl3_sm86.constants import GRAPH_CAPTURE_SIZES, DECODER_BITRATES
from vllm_exl3_sm86.graph import graphs_allowed


def test_capture_sizes_are_1_2_4():
    assert GRAPH_CAPTURE_SIZES == (1, 2, 4)


def test_decoder_bitrates_include_k5():
    assert DECODER_BITRATES == (3, 4, 5, 6)


def test_graphs_remain_fail_closed(monkeypatch):
    monkeypatch.delenv("VLLM_EXL3_ALLOW_GRAPHS", raising=False)
    assert graphs_allowed() is False
