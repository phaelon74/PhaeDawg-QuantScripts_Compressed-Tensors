from __future__ import annotations

from pathlib import Path

from vllm_exl3_sm86.constants import (
    CODEBOOK_FLAGS,
    GRAPH_CAPTURE_SIZES,
    PINNED_EXLLAMAV3_COMMIT,
)


def test_overlay_files_exist():
    root = Path(__file__).resolve().parents[1]
    overlay = root / "kernel" / "overlay"
    assert (overlay / "apply_overlay.py").is_file()
    assert (overlay / "codebook_lut.cuh").is_file()
    assert (overlay / "exl3_decode_lut.cu").is_file()
    text = (overlay / "apply_overlay.py").read_text(encoding="utf-8")
    assert "SEL_GRID(2, 0" in text
    assert "SEL_GRID(5, 0" not in text
    assert "EXL3_INT8_GEMV_CB" in text
    assert "exl3_gemv_allow_3inst" in text
    assert "measured RTX 3090 M=1 policy" in text
    assert "size_k == 12288 && size_n == 7168" in text
    assert "size_k == 12288 && (size_n == 3072 || size_n == 256)" in text
    assert "EXL3_GEMV_K56" in text
    assert "SEL(5, 0, false, 0, 0, true)" in text
    assert "SEL(5, 0, false, 0, 0, false)" in text
    assert "K5/K6 lightweight staged GEMV" in text
    assert "K5/K6 register extraction" in text
    assert "word < TWORDS" in text
    assert "dq_dispatch<bits, cb>(tp, lane << 3, f0, f1)" in text
    assert "dq4_regs_k56<bits>" in text
    assert "Do not hook decode_3inst" in text
    assert "exl3_lut_decode<cb>(x)" in text  # leftover-hook stripper only


def test_codebook_flags_cover_all_three():
    assert (False, False) in CODEBOOK_FLAGS
    assert (True, False) in CODEBOOK_FLAGS
    assert (False, True) in CODEBOOK_FLAGS


def test_capture_sizes_and_pin():
    assert GRAPH_CAPTURE_SIZES[0] == 1
    assert 8 in GRAPH_CAPTURE_SIZES
    assert PINNED_EXLLAMAV3_COMMIT.startswith("0c49587")
