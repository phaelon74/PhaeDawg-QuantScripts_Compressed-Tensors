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
    assert "SMEM_STAGE || bits > 4" in text
    assert "dq_dispatch<bits, cb>(tp, lane << 3, f0, f1)" in text
    assert "dq4_regs_k56<bits, cb>" in text
    assert "EXL3_GEMV_K4_ARITH" in text
    assert "exl3_gemv_select_k4_arith" in text
    assert "decode_pair_cb0_mad_" in text
    assert "decode8_cb0_batched_" in text
    assert "K4_ARITH_MODE = 0" in text
    assert "EXL3_GEMV_K4_SLIM" in text
    assert "exl3_gemv_select_k4_slim" in text
    assert "K4 narrow-16 occupancy layout" in text
    assert "narrow_cols = k4_slim ? 16 : 32" in text
    assert "EXL3_GEMV_K4_TCFOLD" in text
    assert "exl3_gemv_select_k4_tcfold" in text
    assert "K4 cb0 tensor-core fold" in text
    assert "isolate the K4 tensor-core fold" in text
    assert "cb0_fold_" in text
    assert "dq8_regs_4bits_tcfold" in text
    assert "bool K4_TCFOLD = false>" in text
    # The fold instance must stay isolated from every other configuration.
    assert "exl3_gemv_kernel<4, false, 0, 0, 0, false, 0, true>" in text
    assert "!k4_slim && !k4_arith_mode" in text
    assert "mma_ab_h(aq0, aq1, g0, ch[t][0])" in text
    assert "mma_ab_h(aq2, aq3, g3, ch[t][1])" in text
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
