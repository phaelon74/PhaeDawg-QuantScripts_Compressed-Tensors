"""Apply SM86 decode overlays onto a checked-out ExLlamaV3 tree.

Patches the pinned commit in place. Safe to re-run (idempotent via markers).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MARKER = "Phaedawg-SM86-overlay"
OVERLAY_DIR = Path(__file__).resolve().parent

GEMV_K_GUARD_OLD = "if (K < 2 || K > 4) return -1;"
GEMV_K_GUARD_NEW = (
    f"if (K < 2 || K > 8) return -1; // {MARKER}: K=5..8 GEMV instances"
)
GEMV_CB_GUARD_OLD = "if (K != 4 && cb == 0) return -1;"
GEMV_CB_GUARD_NEW = (
    f"if (K != 4 && cb == 0 && !exl3_gemv_allow_3inst()) return -1; "
    f"// {MARKER}"
)

GEMV_SELECT_OLD = """ SEL_GRID(4, 0, false) SEL_GRID(4, 1, false) SEL_GRID(4, 2, false)
 SEL_GRID(2, 1, false) SEL_GRID(2, 2, false) SEL_GRID(2, 1, true) SEL_GRID(2, 2, true)
 SEL_GRID(3, 1, false) SEL_GRID(3, 2, false) SEL_GRID(3, 1, true) SEL_GRID(3, 2, true)"""

GEMV_SELECT_NEW = f""" SEL_GRID(4, 0, false) SEL_GRID(4, 1, false) SEL_GRID(4, 2, false)
 SEL_GRID(2, 0, false) SEL_GRID(2, 1, false) SEL_GRID(2, 2, false) SEL_GRID(2, 1, true) SEL_GRID(2, 2, true)
 SEL_GRID(3, 0, false) SEL_GRID(3, 1, false) SEL_GRID(3, 2, false) SEL_GRID(3, 1, true) SEL_GRID(3, 2, true)
 SEL_GRID(5, 0, false) SEL_GRID(5, 1, false) SEL_GRID(5, 2, false)
 SEL_GRID(6, 0, false) SEL_GRID(6, 1, false) SEL_GRID(6, 2, false)
 SEL_GRID(7, 0, false) SEL_GRID(7, 1, false) SEL_GRID(7, 2, false)
 SEL_GRID(8, 0, false) SEL_GRID(8, 1, false) SEL_GRID(8, 2, false)
 /* {MARKER}: 3inst K=5..8 + cb=0 for K=2,3 */"""

GEMV_HELPER = (
    f"\n// {MARKER}: allow implicit 3inst GEMV (cb=0) beyond K=4 when "
    f"EXL3_GEMV>=2 or EXL3_GEMV_3INST=1\n"
    "static bool exl3_gemv_allow_3inst()\n"
    "{\n"
    '    const char* force = std::getenv("EXL3_GEMV_3INST");\n'
    "    if (force && atoi(force) > 0) return true;\n"
    '    const char* env = std::getenv("EXL3_GEMV");\n'
    "    return env && atoi(env) >= 2;\n"
    "}\n"
)

INT8_GATE_OLD = "if (mul1 && exl3_gemv_int8_enabled())"
INT8_GATE_NEW = (
    f"bool int8_cb = []() {{ const char* e = std::getenv(\"EXL3_INT8_GEMV_CB\"); "
    f"return e && atoi(e) > 0; }}();\n"
    f" if ((mul1 || (cb == 0 && int8_cb)) && exl3_gemv_int8_enabled()) "
    f"// {MARKER}"
)

CODEBOOK_INCLUDE_OLD = '#include "codebook.cuh"'
# we'll include lut from codebook.cuh itself

CODEBOOK_HOOK_OLD = """template <int cb>
__device__ inline half decode_3inst(uint32_t x)
{
 if constexpr (cb == 0)
 {"""

CODEBOOK_HOOK_NEW = f"""#include "codebook_lut.cuh"

template <int cb>
__device__ inline half decode_3inst(uint32_t x)
{{
 if (exl3_lut_enabled())
     return exl3_lut_decode<cb>(x);
 if constexpr (cb == 0)
 {{"""

CODEBOOK_HOOK2_OLD = """template <int cb>
__device__ inline half2 decode_3inst_2(uint32_t x0, uint32_t x1)
{
 if constexpr (cb == 0)
 {"""

CODEBOOK_HOOK2_NEW = f"""template <int cb>
__device__ inline half2 decode_3inst_2(uint32_t x0, uint32_t x1)
{{
 if (exl3_lut_enabled())
     return __halves2half2(exl3_lut_decode<cb>(x0), exl3_lut_decode<cb>(x1));
 if constexpr (cb == 0)
 {{"""


def _replace_once(text: str, old: str, new: str, path: Path) -> str:
    if new.strip() in text and MARKER in new:
        return text
    if old not in text:
        raise SystemExit(f"{path}: expected snippet not found:\n{old[:120]}")
    return text.replace(old, new, 1)


def apply(src: Path) -> None:
    quant = src / "exllamav3" / "exllamav3_ext" / "quant"
    if not quant.is_dir():
        raise SystemExit(f"not an ExLlamaV3 tree: {src}")

    shutil.copy2(OVERLAY_DIR / "codebook_lut.cuh", quant / "codebook_lut.cuh")
    shutil.copy2(OVERLAY_DIR / "exl3_decode_lut.cu", quant / "exl3_decode_lut.cu")
    shutil.copy2(
        OVERLAY_DIR / "exl3_gemv_lut_kernel.cuh", quant / "exl3_gemv_lut_kernel.cuh"
    )

    gemv = quant / "exl3_gemv.cu"
    text = gemv.read_text(encoding="utf-8")
    if "exl3_gemv_allow_3inst" not in text:
        needle = "static int exl3_gemv_env_mode()"
        if needle not in text:
            raise SystemExit(f"{gemv}: missing exl3_gemv_env_mode")
        text = text.replace(needle, GEMV_HELPER + "\n" + needle, 1)
    text = text.replace(GEMV_K_GUARD_OLD, GEMV_K_GUARD_NEW)
    text = text.replace(
        "if (K < 2 || K > 4) return false;",
        f"if (K < 2 || K > 8) return false; // {MARKER}",
    )
    # two copies of the cb/k guards (cfg + try_launch)
    if GEMV_CB_GUARD_OLD in text:
        text = text.replace(GEMV_CB_GUARD_OLD, GEMV_CB_GUARD_NEW)
    if "SEL_GRID(5, 0" not in text:
        if GEMV_SELECT_OLD in text:
            text = text.replace(GEMV_SELECT_OLD, GEMV_SELECT_NEW, 1)
        elif "SEL_GRID(4, 0, false)" in text:
            text = text.replace(
                "SEL_GRID(4, 0, false) SEL_GRID(4, 1, false) SEL_GRID(4, 2, false)",
                GEMV_SELECT_NEW.strip(),
                1,
            )
        else:
            raise SystemExit(f"{gemv}: SEL_GRID block not found")
    gemv.write_text(text, encoding="utf-8")

    gemm = quant / "exl3_gemm.cu"
    text = gemm.read_text(encoding="utf-8")
    if "EXL3_INT8_GEMV_CB" not in text:
        text = _replace_once(text, INT8_GATE_OLD, INT8_GATE_NEW, gemm)
    if "exl3_lut_ensure" not in text:
        needle = "int exl3_gemm_gr"
        if needle not in text:
            raise SystemExit(f"{gemm}: missing exl3_gemm_gr")
        text = text.replace(
            needle,
            f'extern void exl3_lut_ensure();\n\n// {MARKER}\nint exl3_gemm_gr',
            1,
        )
        launch_need = "const at::cuda::OptionalCUDAGuard device_guard(A.device());"
        if launch_need not in text:
            raise SystemExit(f"{gemm}: missing device_guard")
        text = text.replace(
            launch_need,
            launch_need + "\n    exl3_lut_ensure();",
            1,
        )
    gemm.write_text(text, encoding="utf-8")

    codebook = quant / "codebook.cuh"
    text = codebook.read_text(encoding="utf-8")
    if "codebook_lut.cuh" not in text:
        if '#include "codebook_lut.cuh"' not in text:
            # Insert the LUT include after the last existing include / pragma block.
            needle = '#pragma once'
            if needle in text:
                text = text.replace(
                    needle, needle + '\n\n#include "codebook_lut.cuh"\n', 1
                )
            else:
                text = '#include "codebook_lut.cuh"\n' + text
        if "exl3_lut_enabled()" not in text:
            hook = (
                " if (exl3_lut_enabled())\n"
                "     return exl3_lut_decode<cb>(x);\n"
            )
            key = "__device__ inline half decode_3inst"
            idx = text.find(key)
            if idx < 0:
                raise SystemExit(f"{codebook}: decode_3inst not found")
            brace = text.find("{", idx)
            text = text[: brace + 1] + "\n" + hook + text[brace + 1 :]
        if "exl3_lut_decode<cb>(x0)" not in text:
            hook2 = (
                " if (exl3_lut_enabled())\n"
                "     return __halves2half2(exl3_lut_decode<cb>(x0), "
                "exl3_lut_decode<cb>(x1));\n"
            )
            key = "__device__ inline half2 decode_3inst_2"
            idx = text.find(key)
            if idx < 0:
                raise SystemExit(f"{codebook}: decode_3inst_2 not found")
            brace = text.find("{", idx)
            text = text[: brace + 1] + "\n" + hook2 + text[brace + 1 :]
    codebook.write_text(text, encoding="utf-8")

    stamp = quant / ".sm86_overlay_applied"
    stamp.write_text(MARKER + "\n", encoding="utf-8")
    setup = src / "setup.py"
    if setup.is_file():
        setup_txt = setup.read_text(encoding="utf-8")
        if "exl3_decode_lut.cu" not in setup_txt and "glob" not in setup_txt.lower():
            print(
                "WARNING: setup.py does not glob .cu files; "
                "add quant/exl3_decode_lut.cu to the extension sources."
            )
    print(f"applied SM86 overlay to {src}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path, help="ExLlamaV3 checkout root")
    args = parser.parse_args()
    apply(args.src.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
