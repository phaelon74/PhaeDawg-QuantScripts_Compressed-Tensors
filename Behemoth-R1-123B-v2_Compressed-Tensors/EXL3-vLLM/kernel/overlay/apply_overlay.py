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
    f"if (K < 2 || K > 6) return -1; // {MARKER}: K5/K6 opt-in GEMV\n"
    "    if (K > 4 && (cb != 0 || size_m != 1 || "
    "exl3_gemv_k56_mode() == 0)) return -1;"
)
GEMV_TRY_K_GUARD_OLD = "if (K < 2 || K > 4) return false;"
GEMV_TRY_K_GUARD_NEW = (
    f"if (K < 2 || K > 6) return false; // {MARKER}: K5/K6 opt-in GEMV\n"
    "    if (K > 4 && (cb != 0 || size_m != 1 || "
    "exl3_gemv_k56_mode() == 0)) return false;"
)
GEMV_CB_GUARD_OLD = "if (K != 4 && cb == 0) return -1;"
GEMV_CB_GUARD_NEW = (
    f"if (K < 4 && cb == 0 && !exl3_gemv_allow_3inst()) return -1; "
    f"// {MARKER}"
)
GEMV_TRY_CB_GUARD_OLD = "if (K != 4 && cb == 0) return false;"
GEMV_TRY_CB_GUARD_NEW = (
    f"if (K < 4 && cb == 0 && !exl3_gemv_allow_3inst()) return false; "
    f"// {MARKER}"
)

GEMV_SELECT_OLD = """ SEL_GRID(4, 0, false) SEL_GRID(4, 1, false) SEL_GRID(4, 2, false)
 SEL_GRID(2, 1, false) SEL_GRID(2, 2, false) SEL_GRID(2, 1, true) SEL_GRID(2, 2, true)
 SEL_GRID(3, 1, false) SEL_GRID(3, 2, false) SEL_GRID(3, 1, true) SEL_GRID(3, 2, true)"""

GEMV_SELECT_NEW = f""" SEL_GRID(4, 0, false) SEL_GRID(4, 1, false) SEL_GRID(4, 2, false)
 SEL_GRID(2, 0, false) SEL_GRID(2, 1, false) SEL_GRID(2, 2, false) SEL_GRID(2, 1, true) SEL_GRID(2, 2, true)
 SEL_GRID(3, 0, false) SEL_GRID(3, 1, false) SEL_GRID(3, 2, false) SEL_GRID(3, 1, true) SEL_GRID(3, 2, true)
 SEL(5, 0, false, 0, 0, true) SEL(6, 0, false, 0, 0, true)
 SEL(5, 0, false, 0, 0, false) SEL(6, 0, false, 0, 0, false)
 /* {MARKER}: 3inst cb=0 for K=2,3; opt-in staged/register K5/K6 */"""

GEMV_HELPER = (
    f"\n// {MARKER}: allow implicit 3inst GEMV (cb=0) at K=2,3 when "
    f"EXL3_GEMV>=2 or EXL3_GEMV_3INST=1. K=4 3inst is already eligible.\n"
    "static bool exl3_gemv_allow_3inst()\n"
    "{\n"
    '    const char* force = std::getenv("EXL3_GEMV_3INST");\n'
    "    if (force && atoi(force) > 0) return true;\n"
    '    const char* env = std::getenv("EXL3_GEMV");\n'
    "    return env && atoi(env) >= 2;\n"
    "}\n"
)

GEMV_K56_HELPER = f"""
// {MARKER}: K5/K6 M=1 prototype. 1=staged, 2=register extraction.
static int exl3_gemv_k56_mode()
{{
    const char* env = std::getenv("EXL3_GEMV_K56");
    return env ? atoi(env) : 0;
}}
"""

GEMV_SM86_K4_POLICY = f"""
    // {MARKER}: measured RTX 3090 M=1 policy for Behemoth TP4 shapes.
    // Narrow GEMV wins for the large output projections; the regular kernel
    // wins for q/k/v. Keep forced modes 2/3/4 unchanged for experiments.
    if (mode == 1 && cc == CC_AMPERE && K == 4 && cb == 0 && size_m == 1)
    {{
        if ((size_k == 3072 && size_n == 12288) ||
            (size_k == 12288 && size_n == 7168) ||
            (size_k == 7168 && size_n == 12288))
            return 0;
        if (size_k == 12288 && (size_n == 3072 || size_n == 256))
            return -1;
    }}
"""

K56_KERNEL_MARKER = f"{MARKER}: K5/K6 lightweight staged GEMV"

K56_ASSERT_OLD = (
    'static_assert(bits == 2 || bits == 3 || bits == 4, '
    '"exl3_gemv_kernel supports 2, 3 and 4 bpw");'
)
K56_ASSERT_NEW = f"""static_assert(bits >= 2 && bits <= 6);
    // {K56_KERNEL_MARKER}. Limit new instantiations until parity and
    // register-spill profiling have passed on SM86.
    static_assert(bits <= 4 ||
                  (cb == 0 && !c_fp32 && MMODE == 0 &&
                   CFG == 0));"""

K56_CONSTANTS_OLD = """    constexpr int LOADS = bits == 2 ? WNT / 2 : WNT;        // warp loads per k-slice
    constexpr int LSTRIDE = bits == 3 ? 24 : 32;            // uint32 per load"""
K56_CONSTANTS_NEW = """    constexpr int WORD_GROUPS = bits <= 4 ? 1 : 2;
    constexpr int LOADS = bits == 2 ? WNT / 2 : WNT * WORD_GROUPS;
    constexpr int LSTRIDE = bits == 3 ? 24 : 32;
    constexpr int STAGE_WORDS =
        bits <= 4 ? LOADS * LSTRIDE : WNT * TWORDS;"""

K56_STAGE_ARRAY_OLD = (
    "[[maybe_unused]] __shared__ uint32_t "
    "sh_stage[SMEM_STAGE ? WK : 1][SMEM_STAGE ? LOADS * LSTRIDE : 1];"
)
K56_STAGE_ARRAY_NEW = (
    "[[maybe_unused]] __shared__ uint32_t "
    "sh_stage[SMEM_STAGE ? WK : 1][SMEM_STAGE ? STAGE_WORDS : 1];"
)

K56_LOAD_OLD = """            if constexpr (bits == 3)
                return lane < 24 ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
            else
                return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);"""
K56_LOAD_NEW = """            if constexpr (bits == 3)
                return lane < 24 ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
            else if constexpr (bits == 5 || bits == 6)
            {
                int tile = l / WORD_GROUPS;
                int segment = l % WORD_GROUPS;
                int word = segment * 32 + lane;
                return word < TWORDS
                    ? __ldcs(bp + (size_t) i * slice_stride +
                             tile * TWORDS + segment * 32)
                    : 0;
            }
            else
                return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);"""

K56_STAGE_STORE_OLD = """                    if (bits != 3 || lane < 24)
                        sh_stage[warp][l * LSTRIDE + lane] = bw[l];"""
K56_STAGE_STORE_NEW = """                    if constexpr (bits == 5 || bits == 6)
                    {
                        int tile = l / WORD_GROUPS;
                        int segment = l % WORD_GROUPS;
                        int word = segment * 32 + lane;
                        if (word < TWORDS)
                            sh_stage[warp][tile * TWORDS + word] = bw[l];
                    }
                    else if (bits != 3 || lane < 24)
                        sh_stage[warp][l * LSTRIDE + lane] = bw[l];"""

K56_DECODE_BRANCH_OLD = """                    if constexpr (bits == 4)
                        exl3_gemv_ns::dq8_regs_4bits<cb>(tp[(lane + 31) & 31], tp[lane], f0, f1);
                    else if constexpr (bits == 2)
                        exl3_gemv_ns::dq8_regs_2bits<cb>(tp[x_src_a], tp[x_src_b], lane << 3, f0, f1);
                    else
                        exl3_gemv_ns::dq8_regs_3bits<cb>(tp[x_src_a], tp[x_src_b], x_s2, f0, f1);"""
K56_DECODE_BRANCH_NEW = """                    if constexpr (bits == 5 || bits == 6)
                        dq_dispatch<bits, cb>(tp, lane << 3, f0, f1);
                    else if constexpr (bits == 4)
                        exl3_gemv_ns::dq8_regs_4bits<cb>(tp[(lane + 31) & 31], tp[lane], f0, f1);
                    else if constexpr (bits == 2)
                        exl3_gemv_ns::dq8_regs_2bits<cb>(tp[x_src_a], tp[x_src_b], lane << 3, f0, f1);
                    else
                        exl3_gemv_ns::dq8_regs_3bits<cb>(tp[x_src_a], tp[x_src_b], x_s2, f0, f1);"""

K56_REGISTER_MARKER = f"{MARKER}: K5/K6 register extraction"
K56_REGISTER_HELPER = f"""
// {K56_REGISTER_MARKER}: one four-value trellis window.
template <int bits, int cb>
__device__ __forceinline__ void dq4_regs_k56
(
    uint32_t a,
    uint32_t b,
    int s2,
    FragB& frag
)
{{
    uint32_t w3 = fshift(b, a, s2) & 0xffff;
    uint32_t w2 = fshift(b, a, s2 + bits) & 0xffff;
    uint32_t w1 = fshift(b, a, s2 + bits * 2) & 0xffff;
    uint32_t w0 = fshift(b, a, s2 + bits * 3) & 0xffff;
    frag[0] = decode_3inst_2<cb>(w0, w1);
    frag[1] = decode_3inst_2<cb>(w2, w3);
}}
"""

K56_REGISTER_PRECOMPUTE = f"""
    // {K56_REGISTER_MARKER}: source words for two dq4 windows.
    [[maybe_unused]] int x56_a0 = 0, x56_b0 = 0, x56_s0 = 0;
    [[maybe_unused]] int x56_a1 = 0, x56_b1 = 0, x56_s1 = 0;
    if constexpr (bits == 5 || bits == 6)
    {{
        int t0 = lane << 3;
        int p0 = (t0 + 257) * bits - 16;
        int e0 = p0 + 3 * bits + 16;
        int i00 = p0 / 32;
        int i02 = (e0 - 1) / 32;
        x56_s0 = (i02 + 1) * 32 - e0;
        x56_a0 = i00 % TWORDS;
        x56_b0 = i02 % TWORDS;

        int t1 = t0 + 4;
        int p1 = (t1 + 257) * bits - 16;
        int e1 = p1 + 3 * bits + 16;
        int i10 = p1 / 32;
        int i12 = (e1 - 1) / 32;
        x56_s1 = (i12 + 1) * 32 - e1;
        x56_a1 = i10 % TWORDS;
        x56_b1 = i12 % TWORDS;
    }}
"""

K56_REGISTER_DECODE = """                else if constexpr (bits == 5 || bits == 6)
                {
                    uint32_t lo = bw[t * WORD_GROUPS];
                    uint32_t hi = bw[t * WORD_GROUPS + 1];

                    uint32_t av_lo = __shfl_sync(
                        0xffffffffu, lo, x56_a0 & 31);
                    uint32_t av_hi = __shfl_sync(
                        0xffffffffu, hi, x56_a0 & 31);
                    uint32_t bv_lo = __shfl_sync(
                        0xffffffffu, lo, x56_b0 & 31);
                    uint32_t bv_hi = __shfl_sync(
                        0xffffffffu, hi, x56_b0 & 31);
                    uint32_t av = x56_a0 < 32 ? av_lo : av_hi;
                    uint32_t bv = x56_b0 < 32 ? bv_lo : bv_hi;
                    exl3_gemv_ns::dq4_regs_k56<bits, cb>(
                        av, bv, x56_s0, f0);

                    av_lo = __shfl_sync(
                        0xffffffffu, lo, x56_a1 & 31);
                    av_hi = __shfl_sync(
                        0xffffffffu, hi, x56_a1 & 31);
                    bv_lo = __shfl_sync(
                        0xffffffffu, lo, x56_b1 & 31);
                    bv_hi = __shfl_sync(
                        0xffffffffu, hi, x56_b1 & 31);
                    av = x56_a1 < 32 ? av_lo : av_hi;
                    bv = x56_b1 < 32 ? bv_lo : bv_hi;
                    exl3_gemv_ns::dq4_regs_k56<bits, cb>(
                        av, bv, x56_s1, f1);
                }
"""

INT8_GATE_OLD = "if (mul1 && exl3_gemv_int8_enabled())"
INT8_GATE_NEW = (
    f"bool int8_cb = []() {{ const char* e = std::getenv(\"EXL3_INT8_GEMV_CB\"); "
    f"return e && atoi(e) > 0; }}();\n"
    f" if ((mul1 || (cb == 0 && int8_cb)) && exl3_gemv_int8_enabled()) "
    f"// {MARKER}"
)


def _quant_dir(src: Path) -> Path:
    candidates = [
        src / "exllamav3" / "exllamav3_ext" / "quant",
        src / "exllamav3_ext" / "quant",
    ]
    for quant in candidates:
        if quant.is_dir():
            return quant
    raise SystemExit(
        f"not an ExLlamaV3 tree: {src} (no {candidates[0]})"
    )


def _replace_once(text: str, old: str, new: str, path: Path) -> str:
    if new.strip() in text and MARKER in new:
        return text
    if old not in text:
        raise SystemExit(f"{path}: expected snippet not found:\n{old[:120]}")
    return text.replace(old, new, 1)


def apply(src: Path) -> None:
    quant = _quant_dir(src)

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
    if "exl3_gemv_k56_mode" not in text:
        needle = "static int exl3_gemv_env_mode()"
        if needle not in text:
            raise SystemExit(f"{gemv}: missing exl3_gemv_env_mode")
        text = text.replace(needle, GEMV_K56_HELPER + "\n" + needle, 1)
    text = text.replace(
        "!exl3_gemv_allow_k56()", "exl3_gemv_k56_mode() == 0"
    )
    if "measured RTX 3090 M=1 policy" not in text:
        needle = "    if (mode == 2) return size_n <= 8192 ? 0 : 1;"
        if needle not in text:
            raise SystemExit(f"{gemv}: missing GEMV mode dispatch")
        text = text.replace(
            needle,
            GEMV_SM86_K4_POLICY
            + "    if (K > 4) return 0; // opt-in K5/K6 narrow kernel\n"
            + needle,
            1,
        )
    elif "if (K > 4) return 0; // opt-in K5/K6 narrow kernel" not in text:
        needle = "    if (mode == 2) return size_n <= 8192 ? 0 : 1;"
        text = text.replace(
            needle,
            "    if (K > 4) return 0; // opt-in K5/K6 narrow kernel\n"
            + needle,
            1,
        )
    # Migrate earlier overlay guards.
    text = text.replace(
        f"if (K < 2 || K > 4) return -1; // {MARKER}: GEMV kernel is 2/3/4 bpw only",
        GEMV_K_GUARD_NEW,
    )
    text = text.replace(
        f"if (K < 2 || K > 8) return -1; // {MARKER}: K=5..8 GEMV instances",
        GEMV_K_GUARD_NEW,
    )
    text = text.replace(
        f"if (K < 2 || K > 8) return false; // {MARKER}",
        GEMV_TRY_K_GUARD_NEW,
    )
    if GEMV_K_GUARD_OLD in text:
        text = text.replace(GEMV_K_GUARD_OLD, GEMV_K_GUARD_NEW)
    if GEMV_TRY_K_GUARD_OLD in text:
        text = text.replace(GEMV_TRY_K_GUARD_OLD, GEMV_TRY_K_GUARD_NEW)
    # two copies of the cb/k guards (cfg + try_launch)
    text = text.replace(
        f"if (K != 4 && cb == 0 && !exl3_gemv_allow_3inst()) return -1; // {MARKER}",
        GEMV_CB_GUARD_NEW,
    )
    if GEMV_CB_GUARD_OLD in text:
        text = text.replace(GEMV_CB_GUARD_OLD, GEMV_CB_GUARD_NEW)
    text = text.replace(
        f"if (K != 4 && cb == 0 && !exl3_gemv_allow_3inst()) return false; // {MARKER}",
        GEMV_TRY_CB_GUARD_NEW,
    )
    if GEMV_TRY_CB_GUARD_OLD in text:
        text = text.replace(
            GEMV_TRY_CB_GUARD_OLD, GEMV_TRY_CB_GUARD_NEW
        )
    for k in (5, 6, 7, 8):
        text = text.replace(
            f" SEL_GRID({k}, 0, false) SEL_GRID({k}, 1, false) "
            f"SEL_GRID({k}, 2, false)\n",
            "",
        )
    if "SEL_GRID(2, 0" not in text:
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
    if "SEL(5, 0, false, 0, 0, true)" not in text:
        needle = " #undef SEL_GRID"
        if needle not in text:
            raise SystemExit(f"{gemv}: SEL_GRID undef not found")
        text = text.replace(
            needle,
            " SEL(5, 0, false, 0, 0, true) "
            "SEL(6, 0, false, 0, 0, true)\n"
            f" /* {MARKER}: opt-in staged K5/K6 */\n"
            + needle,
            1,
        )
    if "SEL(5, 0, false, 0, 0, false)" not in text:
        needle = " #undef SEL_GRID"
        text = text.replace(
            needle,
            " SEL(5, 0, false, 0, 0, false) "
            "SEL(6, 0, false, 0, 0, false)\n"
            f" /* {MARKER}: register K5/K6 */\n"
            + needle,
            1,
        )
    if "K > 4 ? exl3_gemv_k56_mode() == 1" not in text:
        text = text.replace(
            "bool smem = K > 4 ? true : exl3_gemv_env_smem() == 1;",
            "bool smem = K > 4 ? exl3_gemv_k56_mode() == 1 "
            ": exl3_gemv_env_smem() == 1;",
        )
    if "K > 4 ? exl3_gemv_k56_mode() == 1" not in text:
        text = _replace_once(
            text,
            "bool smem = exl3_gemv_env_smem() == 1;",
            "bool smem = K > 4 ? exl3_gemv_k56_mode() == 1 "
            ": exl3_gemv_env_smem() == 1;",
            gemv,
        )
    gemv.write_text(text, encoding="utf-8")

    gemv_kernel = quant / "exl3_gemv_kernel.cuh"
    kernel_text = gemv_kernel.read_text(encoding="utf-8")
    if K56_KERNEL_MARKER not in kernel_text:
        kernel_text = _replace_once(
            kernel_text, K56_ASSERT_OLD, K56_ASSERT_NEW, gemv_kernel
        )
        kernel_text = _replace_once(
            kernel_text, K56_CONSTANTS_OLD, K56_CONSTANTS_NEW, gemv_kernel
        )
        kernel_text = _replace_once(
            kernel_text, K56_STAGE_ARRAY_OLD, K56_STAGE_ARRAY_NEW, gemv_kernel
        )
        kernel_text = _replace_once(
            kernel_text, K56_LOAD_OLD, K56_LOAD_NEW, gemv_kernel
        )
        kernel_text = _replace_once(
            kernel_text,
            K56_STAGE_STORE_OLD,
            K56_STAGE_STORE_NEW,
            gemv_kernel,
        )
        kernel_text = _replace_once(
            kernel_text,
            K56_DECODE_BRANCH_OLD,
            K56_DECODE_BRANCH_NEW,
            gemv_kernel,
        )
    if K56_REGISTER_MARKER not in kernel_text:
        needle = "}  // namespace exl3_gemv_ns"
        if needle not in kernel_text:
            raise SystemExit(f"{gemv_kernel}: GEMV namespace end not found")
        kernel_text = kernel_text.replace(
            needle, K56_REGISTER_HELPER + "\n" + needle, 1
        )
        needle = "    __shared__ float sh_red[WK][ROWS][COLS];"
        if needle not in kernel_text:
            raise SystemExit(f"{gemv_kernel}: reduction storage not found")
        kernel_text = kernel_text.replace(
            needle, K56_REGISTER_PRECOMPUTE + "\n" + needle, 1
        )
        needle = "                else  // bits == 3"
        if needle not in kernel_text:
            raise SystemExit(f"{gemv_kernel}: 3-bit register branch not found")
        kernel_text = kernel_text.replace(
            needle, K56_REGISTER_DECODE + needle, 1
        )
    gemv_kernel.write_text(kernel_text, encoding="utf-8")

    gemm = quant / "exl3_gemm.cu"
    text = gemm.read_text(encoding="utf-8")
    if "#include <cstdlib>" not in text:
        text = "#include <cstdlib>\n" + text
    if "EXL3_INT8_GEMV_CB" not in text:
        text = _replace_once(text, INT8_GATE_OLD, INT8_GATE_NEW, gemm)
    # Drop leftover lut_ensure inject from the previous overlay (cudaMalloc
    # inside gemm_gr is unsafe during CUDA graph capture).
    text = text.replace("    exl3_lut_ensure();\n", "")
    text = text.replace(
        f"extern void exl3_lut_ensure();\n\n// {MARKER}\n",
        "",
    )
    gemm.write_text(text, encoding="utf-8")

    # Do not hook decode_3inst in codebook.cuh. Without -rdc, nvcc treats
    # extern device symbols as per-TU statics (warning 20044), so a LUT flag
    # set in exl3_decode_lut.cu never reaches GEMM kernels. Strip leftover
    # hooks from the previous overlay so arithmetic decode stays live.
    codebook = quant / "codebook.cuh"
    if codebook.is_file():
        cb_text = codebook.read_text(encoding="utf-8")
        stripped = cb_text
        stripped = stripped.replace('#include "codebook_lut.cuh"\n\n', "")
        stripped = stripped.replace('#include "codebook_lut.cuh"\n', "")
        stripped = stripped.replace(
            " if (exl3_lut_enabled())\n"
            "     return exl3_lut_decode<cb>(x);\n",
            "",
        )
        stripped = stripped.replace(
            " if (exl3_lut_enabled())\n"
            "     return __halves2half2(exl3_lut_decode<cb>(x0), "
            "exl3_lut_decode<cb>(x1));\n",
            "",
        )
        if stripped != cb_text:
            codebook.write_text(stripped, encoding="utf-8")

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
