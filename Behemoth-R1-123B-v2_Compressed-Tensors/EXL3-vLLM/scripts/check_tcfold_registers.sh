#!/usr/bin/env bash
# Register gate for the K4 tensor-core fold (EXL3_GEMV_K4_TCFOLD).
#
# Occupancy on SM86 is 2 blocks/SM at 512 threads only while the kernel stays
# at or below 64 registers per thread (512 * 64 * 2 == the 65536-register file).
# One extra register drops to 1 block/SM and erases the fold's gain, so this
# reads the built binary instead of trusting the source.
#
# Reads resource usage from the compiled extension; no rebuild required.
#
#   bash scripts/check_tcfold_registers.sh
#   bash scripts/check_tcfold_registers.sh /path/to/exllamav3_ext.so
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIMIT="${TCFOLD_REG_LIMIT:-64}"
SO="${1:-}"

if [[ -z "$SO" ]]; then
  EXT_DIR="${VLLM_EXL3_EXT_PATH:-$ROOT/build/exllamav3_ext}"
  SO="$(find "$EXT_DIR" -maxdepth 1 -name 'exllamav3_ext*.so' | head -n 1 || true)"
fi
if [[ -z "$SO" || ! -f "$SO" ]]; then
  echo "no exllamav3_ext*.so found; build first or pass the path" >&2
  exit 2
fi
command -v cuobjdump >/dev/null 2>&1 || {
  echo "cuobjdump not on PATH (CUDA toolkit bin)" >&2
  exit 2
}

echo "binary: $SO"
echo "limit:  $LIMIT registers/thread"

# Distinguish "overlay missing from this binary" from "instance not emitted".
if ! grep -qa 'EXL3_GEMV_K4_TCFOLD' "$SO"; then
  echo "" >&2
  echo "this binary has no EXL3_GEMV_K4_TCFOLD literal: it predates the fold." >&2
  echo "the build's copy step likely failed, so $SO is the previous build." >&2
  exit 1
fi

DUMP="$(mktemp)"
trap 'rm -f "$DUMP"' EXIT
cuobjdump -res-usage "$SO" > "$DUMP"

# cuobjdump emits "Function <mangled>" then an indented "REG:<n>" line.
# Demangle only the GEMV kernels so the fold instance is identifiable by its
# trailing "true>" template argument.
py_filter() {
  python3 - "$DUMP" "$LIMIT" <<'PY'
import re
import subprocess
import sys

dump_path, limit = sys.argv[1], int(sys.argv[2])
text = open(dump_path, encoding="utf-8", errors="replace").read()

# Pair each Function symbol with the REG count that follows it.
entries = []
current = None
for line in text.splitlines():
    m = re.search(r"Function\s*:?\s*(\S+)", line)
    if m:
        # cuobjdump writes "Function <mangled>:" with a trailing colon.
        current = m.group(1).rstrip(":")
        continue
    m = re.search(r"\bREG:(\d+)", line)
    if m and current:
        stack = re.search(r"\bSTACK:(\d+)", line)
        entries.append(
            (current, int(m.group(1)), int(stack.group(1)) if stack else 0)
        )
        current = None

gemv = [e for e in entries if "exl3_gemv_kernel" in e[0]]
if not gemv:
    print("no exl3_gemv_kernel instances in the binary", file=sys.stderr)
    raise SystemExit(2)


# SM86: 65536 registers per SM, allocated per warp in units of 8 per thread.
def blocks_per_sm(regs: int, threads: int) -> int:
    warps = threads // 32
    per_warp = -(-regs // 8) * 8 * 32
    by_regs = 65536 // (per_warp * warps) if per_warp else 16
    return max(0, min(by_regs, 1536 // threads, 16))

try:
    demangled = subprocess.run(
        ["c++filt"],
        input="\n".join(e[0] for e in gemv),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
except (OSError, subprocess.CalledProcessError):
    demangled = [e[0] for e in gemv]

ARGS = re.compile(r"exl3_gemv_kernel<([^>]*)>")
fold = None
control = None
rows = []
for (sym, regs, stack), name in zip(gemv, demangled):
    m = ARGS.search(name)
    if not m:
        continue
    args = [a.strip() for a in m.group(1).split(",")]
    if len(args) < 8:
        args += ["0", "false"][len(args) - 6:]
    bits, c_fp32, cb, mmode, cfg, smem, arith, tcfold = args[:8]
    # The fold only ever runs on the K4 / cb0 / fp16 / M=1 / CFG0 hotspot.
    if (bits, c_fp32, cb, mmode, cfg, smem) != ("4", "false", "0", "0", "0", "false"):
        continue
    threads = 512
    row = (regs, threads, blocks_per_sm(regs, threads), stack, arith, tcfold)
    rows.append(row)
    if tcfold == "true":
        fold = row
    elif arith == "0":
        control = row

print(f"{len(gemv)} exl3_gemv_kernel instances")
print("K4 / cb0 / fp16 / M=1 / CFG0 instances (512 threads):")
for regs, threads, blocks, stack, arith, tcfold in sorted(rows):
    tag = "fold" if tcfold == "true" else (
        "control" if arith == "0" else f"arith{arith}"
    )
    warps = blocks * threads // 32
    spill = "" if stack == 0 else f"  SPILL={stack}B"
    print(
        f"  {tag:<8} REG={regs:3d}  blocks/SM={blocks}  "
        f"occupancy={warps / 48:.0%}{spill}"
    )

if fold is None:
    print(
        "fold instance not in the binary: rebuild with the overlay applied",
        file=sys.stderr,
    )
    raise SystemExit(1)

if control is None:
    print("no unfolded control instance to compare against", file=sys.stderr)
    raise SystemExit(1)
fold_regs, fold_blocks, fold_stack = fold[0], fold[2], fold[3]
ctrl_regs, ctrl_blocks = control[0], control[2]

print(f"control {ctrl_regs} regs / {ctrl_blocks} blocks, "
      f"fold {fold_regs} regs / {fold_blocks} blocks")
if fold_stack:
    # Forcing the register target can be met by spilling, which is worse than
    # the occupancy it buys back.
    print(
        f"fold spills {fold_stack} bytes to local memory; the register target "
        "was met by spilling, not by fitting",
        file=sys.stderr,
    )
    raise SystemExit(1)
if fold_blocks < ctrl_blocks:
    print(
        f"fold loses occupancy ({ctrl_blocks} -> {fold_blocks} blocks/SM); "
        f"it needs to fit in {limit} registers to keep {ctrl_blocks} blocks",
        file=sys.stderr,
    )
    raise SystemExit(1)
if fold_regs > limit:
    print(
        f"fold is over {limit} registers but the control is too, "
        "so occupancy is unchanged: benchmark it"
    )
else:
    print("fold is within the register budget")
PY
}

py_filter
grep -c 'STACK:[1-9]' "$DUMP" > /dev/null 2>&1 && {
  echo "WARNING: some kernels spill to local memory (STACK != 0)" >&2
}
exit 0
