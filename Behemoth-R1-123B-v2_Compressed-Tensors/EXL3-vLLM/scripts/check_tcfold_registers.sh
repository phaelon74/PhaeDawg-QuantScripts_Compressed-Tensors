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
    m = re.search(r"Function\s+(\S+)", line)
    if m:
        current = m.group(1)
        continue
    m = re.search(r"\bREG:(\d+)", line)
    if m and current:
        entries.append((current, int(m.group(1))))
        current = None

gemv = [(s, r) for s, r in entries if "exl3_gemv_kernel" in s]
if not gemv:
    print("no exl3_gemv_kernel instances in the binary", file=sys.stderr)
    raise SystemExit(2)

try:
    demangled = subprocess.run(
        ["c++filt"],
        input="\n".join(s for s, _ in gemv),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
except (OSError, subprocess.CalledProcessError):
    demangled = [s for s, _ in gemv]

fold = []
other = []
for (sym, regs), name in zip(gemv, demangled):
    row = (name, regs)
    # The fold is the only instance whose last template argument is true.
    if re.search(r"false,\s*0,\s*true>", name) or "0, true>" in name:
        fold.append(row)
    else:
        other.append(row)

print(f"{len(gemv)} exl3_gemv_kernel instances")
worst_other = max((r for _, r in other), default=0)
print(f"max registers, non-fold instances: {worst_other}")

if not fold:
    print(
        "fold instance not in the binary: rebuild with the overlay applied",
        file=sys.stderr,
    )
    raise SystemExit(1)

bad = 0
for name, regs in sorted(fold, key=lambda r: -r[1]):
    status = "OK" if regs <= limit else "OVER"
    if regs > limit:
        bad += 1
    print(f"[{status}] REG={regs:3d}  {name}")

if bad:
    print(
        f"fold exceeds {limit} registers: occupancy will drop to 1 block/SM",
        file=sys.stderr,
    )
    raise SystemExit(1)
print("fold is within the register budget")
PY
}

py_filter
grep -c 'STACK:[1-9]' "$DUMP" > /dev/null 2>&1 && {
  echo "WARNING: some kernels spill to local memory (STACK != 0)" >&2
}
exit 0
