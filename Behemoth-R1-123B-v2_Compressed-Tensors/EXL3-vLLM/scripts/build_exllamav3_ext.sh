#!/usr/bin/env bash
# Build exllamav3_ext against the vLLM Torch/CUDA/Python ABI with native SM86 SASS.
# Prefers the in-tree submodule (phaelon74/exllamav3 sm86-decode), then EXLLAMAV3_SRC,
# then clones upstream at the pin and applies kernel/overlay.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN="${EXLLAMAV3_COMMIT:-$(python3 -c 'import json,pathlib; print(json.loads(pathlib.Path(r"'"$ROOT"'/manifests/stack.json").read_text())["exllamav3"]["pinned_commit"])')}"
SUBMODULE="$ROOT/exllamav3"
SRC="${EXLLAMAV3_SRC:-}"
_has_quant() {
  local root="$1"
  [[ -d "$root/exllamav3/exllamav3_ext/quant" || -d "$root/exllamav3_ext/quant" ]]
}
if [[ -z "$SRC" ]]; then
  if { [[ -d "$SUBMODULE/.git" ]] || [[ -f "$SUBMODULE/.git" ]]; } && _has_quant "$SUBMODULE"; then
    SRC="$SUBMODULE"
  else
    SRC="${HOME}/src/exllamav3"
  fi
fi
OUT="${EXLLAMAV3_EXT_OUT:-$ROOT/build/exllamav3_ext}"
export EXLLAMAV3_EXT_OUT="$OUT"
APPLY_OVERLAY="${EXL3_APPLY_OVERLAY:-1}"

if [[ ! -d "$SRC/.git" && ! -f "$SRC/.git" ]]; then
  mkdir -p "$(dirname "$SRC")"
  git clone https://github.com/turboderp-org/exllamav3.git "$SRC"
fi

if [[ "$SRC" != "$SUBMODULE" ]]; then
  git -C "$SRC" fetch --all --tags
  git -C "$SRC" checkout -f "$PIN"
fi

if [[ "$APPLY_OVERLAY" == "1" ]]; then
  python3 "$ROOT/kernel/overlay/apply_overlay.py" "$SRC"
fi

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export FORCE_CUDA=1

python3 - <<'PY'
import torch, sys
print("torch", torch.__version__, "cuda", torch.version.cuda, "abi", torch._C._GLIBCXX_USE_CXX11_ABI)
print("python", sys.version)
PY

if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$(command -v python3)" --no-build-isolation --no-deps -e "$SRC"
else
  pip install --no-build-isolation --no-deps -e "$SRC"
fi

mkdir -p "$OUT"
EXLLAMAV3_SRC_TREE="$SRC" python3 - <<'PY'
import os
import pathlib
import shutil
import sys

dst_dir = pathlib.Path(os.environ["EXLLAMAV3_EXT_OUT"]).resolve()
src_root = pathlib.Path(os.environ["EXLLAMAV3_SRC_TREE"]).resolve()
dst_dir.mkdir(parents=True, exist_ok=True)

# env.sh puts EXLLAMAV3_EXT_OUT on PYTHONPATH for serving. Importing from there
# would resolve to the previous build and copy the stale binary over itself.
sys.path = [
    p for p in sys.path
    if not p or pathlib.Path(p).resolve() != dst_dir
]

import torch  # load libc10 / libtorch before the extension

torch_lib = pathlib.Path(torch.__file__).resolve().parent / "lib"
os.environ["LD_LIBRARY_PATH"] = str(torch_lib) + (
    os.pathsep + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else ""
)

import exllamav3_ext

src = pathlib.Path(exllamav3_ext.__file__).resolve()
if src.parent == dst_dir:
    # Still resolved to the output dir; take the newest build under the tree.
    found = [
        p.resolve() for p in src_root.rglob("exllamav3_ext*.so")
        if p.resolve().parent != dst_dir
    ]
    if not found:
        raise SystemExit(
            f"import resolved to {dst_dir} and no exllamav3_ext*.so under {src_root}"
        )
    src = max(found, key=lambda p: p.stat().st_mtime)
    print("import resolved to the output dir; using", src)

copied = 0
for path in sorted(src.parent.glob("exllamav3_ext*")):
    dst = dst_dir / path.name
    if dst.exists() and path.samefile(dst):
        raise SystemExit(f"refusing to copy {path} onto itself")
    shutil.copy2(path, dst)
    print("copied", path, "->", dst)
    copied += 1
if copied == 0:
    raise SystemExit(f"no exllamav3_ext* next to {src}")
print("exllamav3_ext", src)
print("torch_lib", torch_lib)
PY

echo "Set VLLM_EXL3_EXT_PATH=$OUT before launching vLLM."
echo "Source tree: $SRC"
echo "Pinned commit: $PIN"
echo "Overlay: $APPLY_OVERLAY"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
