#!/usr/bin/env bash
# Build exllamav3_ext against the vLLM Torch/CUDA/Python ABI with native SM86 SASS.
# Run from ~/kld-nightly-vllm. Conversion still uses a separate full ExLlamaV3 env.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN="${EXLLAMAV3_COMMIT:-$(python3 -c 'import json,pathlib; print(json.loads(pathlib.Path(r"'"$ROOT"'/manifests/stack.json").read_text())["exllamav3"]["pinned_commit"])')}"
SRC="${EXLLAMAV3_SRC:-$HOME/src/exllamav3}"
OUT="${EXLLAMAV3_EXT_OUT:-$ROOT/build/exllamav3_ext}"
export EXLLAMAV3_EXT_OUT="$OUT"

if [[ ! -d "$SRC/.git" ]]; then
  mkdir -p "$(dirname "$SRC")"
  git clone https://github.com/turboderp-org/exllamav3.git "$SRC"
fi
git -C "$SRC" fetch --all --tags
git -C "$SRC" checkout "$PIN"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export FORCE_CUDA=1

python3 - <<'PY'
import torch, sys
print("torch", torch.__version__, "cuda", torch.version.cuda, "abi", torch._C._GLIBCXX_USE_CXX11_ABI)
print("python", sys.version)
PY

# Extension-only install into the current venv, no build isolation, Ninja, native SASS.
pip install --no-build-isolation --no-deps -e "$SRC"

mkdir -p "$OUT"
python3 - <<PY
import exllamav3_ext, pathlib, shutil, os
src = pathlib.Path(exllamav3_ext.__file__).resolve()
dst_dir = pathlib.Path(os.environ.get("EXLLAMAV3_EXT_OUT", r"$OUT"))
dst_dir.mkdir(parents=True, exist_ok=True)
for path in src.parent.glob("exllamav3_ext*"):
    shutil.copy2(path, dst_dir / path.name)
    print("copied", path, "->", dst_dir / path.name)
print("exllamav3_ext", src)
PY

echo "Set VLLM_EXL3_EXT_PATH=$OUT before launching vLLM."
echo "Pinned commit: $PIN"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
