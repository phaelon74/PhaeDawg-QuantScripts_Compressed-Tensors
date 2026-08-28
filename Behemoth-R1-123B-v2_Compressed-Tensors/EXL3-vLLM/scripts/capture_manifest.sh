#!/usr/bin/env bash
# Capture the full vLLM / Torch / CUDA / ExLlamaV3 ABI into manifests/stack.json.
# Run inside ~/kld-nightly-vllm on the Linux GPU host.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/manifests/stack.captured.json}"
STACK="$ROOT/manifests/stack.json"

python3 - "$STACK" "$OUT" <<'PY'
import json, os, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

stack_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
data = json.loads(stack_path.read_text(encoding="utf-8"))
data["status"] = "captured"
data["captured_at"] = datetime.now(timezone.utc).isoformat()
data["host"] = platform.node()

data["python"]["version"] = platform.python_version()
data["python"]["abi"] = list(platform.python_version_tuple())
data["python"]["executable"] = sys.executable
data["python"]["implementation"] = platform.python_implementation()

try:
    import torch
    data["torch"]["version"] = torch.__version__
    data["torch"]["cuda_compiled"] = torch.version.cuda
    data["torch"]["cxx11_abi"] = str(torch._C._GLIBCXX_USE_CXX11_ABI)
except Exception as exc:
    data["torch"]["capture_error"] = repr(exc)

try:
    import vllm
    data["vllm"]["reported_version"] = vllm.__version__
    data["vllm"]["python_package_location"] = str(Path(vllm.__file__).resolve())
    git_sha = getattr(vllm, "__version_tuple__", None)
    data["vllm"]["version_tuple"] = list(git_sha) if git_sha else None
except Exception as exc:
    data["vllm"]["capture_error"] = repr(exc)

def _run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"

data["cuda"]["nvcc"] = _run(["nvcc", "--version"])
data["cuda"]["nvidia_smi"] = _run([
    "nvidia-smi",
    "--query-gpu=index,name,compute_cap,clocks.gr,clocks.mem,power.limit,memory.total",
    "--format=csv",
])
data["cuda"]["topo"] = _run(["nvidia-smi", "topo", "-m"])
data["compiler"]["cxx"] = os.environ.get("CXX", "c++")
data["compiler"]["cxx_version"] = _run([data["compiler"]["cxx"], "--version"])

ext_sha = os.environ.get("EXLLAMAV3_COMMIT")
if ext_sha:
    data["exllamav3"]["built_commit"] = ext_sha

out_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {out_path}")
PY
