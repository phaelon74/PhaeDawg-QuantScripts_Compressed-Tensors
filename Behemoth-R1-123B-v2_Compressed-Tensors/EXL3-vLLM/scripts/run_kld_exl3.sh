#!/usr/bin/env bash
# Frozen KLD suite for the Behemoth EXL3 checkpoint.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"
JSON="${1:-$ROOT/kld/Models_KLD_Behemoth_EXL3.json}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"
export VLLM_PLUGINS="${VLLM_PLUGINS:-vllm_exl3_sm86}"
export VLLM_EXL3_EXT_PATH="${VLLM_EXL3_EXT_PATH:-$ROOT/build/exllamav3_ext}"

"$REPO/KLD_Scripts/run_batch_kld.sh" "$JSON"

python3 - "$JSON" <<'PY'
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text())
out = Path(cfg["outputDirectory"]) / f"{cfg['resultsBasename']}_KLD-Results.json"
if not out.is_file():
    print("KLD results not written yet:", out)
    sys.exit(0)
data = json.loads(out.read_text())
gate = 0.042380
stretch = 0.034004
for model in data.get("models", data if isinstance(data, list) else []):
    mean = model.get("meanKld") or model.get("mean_kld")
    print("meanKld", mean)
    if mean is None:
        sys.exit(1)
    if float(mean) > gate:
        print(f"FAIL quality gate {mean} > {gate}")
        sys.exit(1)
    if float(mean) > stretch:
        print(f"PASS minimum gate; stretch {stretch} not reached")
    else:
        print("PASS stretch gate")
PY
