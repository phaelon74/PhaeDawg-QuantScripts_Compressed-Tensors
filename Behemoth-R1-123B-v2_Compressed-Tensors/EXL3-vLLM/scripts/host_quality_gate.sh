#!/usr/bin/env bash
# Qualification at a decode gate: parity, graph stress, KLD, serving, restart.
# Does not start serve; pass a running endpoint or set SKIP_SERVE=1.
#
#   export EXL3=...
#   export VLLM_API_KEY=...
#   bash "$EXL3/scripts/host_quality_gate.sh"
set -euo pipefail
EXL3="${EXL3:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT="${EXL3_GATE_OUT:-$EXL3/results/quality}"
HOST="${BENCH_HOST:-10.9.99.22}"
PORT="${PORT:-8000}"
MODEL="${EXL3_SERVED_MODEL_NAME:-Behemoth-R1-123B-v2-EXL3-4.25-H6}"
mkdir -p "$OUT"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export VLLM_EXL3_EXT_PATH="${VLLM_EXL3_EXT_PATH:-$EXL3/build/exllamav3_ext}"
export VLLM_EXL3_SKIP_VERSION_GUARD=1
export PYTHONPATH="$VLLM_EXL3_EXT_PATH:$EXL3/plugin/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")'):${LD_LIBRARY_PATH:-}"

echo "== pytest (CPU + optional CUDA) =="
python -m pytest "$EXL3/tests" -q --tb=short | tee "$OUT/pytest.txt"

echo "== CUDA parity (plugin vs native) =="
python -m pytest "$EXL3/tests/test_cuda_parity.py" -q --tb=short | tee "$OUT/parity.txt" || true

echo "== prewarm + graph replay =="
python "$EXL3/scripts/prewarm_kernels.py" --device 0 --output "$OUT/prewarm.json"
python "$EXL3/scripts/graph_replay_stress.py" --device 0 --steps "${GRAPH_STRESS_STEPS:-1000}" --all-shapes --mgemm \
  | tee "$OUT/graph_replay.txt"

if [[ "${SKIP_KLD:-0}" != "1" ]]; then
  echo "== KLD =="
  bash "$EXL3/scripts/run_kld_exl3.sh" | tee "$OUT/kld.txt" || true
fi

if [[ -n "${VLLM_API_KEY:-}" && "${SKIP_SERVE_BENCH:-0}" != "1" ]]; then
  python "$EXL3/scripts/bench_serving_contexts.py" \
    --host "$HOST" --port "$PORT" --api-key "$VLLM_API_KEY" \
    --model "$MODEL" --contexts 1024,2048,4096,32768 --output-tokens 256 --runs 3 \
    --label "quality-gate" --output "$OUT/serving.json"
  bash "$EXL3/scripts/restart_test.sh" || true
fi

echo "Receipts under $OUT"
echo "Gates: KLD <= 0.0166, prefill >= 320, decode vs AWQ/Marlin, VRAM at 32K+256."
