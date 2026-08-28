# Host run order (Linux GPU box)

All GPU/ABI work runs on the 3090 host, not from this Windows checkout.

```bash
cd Behemoth-R1-123B-v2_Compressed-Tensors/EXL3-vLLM

# 1. Freeze ABI + Marlin TP4 baseline (physical GPUs 1-4)
source ~/kld-nightly-vllm/bin/activate
./scripts/capture_manifest.sh
MODEL_DIR=/media/fmodels/TheHouseOfTheDude/Behemoth-R1-123B-v2/NewQuants/AutoRound_GS32_Mixed_72G \
  ./scripts/capture_marlin_baseline.sh

# 2. Native SM86 extension + kernel gates
./scripts/build_exllamav3_ext.sh
export VLLM_EXL3_EXT_PATH=$PWD/build/exllamav3_ext
python scripts/kernel_microbench.py --fail-on-gate
python scripts/select_sm86_crossover.py \
  --microbench results/kernel_microbench.json \
  --output manifests/sm86_crossover.json

# 3. Plugin
pip install -e plugin
export VLLM_PLUGINS=vllm_exl3_sm86
pytest tests -q

# 4. Small Mistral
./scripts/convert_small_mistral.sh
python scripts/validate_small_mistral.py --model build/mistral-exl3-small --tp 1,4

# 5. Behemoth convert (GPU 0, conversion venv)
source ~/exllamav3-convert/bin/activate
./scripts/convert_behemoth_exl3.sh

# 6. Quality gate
source ~/kld-nightly-vllm/bin/activate
python scripts/quality_gate_behemoth.py --model "$OUT_DIR"
python scripts/compare_native_logits.py --model "$OUT_DIR" --tp 4
./scripts/run_kld_exl3.sh

# 7-8. Prefill crossover already selected; graphs after eager is green
python scripts/prewarm_kernels.py
python scripts/graph_replay_stress.py
ENFORCE_EAGER=0 ./scripts/serve_exl3_sm86.sh "$VLLM_API_KEY"   # only after prewarm

# 9. Release
./scripts/restart_test.sh
python scripts/bench_exl3_vs_marlin.py --mode exl3 --model-dir "$OUT_DIR" \
  --output results/exl3_tp4_bench.json
```
