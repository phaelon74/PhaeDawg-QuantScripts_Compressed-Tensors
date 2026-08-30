# Host run order (Linux GPU box)

All GPU/ABI work runs on the 3090 host, not from this Windows checkout.
Physical GPUs **0 and 5 are reserved**. Serving and microbenches use 1-4.
Stop serve before any kernel microbench.

```bash
export EXL3=/home/phaedawg/kld-exl3-vllm/PhaeDawg-QuantScripts_Compressed-Tensors/Behemoth-R1-123B-v2_Compressed-Tensors/EXL3-vLLM
source /home/phaedawg/kld-exl3-vllm/kld-exl3-vllm/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_EXL3_EXT_PATH="$EXL3/build/exllamav3_ext"
export VLLM_EXL3_SKIP_VERSION_GUARD=1
export PYTHONPATH="$VLLM_EXL3_EXT_PATH:$EXL3/plugin/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")'):${LD_LIBRARY_PATH:-}"
cd "$EXL3"
```

## 0. Phase 0 (env sweep, GB/s, ncu, power A/B)

Stop serve first. Then:

```bash
export CUDA_VISIBLE_DEVICES=1
bash "$EXL3/scripts/host_phase0.sh"
# or ncu only:
bash "$EXL3/scripts/profile_ncu_gate.sh"
```

Gate G0: any 1K/256 decode >= 21 tok/s from `EXL3_GEMV`/`EXL3_INT8_GEMV` alone,
or a documented no with the ncu + GB/s table under `results/phase0/`.

## 1. Overlay build

```bash
bash "$EXL3/scripts/fork_exllamav3.sh"
export TORCH_CUDA_ARCH_LIST=8.6
bash "$EXL3/scripts/build_exllamav3_ext.sh"
pip install -e "$EXL3/plugin"
python -m pytest "$EXL3/tests" -q
```

## 2. Decode kernels + fusion

```bash
export CUDA_VISIBLE_DEVICES=1
python "$EXL3/scripts/kernel_microbench.py" --bitrates 4,5,6 --m 1,2,4,8 \
  --output "$EXL3/results/kernel_microbench_lut.json"
python "$EXL3/scripts/mgemm_microbench.py"
python "$EXL3/scripts/prewarm_kernels.py"
python "$EXL3/scripts/graph_replay_stress.py" --all-shapes --mgemm --steps 1000
```

Gate G1: serve with graphs, 1K/256 >= 23 tok/s, prefill >= 320. Look for
`fused exl3_mgemm decode enabled for K/V pairs` in the worker log.

## 3. Spec decode (the 30 tok/s lever)

```bash
export LAUNCH=/home/phaedawg/kld-exl3-vllm/PhaeDawg-QuantScripts_Compressed-Tensors/VLLM-Launch_Scripts/behemoth123b-r1-v2-exl3-4p25.sh
bash "$EXL3/scripts/host_phase2_specdecode.sh"
# optional draft:
# python "$EXL3/scripts/verify_draft_tokenizer.py" --target "$EXL3_MODEL_DIR" --draft "$EXL3_DRAFT_MODEL"
# EXL3_DRAFT_MODEL=/path/to/mistral-7b-awq bash "$EXL3/scripts/host_phase2_specdecode.sh"
```

Gate G2: 1K/256 >= 30 tok/s with n-gram or draft. Spec decode is lossless;
KLD is unchanged.

## 4. LUT / INT8_CB A/B after overlay rebuild

```bash
EXL3_GEMV_LUT=1 EXL3_GEMV=2 bash "$LAUNCH" "$VLLM_API_KEY"
# 3inst int8 activations (KLD-gate before keeping):
EXL3_INT8_GEMV_CB=1 EXL3_INT8_GEMV=2 bash "$LAUNCH" "$VLLM_API_KEY"
bash "$EXL3/scripts/run_kld_exl3.sh"
```

Gate G3: `gate` M=1 >= 550 GB/s in the microbench JSON, and >= 26 tok/s
without spec decode.

## 5. Qualification

```bash
bash "$EXL3/scripts/host_quality_gate.sh"
python "$EXL3/scripts/validate_exl3_generality.py" --roots /media/fmodels
```

## 6. Serving (production)

```bash
export LAUNCH=/home/phaedawg/kld-exl3-vllm/PhaeDawg-QuantScripts_Compressed-Tensors/VLLM-Launch_Scripts/behemoth123b-r1-v2-exl3-4p25.sh
bash "$LAUNCH" "$VLLM_API_KEY"
python "$EXL3/scripts/bench_serving_contexts.py" --host 10.9.99.22 \
  --api-key "$VLLM_API_KEY" --model Behemoth-R1-123B-v2-EXL3-4.25-H6
```

Kernel env vars the overlay reads (export before `vllm serve`):

| Var | Default | Meaning |
| --- | --- | --- |
| `EXL3_GEMV` | 1 | 0=off, 1=heuristic, 2=force eligible GEMV |
| `EXL3_GEMV_3INST` | unset | 1=allow cb=0 GEMV at K!=4 |
| `EXL3_GEMV_LUT` | reserved | LUT fill is compiled, not wired into decode yet |
| `EXL3_INT8_GEMV` | 2 in upstream | 0=off |
| `EXL3_INT8_GEMV_CB` | 0 | 1=try int8 activations on 3inst |
| `EXL3_INT8_GEMV_MAX_K` | arch default | 5 on Ampere |
| `EXL3_NGRAM_SPEC` | 0 | 1=ngram speculative decode |
| `EXL3_SPECULATIVE_CONFIG` | unset | raw vLLM spec JSON |
| `EXL3_CUDAGRAPH_CAPTURE_SIZES` | `[1,2,3,4,5,6,8]` | must cover spec M |
