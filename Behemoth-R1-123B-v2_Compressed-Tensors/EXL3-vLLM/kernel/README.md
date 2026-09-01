# SM86 kernel overlay

This directory is applied onto a pinned ExLlamaV3 checkout by
`scripts/build_exllamav3_ext.sh` (and `scripts/fork_exllamav3.sh`).

Upstream pin: `0c49587a7c235e6303a6bbedc8b665272ad3a2ea`
(`turboderp-org/exllamav3`). Fork target: `phaelon74/exllamav3` branch
`sm86-decode`.

What the overlay changes:

1. **QTIP GEMV dispatch** (`exl3_gemv.cu`): implicit 3inst (cb=0) at K=2
   and K=3 when `EXL3_GEMV>=2` or `EXL3_GEMV_3INST=1`. At K=4/M=1/SM86,
   heuristic mode uses the measured Behemoth TP4 policy: regular q/k/v and
   narrow GEMV o/gate/up/down.
2. **Experimental K5/K6 GEMV** (`exl3_gemv_kernel.cuh`): M=1 cb0 fp16
   calls use a narrow 512-thread kernel when `EXL3_GEMV_K56` is nonzero.
   Mode 1 stages each coalesced 40/48-word tile in warp-private shared
   memory. Mode 2 uses conflict-free shared reads for words 0–31 and warp
   shuffles for words 32–47, avoiding the staged path's bank conflicts
   without paying for two shuffles per source. Both remain opt-in.
3. **Experimental K4 cb0 arithmetic** (`exl3_gemv_kernel.cuh`): mode 1
   forces one `mad.lo.u32` per code, while mode 2 batches eight independent
   MAD/LOP3 chains to expose more instruction-level parallelism. Set
   `EXL3_GEMV_K4_ARITH=1` or `2`; mode 0 retains the current kernel.
4. **16-bit codebook LUT fill** (`exl3_decode_lut.cu`): 65536 fp16 entries
   per codebook in global memory. Compiled but not invoked yet: without
   `-rdc`, nvcc treats `extern __constant__` as a per-translation-unit
   static (warning 20044), so a flag set in the fill TU never reaches GEMM
   kernels. Arithmetic `decode_3inst` stays live. `EXL3_GEMV_LUT=0` is
   reserved for when the LUT is wired as a GEMV kernel argument.
5. **INT8-activation GEMV on 3inst** (`exl3_gemm.cu`): `EXL3_INT8_GEMV_CB=1`
   also tries `exl3_gemv_int8` for cb=0. Default off. KLD-gate before serving.

Markers: `Phaedawg-SM86-overlay`. Re-running the applier is idempotent.

```bash
export EXL3=/home/phaedawg/kld-exl3-vllm/PhaeDawg-QuantScripts_Compressed-Tensors/Behemoth-R1-123B-v2_Compressed-Tensors/EXL3-vLLM
bash "$EXL3/scripts/fork_exllamav3.sh"
bash "$EXL3/scripts/build_exllamav3_ext.sh"
```

K4 arithmetic acceptance sequence on one RTX 3090:

```bash
EXL3_GEMV_K4_ARITH=2 python -m pytest tests/test_cuda_parity.py \
  -k 'k4_arithmetic_gemv' -q

for mode in 0 1 2; do
  if [[ "$mode" == 0 ]]; then
    unset EXL3_GEMV_K4_ARITH
  else
    export EXL3_GEMV_K4_ARITH="$mode"
  fi
  python scripts/kernel_microbench.py \
    --device 0 --bitrates 4 --m 1 \
    --shapes o_proj,gate_proj,up_proj,down_proj \
    --warmup 10 --iters 100 \
    --output "results/k4_arith_mode${mode}_m1.json"
done
```

K5/K6 acceptance sequence on one RTX 3090:

```bash
export EXL3_GEMV=1
export EXL3_GEMV_SMEM=0

EXL3_GEMV_K56=2 python -m pytest tests/test_cuda_parity.py \
  -k 'k56_3inst_gemv' -q

unset EXL3_GEMV_K56
python scripts/kernel_microbench.py \
  --device 0 --bitrates 5,6 --m 1 \
  --shapes q_proj,k_proj,o_proj,gate_proj,down_proj \
  --warmup 10 --iters 50 \
  --output results/k56_regular_m1.json

python scripts/kernel_microbench.py \
  --device 0 --bitrates 5,6 --m 1 \
  --shapes q_proj,k_proj,o_proj,gate_proj,down_proj \
  --gemv-k56 --warmup 10 --iters 50 \
  --output results/k56_staged_m1.json

python scripts/kernel_microbench.py \
  --device 0 --bitrates 5,6 --m 1 \
  --shapes q_proj,k_proj,o_proj,gate_proj,down_proj \
  --gemv-k56 2 --warmup 10 --iters 50 \
  --output results/k56_hybrid_m1.json

sudo -E bash scripts/profile_ncu_gate.sh \
  results/phase0/ncu_down_k5_hybrid down_proj 5 hybrid
```

Do not serve with `EXL3_GEMV_K56` enabled unless parity passes, ptxas reports no
spills, occupancy is at least 33%, and every selected projection beats the
regular-kernel baseline.
