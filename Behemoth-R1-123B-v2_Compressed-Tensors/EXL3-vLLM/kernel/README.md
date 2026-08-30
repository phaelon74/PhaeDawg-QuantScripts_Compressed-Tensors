# SM86 kernel overlay

This directory is applied onto a pinned ExLlamaV3 checkout by
`scripts/build_exllamav3_ext.sh` (and `scripts/fork_exllamav3.sh`).

Upstream pin: `0c49587a7c235e6303a6bbedc8b665272ad3a2ea`
(`turboderp-org/exllamav3`). Fork target: `phaelon74/exllamav3` branch
`sm86-decode`.

What the overlay changes:

1. **QTIP GEMV eligibility** (`exl3_gemv.cu`): implicit 3inst (cb=0) at
   K=2 and K=3 when `EXL3_GEMV>=2` or `EXL3_GEMV_3INST=1`. K=4 3inst is
   already eligible upstream. The GEMV kernel `static_assert`s 2/3/4 bpw
   only — do not instantiate K=5..8.
2. **16-bit codebook LUT fill** (`exl3_decode_lut.cu`): 65536 fp16 entries
   per codebook in global memory. Compiled but not invoked yet: without
   `-rdc`, nvcc treats `extern __constant__` as a per-translation-unit
   static (warning 20044), so a flag set in the fill TU never reaches GEMM
   kernels. Arithmetic `decode_3inst` stays live. `EXL3_GEMV_LUT=0` is
   reserved for when the LUT is wired as a GEMV kernel argument.
3. **INT8-activation GEMV on 3inst** (`exl3_gemm.cu`): `EXL3_INT8_GEMV_CB=1`
   also tries `exl3_gemv_int8` for cb=0. Default off. KLD-gate before serving.

Markers: `Phaedawg-SM86-overlay`. Re-running the applier is idempotent.

```bash
export EXL3=/home/phaedawg/kld-exl3-vllm/PhaeDawg-QuantScripts_Compressed-Tensors/Behemoth-R1-123B-v2_Compressed-Tensors/EXL3-vLLM
bash "$EXL3/scripts/fork_exllamav3.sh"
bash "$EXL3/scripts/build_exllamav3_ext.sh"
```
