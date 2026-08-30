# SM86 kernel overlay

This directory is applied onto a pinned ExLlamaV3 checkout by
`scripts/build_exllamav3_ext.sh` (and `scripts/fork_exllamav3.sh`).

Upstream pin: `0c49587a7c235e6303a6bbedc8b665272ad3a2ea`
(`turboderp-org/exllamav3`). Fork target: `phaelon74/exllamav3` branch
`sm86-decode`.

What the overlay changes:

1. **QTIP GEMV eligibility** (`exl3_gemv.cu`): K=2..8, implicit 3inst (cb=0)
   when `EXL3_GEMV>=2` or `EXL3_GEMV_3INST=1`, plus compiled instances for
   those shapes.
2. **16-bit codebook LUT** (`codebook_lut.cuh`, `exl3_decode_lut.cu`):
   65536 fp16 entries per codebook in global memory, `__ldg` in
   `decode_3inst` / `decode_3inst_2`. Default on; `EXL3_GEMV_LUT=0` disables.
   A full 16-bit table is 128 KiB and does not fit in SM86 smem (100 KiB).
3. **INT8-activation GEMV on 3inst** (`exl3_gemm.cu`): `EXL3_INT8_GEMV_CB=1`
   also tries `exl3_gemv_int8` for cb=0. Default off. KLD-gate before serving.

Markers: `Phaedawg-SM86-overlay`. Re-running the applier is idempotent.

```bash
export EXL3=/home/phaedawg/kld-exl3-vllm/PhaeDawg-QuantScripts_Compressed-Tensors/Behemoth-R1-123B-v2_Compressed-Tensors/EXL3-vLLM
bash "$EXL3/scripts/fork_exllamav3.sh"
bash "$EXL3/scripts/build_exllamav3_ext.sh"
```
