#pragma once

// Process-wide 16-bit codebook LUT. 3INST/MCG/MUL1 each have 65536 fp16
// entries (128 KiB). Too large for SM86 smem (100 KiB), so this lives in
// global memory and is read with __ldg.
//
// Do not include this header from codebook.cuh. Without -rdc, nvcc treats
// `extern __constant__` / `extern __device__` as a *per-TU static* (warning
// 20044), so a flag set in exl3_decode_lut.cu is invisible to GEMM/GEMV
// kernels. Arithmetic decode_3inst stays the live path until a kernel-arg
// or relocatable-device-code wiring exists.
//
// Fill tables with EXL3_GEMV_LUT=1 (default). EXL3_GEMV_LUT=0 skips fill.

#include <cuda_fp16.h>
#include <cstdint>

void exl3_lut_ensure();
