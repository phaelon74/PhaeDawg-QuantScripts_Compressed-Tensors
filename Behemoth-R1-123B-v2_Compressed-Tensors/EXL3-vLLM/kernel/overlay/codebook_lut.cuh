#pragma once

// Process-wide 16-bit codebook LUT. 3INST/MCG/MUL1 each have 65536 fp16
// entries (128 KiB). Too large for SM86 smem (100 KiB), so this lives in
// global memory and is read with __ldg. Enable with EXL3_GEMV_LUT=1 (default
// on after overlay init unless EXL3_GEMV_LUT=0).
//
// Arithmetic decode remains the fallback when the LUT is not ready.

#include <cuda_fp16.h>
#include <cstdint>

extern __constant__ int exl3_lut_ready_flag;
extern __device__ const __half* exl3_lut_ptrs[3];

__device__ __forceinline__ bool exl3_lut_enabled()
{
    return exl3_lut_ready_flag != 0;
}

template <int cb>
__device__ __forceinline__ __half exl3_lut_decode(uint32_t x)
{
    const __half* table = exl3_lut_ptrs[cb];
    if (table == nullptr)
        return __float2half_rn(0.0f);
    return __ldg(table + (x & 0xffffu));
}

void exl3_lut_ensure();
