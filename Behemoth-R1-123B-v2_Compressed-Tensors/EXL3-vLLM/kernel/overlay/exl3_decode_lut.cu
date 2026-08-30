#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <mutex>

// codebook.cuh uses half_uint16 / half2_uint32 from util.cuh; those types are
// not declared in codebook.cuh itself.
#include "../util.cuh"
#include "codebook.cuh"

__constant__ int exl3_lut_ready_flag = 0;
__device__ const __half* exl3_lut_ptrs[3] = {nullptr, nullptr, nullptr};

static __half* d_lut[3] = {nullptr, nullptr, nullptr};
static bool lut_ready = false;
static std::mutex lut_mu;

__global__ void exl3_lut_fill_kernel(__half* out, int cb)
{
    int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= 65536)
        return;
    uint32_t x = static_cast<uint32_t>(i);
    if (cb == 0)
        out[i] = decode_3inst<0>(x);
    else if (cb == 1)
        out[i] = decode_3inst<1>(x);
    else
        out[i] = decode_3inst<2>(x);
}

__global__ void exl3_lut_bind_kernel(const __half* p0, const __half* p1, const __half* p2)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        exl3_lut_ptrs[0] = p0;
        exl3_lut_ptrs[1] = p1;
        exl3_lut_ptrs[2] = p2;
    }
}

static int lut_env_on()
{
    const char* e = std::getenv("EXL3_GEMV_LUT");
    if (!e)
        return 1;
    return atoi(e);
}

void exl3_lut_ensure()
{
    if (!lut_env_on())
        return;
    std::lock_guard<std::mutex> lock(lut_mu);
    if (lut_ready)
        return;

    for (int cb = 0; cb < 3; ++cb)
    {
        cudaMalloc(reinterpret_cast<void**>(&d_lut[cb]), 65536 * sizeof(__half));
        exl3_lut_fill_kernel<<<256, 256>>>(d_lut[cb], cb);
    }
    exl3_lut_bind_kernel<<<1, 32>>>(d_lut[0], d_lut[1], d_lut[2]);
    int one = 1;
    cudaMemcpyToSymbol(exl3_lut_ready_flag, &one, sizeof(int));
    cudaDeviceSynchronize();
    lut_ready = true;
}
