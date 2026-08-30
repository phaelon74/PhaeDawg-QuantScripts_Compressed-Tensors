#pragma once

// Placeholder for a dedicated SM86 GEMV that stages the trellis with cp.async
// and consumes the 16-bit codebook LUT. The LUT itself is wired through
// codebook.cuh (decode_3inst / decode_3inst_2), so the existing GEMV and GEMM
// kernels pick it up without a second launch path.
//
// Future work: a GEMV instance that keeps three trellis stages in flight
// once ncu shows DRAM rather than INT32 as the limiter.

#include "codebook_lut.cuh"
