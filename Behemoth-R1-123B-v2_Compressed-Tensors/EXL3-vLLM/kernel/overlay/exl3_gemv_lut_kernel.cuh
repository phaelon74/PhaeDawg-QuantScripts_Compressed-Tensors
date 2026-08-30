#pragma once

// Placeholder for a dedicated SM86 GEMV that stages the trellis with cp.async
// and consumes the 16-bit codebook LUT as a kernel argument. Device symbols
// in codebook.cuh cannot be shared across TUs without -rdc (nvcc 20044).
// Arithmetic decode_3inst remains the live path until that wiring lands.
//
// Future work: a GEMV instance that keeps three trellis stages in flight
// once ncu shows DRAM rather than INT32 as the limiter.

#include "codebook_lut.cuh"
