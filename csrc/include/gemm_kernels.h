#ifndef GEMM_KERNELS_H
#define GEMM_KERNELS_H

#include <cuda_runtime.h>

#define TILE 32

void launch_gemm_bias(
    const float* X,
    const float* W,
    const float* B,
    float* Y,
    int M,
    int K,
    int N_out,
    cudaStream_t stream
);

void launch_gemm(
    const float* X,
    const float* W,
    float* Y,
    int M,
    int K,
    int N_out,
    cudaStream_t stream
);

void launch_gemm_gelu(
    const float* X,
    const float* W,
    const float* B,
    float* Y,
    int M,
    int K,
    int N_out,
    cudaStream_t stream
);

#endif