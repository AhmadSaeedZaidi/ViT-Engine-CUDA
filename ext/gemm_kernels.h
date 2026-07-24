#ifndef GEMM_KERNELS_H
#define GEMM_KERNELS_H

#define TILE 32

void launch_gemm_bias(
    const float* X, const float* W, const float* B,
    float* Y, int M, int K, int N_out
);

void launch_gemm(
    const float* X, const float* W,
    float* Y, int M, int K, int N_out
);

void launch_gemm_gelu(
    const float* X, const float* W, const float* B,
    float* Y, int M, int K, int N_out
);

#endif