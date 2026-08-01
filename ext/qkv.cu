#include <cuda_runtime.h>
#include "float4_utils.cuh"
#include "gemm_kernels.h"

void qkv_kernel_launcher(
    const float* X,
    const float* W,
    const float* B,
    float* Y,
    int M,
    int K,
    int N_out,
    cudaStream_t stream
) {
    launch_gemm_bias(X, W, B, Y, M, K, N_out, stream);
}