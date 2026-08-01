#include <cuda_runtime.h>
#include "float4_utils.cuh"
#include "gemm_kernels.h"

void mlp_forward_cuda(
    const float* X,
    const float* W1,
    const float* B1,
    const float* W2,
    const float* B2,
    float* H,
    float* O,
    int M,
    int E,
    int E_expand,
    cudaStream_t stream
) {
    launch_gemm_gelu(X, W1, B1, H, M, E, E_expand, stream);
    launch_gemm_bias(H, W2, B2, O, M, E_expand, E, stream);
}