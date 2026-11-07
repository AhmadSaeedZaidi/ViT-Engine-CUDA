// Tiny C ABI for testing the real launchers from Python without rebuilding PyTorch bindings.
#include "gemm_kernels.h"

extern "C" int run_gemm(
    const float* X, const float* W, const float* bias, float* Y,
    int M, int K, int N, bool use_bias, bool use_gelu, cudaStream_t stream
) {
    if (use_gelu) {
        launch_gemm_gelu(X, W, bias, Y, M, K, N, stream);
    } else if (use_bias) {
        launch_gemm_bias(X, W, bias, Y, M, K, N, stream);
    } else {
        launch_gemm(X, W, Y, M, K, N, stream);
    }
    return int(cudaGetLastError());
}
