#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "cuda_checks.h"

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

at::Tensor gemm_bias(at::Tensor X, at::Tensor W, at::Tensor B) {
    vit_checks::check_same_device({X, W, B}, "gemm_bias");

    auto Xc = vit_checks::contig(X, "X");
    auto Wc = vit_checks::contig(W, "W");
    auto Bc = vit_checks::contig(B, "B");

    TORCH_CHECK(Xc.dim() == 2 || Xc.dim() == 3, "X must be [N, K] or [B, N, K]");
    TORCH_CHECK(W.dim() == 2, "W must be [N_out, K]");
    TORCH_CHECK(B.dim() == 1, "B must be [N_out]");

    int B_batch, N, K;
    if (Xc.dim() == 3) {
        B_batch = (int)Xc.size(0);
        N = (int)Xc.size(1);
        K = (int)Xc.size(2);
    } else {
        B_batch = 1;
        N = (int)Xc.size(0);
        K = (int)Xc.size(1);
    }
    int N_out = (int)Wc.size(0);

    TORCH_CHECK(Wc.size(1) == K, "W second dim must match K");
    TORCH_CHECK(Bc.numel() == N_out, "bias size mismatch");

    int M = B_batch * N;
    auto Y = at::empty({B_batch, N, N_out}, Xc.options());

    launch_gemm_bias(
        Xc.data_ptr<float>(),
        Wc.data_ptr<float>(),
        Bc.data_ptr<float>(),
        Y.data_ptr<float>(),
        M, K, N_out,
        c10::cuda::getCurrentCUDAStream()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (Xc.dim() == 3) {
        return Y;  // [B, N, N_out]
    }
    return Y.view({M, N_out});
}