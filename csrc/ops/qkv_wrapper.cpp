#include <torch/extension.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>
#include "cuda_checks.h"

void qkv_kernel_launcher(
    const float* X,
    const float* W,
    const float* B,
    float* Y,
    int M,
    int K,
    int N_out,
    cudaStream_t stream
);

std::vector<at::Tensor> qkv_proj(at::Tensor X, at::Tensor W, at::Tensor B) {
    vit_checks::check_same_device({X, W, B}, "qkv_proj");

    auto Xc = vit_checks::contig(X, "X");
    auto Wc = vit_checks::contig(W, "W");
    auto Bc = vit_checks::contig(B, "B");

    TORCH_CHECK(Xc.dim() == 3, "X must be [B, N, E]");

    int B_batch = (int)Xc.size(0);
    int N = (int)Xc.size(1);
    int E = (int)Xc.size(2);
    int N_out = (int)Wc.size(0);

    TORCH_CHECK(Wc.size(1) == E, "W second dim must match embed dim");
    TORCH_CHECK(Bc.numel() == N_out, "bias size mismatch");
    TORCH_CHECK(N_out % 3 == 0, "N_out must be divisible by 3 (for Q, K, V)");

    int M = B_batch * N;

    auto Y = at::empty({M, N_out}, Xc.options());

    qkv_kernel_launcher(
        Xc.data_ptr<float>(),
        Wc.data_ptr<float>(),
        Bc.data_ptr<float>(),
        Y.data_ptr<float>(),
        M, E, N_out,
        c10::cuda::getCurrentCUDAStream()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto Y_reshaped = Y.view({B_batch, N, 3, E});
    auto q = Y_reshaped.select(2, 0).contiguous();
    auto k = Y_reshaped.select(2, 1).contiguous();
    auto v = Y_reshaped.select(2, 2).contiguous();

    return {q, k, v};
}