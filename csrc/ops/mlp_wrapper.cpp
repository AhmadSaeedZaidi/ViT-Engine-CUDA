#include <torch/extension.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>
#include "cuda_checks.h"

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
);

std::vector<at::Tensor> mlp_forward(
    at::Tensor X, 
    at::Tensor W1, at::Tensor B1, 
    at::Tensor W2, at::Tensor B2
) {
    vit_checks::check_same_device({X, W1, B1, W2, B2}, "mlp_forward");

    at::Tensor X_c = vit_checks::contig(X, "X");
    at::Tensor W1_c = vit_checks::contig(W1, "W1");
    at::Tensor B1_c = vit_checks::contig(B1, "B1");
    at::Tensor W2_c = vit_checks::contig(W2, "W2");
    at::Tensor B2_c = vit_checks::contig(B2, "B2");

    TORCH_CHECK(X_c.dim() == 3, "X must be 3D");
    TORCH_CHECK(W1_c.dim() == 2, "W1 must be 2D");
    TORCH_CHECK(W2_c.dim() == 2, "W2 must be 2D");

    int B = X_c.size(0);
    int N = X_c.size(1);
    int E = X_c.size(2);
    int E_expand = W1_c.size(0);

    TORCH_CHECK(W1_c.size(1) == E, "W1 shape mismatch");
    TORCH_CHECK(W2_c.size(0) == E, "W2 shape mismatch");
    TORCH_CHECK(W2_c.size(1) == E_expand, "W2 shape mismatch");
    TORCH_CHECK(B1_c.numel() == E_expand, "B1 size mismatch");
    TORCH_CHECK(B2_c.numel() == E, "B2 size mismatch");

    auto H = at::empty({B, N, E_expand}, X_c.options());
    auto O = at::empty({B, N, E}, X_c.options());

    int M = B * N;

    mlp_forward_cuda(
        X_c.data_ptr<float>(),
        W1_c.data_ptr<float>(), B1_c.data_ptr<float>(),
        W2_c.data_ptr<float>(), B2_c.data_ptr<float>(),
        H.data_ptr<float>(), O.data_ptr<float>(),
        M, E, E_expand,
        c10::cuda::getCurrentCUDAStream()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {O, H};
}