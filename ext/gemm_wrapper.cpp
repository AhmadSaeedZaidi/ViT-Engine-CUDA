#include <torch/extension.h>

void launch_gemm_bias(
    const float* X, const float* W, const float* B,
    float* Y, int M, int K, int N_out
);

at::Tensor gemm_bias(at::Tensor X, at::Tensor W, at::Tensor B) {
    TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(W.is_cuda(), "W must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(X.scalar_type() == at::kFloat, "X must be float32");
    TORCH_CHECK(W.scalar_type() == at::kFloat, "W must be float32");
    TORCH_CHECK(B.scalar_type() == at::kFloat, "B must be float32");

    auto Xc = X.contiguous();
    auto Wc = W.contiguous();
    auto Bc = B.contiguous();

    TORCH_CHECK(Xc.dim() == 3, "X must be [B, N, K]");
    TORCH_CHECK(W.dim() == 2, "W must be [N_out, K]");
    TORCH_CHECK(B.dim() == 1, "B must be [N_out]");

    int B_batch = (int)Xc.size(0);
    int N = (int)Xc.size(1);
    int K = (int)Xc.size(2);
    int N_out = (int)Wc.size(0);

    TORCH_CHECK(Wc.size(1) == K, "W second dim must match K");
    TORCH_CHECK(Bc.numel() == N_out, "bias size mismatch");

    auto Y = at::empty({B_batch, N, N_out}, Xc.options());

    launch_gemm_bias(
        Xc.data_ptr<float>(),
        Wc.data_ptr<float>(),
        Bc.data_ptr<float>(),
        Y.data_ptr<float>(),
        B_batch * N, K, N_out
    );

    return Y;
}