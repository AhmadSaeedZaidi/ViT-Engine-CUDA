#include <torch/extension.h>
#include <vector>

void qkv_kernel_launcher(
    const float* X, const float* W, const float* B,
    float* Y, int M, int K, int N_out
);

std::vector<at::Tensor> qkv_proj(at::Tensor X, at::Tensor W, at::Tensor B) {
    TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(W.is_cuda(), "W must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(X.scalar_type() == at::kFloat, "X must be float32");
    TORCH_CHECK(W.scalar_type() == at::kFloat, "W must be float32");
    TORCH_CHECK(B.scalar_type() == at::kFloat, "B must be float32");

    auto Xc = X.contiguous();
    auto Wc = W.contiguous();
    auto Bc = B.contiguous();

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
        M, E, N_out
    );

    auto Y_reshaped = Y.view({B_batch, N, 3, E});
    auto q = Y_reshaped.select(2, 0).contiguous();
    auto k = Y_reshaped.select(2, 1).contiguous();
    auto v = Y_reshaped.select(2, 2).contiguous();

    return {q, k, v};
}