#include <torch/extension.h>
#include <vector>

// Forward declarations of ops implemented in csrc/ops/
at::Tensor flash_attn_2(at::Tensor Q, at::Tensor K, at::Tensor V, float scale);
at::Tensor patch_embed(at::Tensor img, at::Tensor weights, c10::optional<at::Tensor> bias_opt = c10::nullopt);
at::Tensor pos_encoding(at::Tensor patches, at::Tensor cls_token, at::Tensor pos_embeddings);
std::vector<at::Tensor> mlp_forward(at::Tensor X, at::Tensor W1, at::Tensor B1, at::Tensor W2, at::Tensor B2);
at::Tensor layernorm_forward(at::Tensor X, at::Tensor gamma, at::Tensor beta, float eps);
at::Tensor classifier_forward(at::Tensor X, at::Tensor W, at::Tensor bias);
std::vector<at::Tensor> qkv_proj(at::Tensor X, at::Tensor W, at::Tensor B);
at::Tensor gemm_bias(at::Tensor X, at::Tensor W, at::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "vit_cuda extension (Flash Attention 2 implemented)";
    m.def("flash_attn_2", &flash_attn_2, "Flash Attention 2 forward pass");
    m.def("patch_embed", &patch_embed, "Patch embedding (CUDA)");
    m.def("pos_encoding", &pos_encoding, "Add positional encoding (CUDA)");
    m.def("mlp_forward", &mlp_forward, "Fused MLP Block (GEMM + bias + GeLU)");
    m.def("layernorm_forward", &layernorm_forward, "Fused LayerNorm forward pass");
    m.def("classifier_forward", &classifier_forward, "Fused classifier forward pass");
    m.def("qkv_proj", &qkv_proj, "QKV projection (CUDA)");
    m.def("gemm_bias", &gemm_bias, "General GEMM with bias (CUDA)");
}
