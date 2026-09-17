#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "cuda_checks.h"

void flash_attn_2_forward_cuda(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int B,
    int N,
    float scale,
    cudaStream_t stream
);

at::Tensor flash_attn_2(at::Tensor Q, at::Tensor K, at::Tensor V, float scale) {
    vit_checks::check_same_device({Q, K, V}, "flash_attn_2");

    auto Qc = vit_checks::contig(Q, "Q");
    auto Kc = vit_checks::contig(K, "K");
    auto Vc = vit_checks::contig(V, "V");

    // Q, K, V are expected to be [B, N, 768]
    TORCH_CHECK(Qc.dim() == 3, "Expected 3D tensor [B, N, 768]");
    TORCH_CHECK(Kc.dim() == 3, "K must be [B, N, 768]");
    TORCH_CHECK(Vc.dim() == 3, "V must be [B, N, 768]");

    TORCH_CHECK(Qc.size(0) == Kc.size(0) && Qc.size(1) == Kc.size(1) && Qc.size(2) == Kc.size(2),
                "K shape must match Q shape");
    TORCH_CHECK(Qc.size(0) == Vc.size(0) && Qc.size(1) == Vc.size(1) && Qc.size(2) == Vc.size(2),
                "V shape must match Q shape");

    int B = Qc.size(0);
    int N = Qc.size(1);
    int E = Qc.size(2);

    TORCH_CHECK(E == 768, "E dimension must be 768");

    auto O = at::zeros_like(Qc);

    flash_attn_2_forward_cuda(
        Qc.data_ptr<float>(),
        Kc.data_ptr<float>(),
        Vc.data_ptr<float>(),
        O.data_ptr<float>(),
        B, N, scale,
        c10::cuda::getCurrentCUDAStream()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return O;
}
