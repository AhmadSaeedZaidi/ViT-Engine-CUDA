#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "cuda_checks.h"

void launch_classifier(
    const float* X,
    const float* W,
    const float* bias,
    float* Y,
    int B,
    int num_classes,
    cudaStream_t stream
);

at::Tensor classifier_forward(at::Tensor X, at::Tensor W, at::Tensor bias) {
    vit_checks::check_same_device({X, W, bias}, "classifier_forward");

    auto Xc = vit_checks::contig(X, "X");
    auto Wc = vit_checks::contig(W, "W");
    auto bc = vit_checks::contig(bias, "bias");

    TORCH_CHECK(Xc.dim() == 3, "X must be [B, SEQ_LEN, E]");
    TORCH_CHECK(Wc.dim() == 2, "W must be [num_classes, E]");
    TORCH_CHECK(bc.dim() == 1, "bias must be [num_classes]");

    int B = (int)Xc.size(0);
    int SEQ = (int)Xc.size(1);
    int E = (int)Xc.size(2);
    int num_classes = (int)Wc.size(0);

    TORCH_CHECK((int)Wc.size(1) == E, "W second dim must match embed dim");
    TORCH_CHECK((int)bc.numel() == num_classes, "bias size mismatch");

    auto Y = at::empty({B, num_classes}, X.options());

    launch_classifier(
        Xc.data_ptr<float>(),
        Wc.data_ptr<float>(),
        bc.data_ptr<float>(),
        Y.data_ptr<float>(),
        B, num_classes,
        c10::cuda::getCurrentCUDAStream()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return Y;
}
