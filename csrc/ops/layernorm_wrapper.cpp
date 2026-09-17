#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "cuda_checks.h"

void launch_layernorm(
    const float* X,
    const float* gamma,
    const float* beta,
    float* Y,
    int B,
    int N,
    float eps,
    cudaStream_t stream
);

at::Tensor layernorm_forward(at::Tensor X, at::Tensor gamma, at::Tensor beta, float eps) {
	vit_checks::check_same_device({X, gamma, beta}, "layernorm_forward");

	auto Xc = vit_checks::contig(X, "X");
	auto gc = vit_checks::contig(gamma, "gamma");
	auto bc = vit_checks::contig(beta, "beta");

	TORCH_CHECK(Xc.dim() == 3, "X must be [B, N, E]");

	int B = (int)Xc.size(0);
	int N = (int)Xc.size(1);
	int E = (int)Xc.size(2);

	TORCH_CHECK((int)gc.numel() == E, "gamma size mismatch");
	TORCH_CHECK((int)bc.numel() == E, "beta size mismatch");

	auto Y = at::empty_like(Xc);

	launch_layernorm(
		Xc.data_ptr<float>(),
		gc.data_ptr<float>(),
		bc.data_ptr<float>(),
		Y.data_ptr<float>(),
		B, N, eps,
		c10::cuda::getCurrentCUDAStream()
	);
	C10_CUDA_KERNEL_LAUNCH_CHECK();

	return Y;
}
