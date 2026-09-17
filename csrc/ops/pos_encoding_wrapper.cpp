#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "cuda_checks.h"

void launch_pos_encoding(
    float* patches,
    float* cls_token,
    float* pos_embed,
    float* out,
    int batch_size,
    cudaStream_t stream
);

at::Tensor pos_encoding(at::Tensor patches, at::Tensor cls_token, at::Tensor pos_embeddings) { // Defines the C++ function accepting PyTorch tensors
	vit_checks::check_same_device({patches, cls_token, pos_embeddings}, "pos_encoding");

	at::Tensor p_c = vit_checks::contig(patches, "patches");
	at::Tensor c_c = vit_checks::contig(cls_token, "cls_token");
	at::Tensor pos_c = vit_checks::contig(pos_embeddings, "pos_embeddings");

	int64_t batch_size = p_c.size(0); // Extracts the batch size
	int64_t num_patches = p_c.size(1); // Extracts the number of patches
	int64_t embed_dim = p_c.size(2); // Extracts the embedding dimension

    // More checks to ensure cls_token and pos_embeddings have the expected shapes
	TORCH_CHECK(c_c.dim() == 2, "cls_token must be [B,embed_dim]"); 
	TORCH_CHECK(c_c.size(0) == batch_size, "cls_token batch size mismatch");
	TORCH_CHECK(c_c.size(1) == embed_dim, "cls_token embed dim mismatch");

	TORCH_CHECK(pos_c.dim() == 2, "pos_embeddings must be [SEQ_LEN,embed_dim]");
	TORCH_CHECK(pos_c.size(1) == embed_dim, "pos_embeddings embed dim mismatch");
	TORCH_CHECK(pos_c.size(0) == (num_patches + 1), "pos_embeddings length must equal num_patches + 1 (including CLS token)");

	at::Tensor out = at::zeros({batch_size, num_patches + 1, embed_dim}, p_c.options()); // Creates an output tensor initialized to zeros with shape [B, SEQ_LEN, embed_dim]

	launch_pos_encoding(
		p_c.data_ptr<float>(),
		c_c.data_ptr<float>(),
		pos_c.data_ptr<float>(),
		out.data_ptr<float>(),
		(int)batch_size,
		c10::cuda::getCurrentCUDAStream()
	);
	C10_CUDA_KERNEL_LAUNCH_CHECK();

	return out;
}

