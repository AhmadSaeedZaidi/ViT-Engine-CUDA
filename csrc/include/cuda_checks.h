#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#include <initializer_list>

// Shared correctness checks for the vit_cuda wrappers.
//
// Every op runs on the current CUDA stream (c10::cuda::getCurrentCUDAStream()),
// so we require all inputs to be CUDA, float32, and on the SAME device that the
// current stream targets. Passing a tensor on GPU0 while the current device is
// GPU1 would otherwise launch a kernel that reads GPU0 memory but writes into
// buffers the caller expects on GPU1 — silent corruption.

namespace vit_checks {

// Return the contiguous form of a CUDA float32 tensor, verifying basic props.
inline at::Tensor contig(const at::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    return t.contiguous();
}

// Verify every tensor is CUDA, on the same device, and that this device is the
// one the current CUDA stream is bound to.
inline void check_same_device(
    std::initializer_list<at::Tensor> tensors,
    const char* op
) {
    TORCH_CHECK(tensors.size() > 0, op, ": expected at least one tensor");
    int64_t device = -1;
    for (const at::Tensor& t : tensors) {
        TORCH_CHECK(t.is_cuda(), op, ": all inputs must be CUDA tensors");
        if (device < 0) device = t.get_device();
        TORCH_CHECK(
            t.get_device() == device,
            op, ": all inputs must be on the same CUDA device"
        );
    }
    TORCH_CHECK(device >= 0, op, ": inputs are not on a CUDA device");
    TORCH_CHECK(
        device == at::cuda::current_device(),
        op, ": inputs are on device ", device,
        " but the current CUDA device is ", at::cuda::current_device(),
        " (run torch.cuda.set_device, or move tensors)"
    );
}

}  // namespace vit_checks
