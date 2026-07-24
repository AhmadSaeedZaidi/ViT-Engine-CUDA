#include <cuda_runtime.h>
#include "float4_utils.cuh"

#define SEQ_LEN 197
#define EMBED_DIM 768
#define VEC_SIZE 4
#define THREADS_PER_BLOCK 192 // 768 / 4

// Warp-level reduction
__inline__ __device__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

// Block-level reduction
__inline__ __device__ float block_reduce_sum(float val) {
    __shared__ float shared_val[6]; // 192 threads = 6 warps
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);
    
    if (lane == 0) {
        shared_val[wid] = val;
    }
    __syncthreads();

    // The first warp reduces the 6 partial sums
    val = (threadIdx.x < 6) ? shared_val[lane] : 0.0f;
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

__global__ void classifier_kernel(const float* __restrict__ X, const float* __restrict__ W, const float* __restrict__ bias, float* __restrict__ Y, int num_classes) {
    int b = blockIdx.y;
    int c = blockIdx.x;
    int tid = threadIdx.x;

    // Isolate the CLS token (seq_idx = 0)
    int x_offset = (b * SEQ_LEN * EMBED_DIM) + (tid * VEC_SIZE);
    
    // W is stored as [Num_Classes, 768] in standard PyTorch format
    int w_offset = (c * EMBED_DIM) + (tid * VEC_SIZE);

    const float4* x_vec = reinterpret_cast<const float4*>(&X[x_offset]);
    const float4* w_vec = reinterpret_cast<const float4*>(&W[w_offset]);

    Vec4 x_val = Vec4::from_float4(x_vec[0]);
    Vec4 w_val = Vec4::from_float4(w_vec[0]);

    float sum = x_val.dot(w_val);

    // Sum across all 192 threads in the block to get the full dot product
    sum = block_reduce_sum(sum);

    // Thread 0 adds the bias and writes the final class score to memory
    if (tid == 0) {
        Y[b * num_classes + c] = sum + bias[c];
    }
}

void launch_classifier(const float* X, const float* W, const float* bias, float* Y, int B, int num_classes) {
    // Grid maps Y-axis to Batch Size and X-axis to Number of Classes
    dim3 grid(num_classes, B);
    dim3 block(THREADS_PER_BLOCK);
    classifier_kernel<<<grid, block>>>(X, W, bias, Y, num_classes);
}