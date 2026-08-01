#include <cuda_runtime.h>
#include <math.h>

#define E 768
#define WARPS_PER_BLOCK 24 // 768 / 32

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
    __shared__ float shared_val[WARPS_PER_BLOCK];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);
    
    if (lane == 0) {
        shared_val[wid] = val;
    }
    __syncthreads();

    // Only the first warp reads the shared memory and reduces the final sum
    val = (threadIdx.x < WARPS_PER_BLOCK) ? shared_val[lane] : 0.0f;
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

__global__ void layernorm_kernel(const float* __restrict__ X, const float* __restrict__ gamma, const float* __restrict__ beta, float* __restrict__ Y, int B, int N, float eps) {
    int b = blockIdx.y;
    int n = blockIdx.x;
    int tid = threadIdx.x;

    int offset = (b * N * E) + (n * E) + tid;
    float val = X[offset];

    // 1. Compute Mean
    float sum = block_reduce_sum(val);
    __shared__ float mean;
    if (tid == 0) mean = sum / E;
    __syncthreads();

    // 2. Compute Variance
    float diff = val - mean;
    float sq_diff = diff * diff;
    float var_sum = block_reduce_sum(sq_diff);
    
    __shared__ float variance;
    if (tid == 0) variance = var_sum / E;
    __syncthreads();

    // 3. Normalize and scale
    // rsqrtf executes directly on the Ampere Special Function Unit
    float norm_val = diff * rsqrtf(variance + eps);
    Y[offset] = norm_val * gamma[tid] + beta[tid];
}

void launch_layernorm(
    const float* X,
    const float* gamma,
    const float* beta,
    float* Y,
    int B,
    int N,
    float eps,
    cudaStream_t stream
) {
    dim3 grid(N, B);
    dim3 block(E);
    layernorm_kernel<<<grid, block, 0, stream>>>(X, gamma, beta, Y, B, N, eps);
}