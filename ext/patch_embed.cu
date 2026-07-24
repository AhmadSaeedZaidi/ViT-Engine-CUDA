#include <cuda_runtime.h>
#include <math.h>
#include "float4_utils.cuh"

#define IMG_SIZE 224
#define PATCH_SIZE 16
#define PATCHES_PER_SIDE (IMG_SIZE / PATCH_SIZE)
#define CHANNELS 3
#define EMBED_DIM 768
#define NUM_PATCHES 196
#define PATCH_VOLUME 768 // 3 * 16 * 16

__inline__ __device__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

// 192 threads per block (6 warps).
__global__ void patch_embed_kernel(const float* __restrict__ img, const float* __restrict__ weights, float* __restrict__ out) {
    int patch_idx = blockIdx.x; 
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x; // 0 to 191

    // 192 float4s = 768 floats (use float4 for shared memory to avoid non-trivial constructor)
    __shared__ float4 patch_s[192];

    // Top-left corner of the patch in the original image (y,x)
    int patch_row = (patch_idx / PATCHES_PER_SIDE) * PATCH_SIZE;
    int patch_col = (patch_idx % PATCHES_PER_SIDE) * PATCH_SIZE;

    const float* batch_img = img + (batch_idx * CHANNELS * IMG_SIZE * IMG_SIZE);

    // Each thread loads exactly 1 float4 from the image
    int c = tid / 64;           // Channel 0, 1, or 2
    int tid_in_c = tid % 64;    // 0 to 63
    int py = tid_in_c / 4;      // 0 to 15 (row within patch)
    int px_vec = tid_in_c % 4;  // 0 to 3 (float4 index within row)

    // 3D Tensor in Channel-major order: [C][H][W]
    int global_offset = c * IMG_SIZE * IMG_SIZE + (patch_row + py) * IMG_SIZE + (patch_col + px_vec * 4);
    
    // Load 1 float4 (16 bytes) cohesively per thread
    patch_s[tid] = Vec4::from_float4(reinterpret_cast<const float4*>(batch_img + global_offset)[0]);

    __syncthreads();

    // Now, compute the dot products. We have 768 output dimensions.
    // 6 warps, each computes 4 output dimensions per iteration. 
    // Total 24 dimensions per iteration. 768 / 24 = 32 iterations.
    
    int warp_id = tid / 32;     // 0 to 5
    int lane_id = tid % 32;     // 0 to 31
    int out_batch_offset = batch_idx * NUM_PATCHES * EMBED_DIM; // Output offset for the current batch

    // 32 iterations to cover all 768 output dimensions 
    for (int iter = 0; iter < 32; iter++) {
        // Each warp computes 4 output dimensions in this iteration
        int out_dim_base = iter * 24 + warp_id * 4;
        
        // 4 output accumulators (one per output dimension)
        Vec4 sum;

        // Warp loops over the 192 float4s of the patch.
        // 192 / 32 = 6 float4s per thread
        const float4* w_base = reinterpret_cast<const float4*>(weights);
        
        #pragma unroll
        for (int step = 0; step < 6; step++) {
            int i = step * 32 + lane_id;
            Vec4 p = Vec4::from_float4(patch_s[i]);
            
            Vec4 w0 = Vec4::from_float4(w_base[(out_dim_base + 0) * 192 + i]);
            Vec4 w1 = Vec4::from_float4(w_base[(out_dim_base + 1) * 192 + i]);
            Vec4 w2 = Vec4::from_float4(w_base[(out_dim_base + 2) * 192 + i]);
            Vec4 w3 = Vec4::from_float4(w_base[(out_dim_base + 3) * 192 + i]);
            
            sum.x += p.dot(w0);
            sum.y += p.dot(w1);
            sum.z += p.dot(w2);
            sum.w += p.dot(w3);
        }

        sum.x = warp_reduce_sum(sum.x);
        sum.y = warp_reduce_sum(sum.y);
        sum.z = warp_reduce_sum(sum.z);
        sum.w = warp_reduce_sum(sum.w);

        // Store back exactly 4 floats (1 float4) aligned perfectly in memory
        if (lane_id == 0) {
            int out_offset = out_batch_offset + patch_idx * EMBED_DIM + out_dim_base;
            reinterpret_cast<float4*>(out + out_offset)[0] = sum;
        }
    }
}

void launch_patch_embed(float* img, float* weights, float* out, int B) {
    dim3 grid(196, B); 
    dim3 block(192); // Optimally reduced to exactly 192 threads
    // Overall input image is [B][3][224][224] and weights are [768][768], output is [B][196][768]
    patch_embed_kernel<<<grid, block>>>(img, weights, out);
}