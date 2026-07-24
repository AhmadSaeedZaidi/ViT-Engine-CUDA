#include <cuda_runtime.h>
#include "float4_utils.cuh"
#define SEQ_LEN 197 // 196 patches + 1 CLS token
#define EMBED_DIM 768
#define VEC_SIZE 4
#define THREADS_PER_BLOCK 192 // 768 / 4

__global__ void pos_encoding_kernel(const float* __restrict__ patches, const float* __restrict__ cls_token, const float* __restrict__ pos_embed, float* __restrict__ out) {
    int batch_idx = blockIdx.y;
    int seq_idx = blockIdx.x;
    int tid = threadIdx.x;

    int out_offset = (batch_idx * SEQ_LEN * EMBED_DIM) + (seq_idx * EMBED_DIM) + (tid * VEC_SIZE);

    Vec4 p_val = Vec4::from_float4(reinterpret_cast<const float4*>(&pos_embed[seq_idx * EMBED_DIM])[tid]);

    if (seq_idx == 0) {
        Vec4 c_val = Vec4::from_float4(reinterpret_cast<const float4*>(&cls_token[0])[tid]);
        Vec4 res = c_val + p_val;
        reinterpret_cast<float4*>(&out[out_offset])[0] = res;
    } 
    else {
        int patch_offset = (batch_idx * 196 * EMBED_DIM) + ((seq_idx - 1) * EMBED_DIM) + (tid * VEC_SIZE);
        Vec4 patch_val = Vec4::from_float4(reinterpret_cast<const float4*>(&patches[patch_offset])[0]);
        Vec4 res = patch_val + p_val;
        reinterpret_cast<float4*>(&out[out_offset])[0] = res;
    }
}

void launch_pos_encoding(float* patches, float* cls_token, float* pos_embed, float* out, int batch_size) {
    dim3 grid(SEQ_LEN, batch_size);
    dim3 block(THREADS_PER_BLOCK);
    pos_encoding_kernel<<<grid, block>>>(patches, cls_token, pos_embed, out);
}