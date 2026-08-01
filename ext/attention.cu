#include <cuda_runtime.h>
#if __CUDA_ARCH__ >= 800
#include <cuda_pipeline.h>
#endif
#include <math.h>
#include "float4_utils.cuh"

// ViT & Ampere Specific Constants
#define E 768 // Total embedding dimension
#define H_DIM 12  // Number of heads
#define D 64  // Head dimension (768 / 12)
#define VEC_SIZE 4
#define VECS_PER_ROW 16 // 64 / 4
#define THREADS_PER_BLOCK 128
#define WARPS_PER_BLOCK 4
#define ROWS_PER_WARP 2
#define BR 8 // WARPS_PER_BLOCK * ROWS_PER_WARP
#define BC 16

__global__ void flash_attn_2_forward(
    const float* __restrict__ Q, // [B, N, 768]
    const float* __restrict__ K, // [B, N, 768]
    const float* __restrict__ V, // [B, N, 768]
    float* __restrict__ O,       // [B, N, 768]
    int B, int N,
    float scale
) {
    // which batch are we at
    int b = blockIdx.x / H_DIM;
    int h = blockIdx.x % H_DIM;
    // where we are in current seq
    int i_start = blockIdx.y * BR;
    
    
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;

    // 1 warp handles 2 rows. 1 half-warp (16 threads) handles 1 row of D=64.
    int local_row_idx = warp_id * 2 + (lane_id >= 16 ? 1 : 0);
    int global_row_idx = i_start + local_row_idx;
    int vec_idx = lane_id % 16; 

    // Base offset for this batch (b)
    int base_offset = b * N * E;
    
    // Load Q into registers
    Vec4 q_vec;
    if (global_row_idx < N) {
        int q_offset = base_offset + (global_row_idx * E) + (h * D) + (vec_idx * VEC_SIZE);
        q_vec = Vec4::from_float4(reinterpret_cast<const float4*>(Q + q_offset)[0]);
    }

    // Shared memory: declared once, layout differs by architecture
    extern __shared__ float4 s_mem[];
#if __CUDA_ARCH__ >= 800
    // Ampere: 4 tiles for double buffering
    float4* s_K[2];
    float4* s_V[2];
    s_K[0] = s_mem;                                  // [BC][16]
    s_V[0] = s_mem + BC * VECS_PER_ROW;              // [BC][16]
    s_K[1] = s_mem + 2 * BC * VECS_PER_ROW;          // [BC][16]
    s_V[1] = s_mem + 3 * BC * VECS_PER_ROW;          // [BC][16]
#else
    // Turing: 2 tiles, no double buffering
    float4* s_K = s_mem;                             // [BC][16]
    float4* s_V = s_mem + BC * VECS_PER_ROW;         // [BC][16]
#endif
    
    float m = -INFINITY;
    float l = 0.0f;
    Vec4 acc;

#if __CUDA_ARCH__ >= 800
    // Helper lambda for async loading (Ampere+)
    auto load_tile_async = [&](int j_start_val, int buf_idx) {
        for (int load_iter = 0; load_iter < 2; ++load_iter) {
            // Each iteration loads half the tile (8 rows) for K and V. 128 threads can load 16 rows in total, so each thread loads 1/2 row per iteration.
            int flat_idx = tid + load_iter * THREADS_PER_BLOCK;
            int k_row = flat_idx / VECS_PER_ROW;
            int k_vec = flat_idx % VECS_PER_ROW;
            
            if (j_start_val + k_row < N) { // bounds check for tile edge
                int offset = base_offset + ((j_start_val + k_row) * E) + (h * D) + (k_vec * VEC_SIZE);
                __pipeline_memcpy_async(&s_K[buf_idx][flat_idx], &K[offset], sizeof(float4));
                __pipeline_memcpy_async(&s_V[buf_idx][flat_idx], &V[offset], sizeof(float4));
            } else { 
                s_K[buf_idx][flat_idx] = {0.0f, 0.0f, 0.0f, 0.0f};
                s_V[buf_idx][flat_idx] = {0.0f, 0.0f, 0.0f, 0.0f};
            }
        }
    };
#else
    // Helper lambda for synchronous loading (Turing and older)
    auto load_tile_sync = [&](int j_start_val) {
        for (int load_iter = 0; load_iter < 2; ++load_iter) {
            int flat_idx = tid + load_iter * THREADS_PER_BLOCK;
            int k_row = flat_idx / VECS_PER_ROW;
            int k_vec = flat_idx % VECS_PER_ROW;
            
            if (j_start_val + k_row < N) { // bounds check for tile edge
                int offset = base_offset + ((j_start_val + k_row) * E) + (h * D) + (k_vec * VEC_SIZE);
                s_K[flat_idx] = Vec4::from_float4(reinterpret_cast<const float4*>(&K[offset])[0]);
                s_V[flat_idx] = Vec4::from_float4(reinterpret_cast<const float4*>(&V[offset])[0]);
            } else { 
                s_K[flat_idx] = {0.0f, 0.0f, 0.0f, 0.0f};
                s_V[flat_idx] = {0.0f, 0.0f, 0.0f, 0.0f};
            }
        }
    };
#endif

    int num_tiles = (N + BC - 1) / BC;

#if __CUDA_ARCH__ >= 800
    // Prologue: start loading tile 0
    load_tile_async(0, 0);
    __pipeline_commit(); // Commit the async loads for the next tile

    for (int j_step = 0; j_step < num_tiles; ++j_step) {
        int j_start_curr = j_step * BC;
        int j_start_next = (j_step + 1) * BC;
        int buf_curr = j_step % 2;
        int buf_next = (j_step + 1) % 2;

        // Issue async load for NEXT tile
        if (j_start_next < N) {
            load_tile_async(j_start_next, buf_next);
        }
        __pipeline_commit(); // Commit the async loads for the next tile

        // Wait for CURRENT tile to be fully loaded (1 stage remaining in pipeline)
        __pipeline_wait_prior(1);
        __syncthreads();

        float4* cur_K = s_K[buf_curr];
        float4* cur_V = s_V[buf_curr];
#else
    for (int j_step = 0; j_step < num_tiles; ++j_step) {
        int j_start_curr = j_step * BC;

        // Load tile synchronously
        load_tile_sync(j_start_curr);
        __syncthreads();

        float4* cur_K = s_K;
        float4* cur_V = s_V;
#endif

        float S[BC];
        float m_tile = -INFINITY;

        // Compute S_j = Q_i @ K_j^T
        for (int j = 0; j < BC; ++j) {
            Vec4 k_vec = Vec4::from_float4(cur_K[j * VECS_PER_ROW + vec_idx]);
            float dot = q_vec.dot(k_vec);
            
            dot += __shfl_down_sync(0xffffffff, dot, 8);
            dot += __shfl_down_sync(0xffffffff, dot, 4);
            dot += __shfl_down_sync(0xffffffff, dot, 2);
            dot += __shfl_down_sync(0xffffffff, dot, 1);
            
            if (lane_id < 16) dot = __shfl_sync(0xffffffff, dot, 0);
            else dot = __shfl_sync(0xffffffff, dot, 16);
            
            dot *= scale;
            if (j_start_curr + j >= N) dot = -INFINITY;

            S[j] = dot;
            if (dot > m_tile) m_tile = dot;
        }

        // Online Softmax
        float m_new = max(m, m_tile);
        // Using __expf for faster hardware execution
        float exp_prev = __expf(m - m_new);
        float l_tile = 0.0f;

        for (int j = 0; j < BC; ++j) {
            S[j] = __expf(S[j] - m_new);
            l_tile += S[j];
        }
        float l_new = l * exp_prev + l_tile;

        acc *= exp_prev;

        for (int j = 0; j < BC; ++j) {
            Vec4 v_vec = Vec4::from_float4(cur_V[j * VECS_PER_ROW + vec_idx]);
            acc = acc.fma(Vec4(S[j]), v_vec);
        }

        m = m_new;
        l = l_new;
        
        // Sync before the next loop iteration potentially overwrites buffers
        __syncthreads();
    }

    // Finalize output
    if (global_row_idx < N) {
        acc /= l;
        
        int o_offset = base_offset + (global_row_idx * E) + (h * D) + (vec_idx * VEC_SIZE);
        reinterpret_cast<float4*>(O + o_offset)[0] = acc;
    }
}

void flash_attn_2_forward_cuda(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int B,
    int N,
    float scale,
    cudaStream_t stream
) {
    dim3 grid(B * H_DIM, (N + BR - 1) / BR);
    dim3 block(THREADS_PER_BLOCK);

    size_t shared_mem_size = 4 * BC * VECS_PER_ROW * sizeof(float4);

    flash_attn_2_forward<<<grid, block, shared_mem_size, stream>>>(
        Q, K, V, O, B, N, scale
    );
}
