#include <cuda_runtime.h>
#include <math.h>
#include "float4_utils.cuh"

#define TILE 32

__inline__ __device__ float gelu_stable(float x) {
    const float inv_sqrt2 = 0.70710678118654752440f;
    return 0.5f * x * (1.0f + erff(x * inv_sqrt2));
}

template <bool UseBias, bool UseGeLU>
__global__ void gemm_kernel(
    const float* __restrict__ X,
    const float* __restrict__ W,
    const float* __restrict__ B,
    float* __restrict__ Y,
    int M, int K, int N_out
) {
    int tid = threadIdx.x;

    __shared__ float s_X[TILE][TILE + 1];
    __shared__ float s_W_T[TILE][TILE + 1];

    int row_load = tid / 8;
    int col_load = (tid % 8) * 4;

    Vec4 acc;

    int global_row = blockIdx.y * TILE + row_load;
    int global_col = blockIdx.x * TILE + col_load;

    for (int t = 0; t < K; t += TILE) {
        if (blockIdx.y * TILE + row_load < M && t + col_load < K) {
            Vec4 x_vec = Vec4::from_float4(
                reinterpret_cast<const float4*>(&X[(blockIdx.y * TILE + row_load) * K + t + col_load])[0]
            );
            s_X[row_load][col_load + 0] = x_vec.x;
            s_X[row_load][col_load + 1] = x_vec.y;
            s_X[row_load][col_load + 2] = x_vec.z;
            s_X[row_load][col_load + 3] = x_vec.w;
        } else {
            s_X[row_load][col_load + 0] = 0.0f;
            s_X[row_load][col_load + 1] = 0.0f;
            s_X[row_load][col_load + 2] = 0.0f;
            s_X[row_load][col_load + 3] = 0.0f;
        }

        if (blockIdx.x * TILE + row_load < N_out && t + col_load < K) {
            Vec4 w_vec = Vec4::from_float4(
                reinterpret_cast<const float4*>(&W[(blockIdx.x * TILE + row_load) * K + t + col_load])[0]
            );
            s_W_T[col_load + 0][row_load] = w_vec.x;
            s_W_T[col_load + 1][row_load] = w_vec.y;
            s_W_T[col_load + 2][row_load] = w_vec.z;
            s_W_T[col_load + 3][row_load] = w_vec.w;
        } else {
            s_W_T[col_load + 0][row_load] = 0.0f;
            s_W_T[col_load + 1][row_load] = 0.0f;
            s_W_T[col_load + 2][row_load] = 0.0f;
            s_W_T[col_load + 3][row_load] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            float x_val = s_X[row_load][k];
            Vec4 w_vec(s_W_T[k][col_load + 0], s_W_T[k][col_load + 1],
                       s_W_T[k][col_load + 2], s_W_T[k][col_load + 3]);
            acc = acc.fma(Vec4(x_val), w_vec);
        }

        __syncthreads();
    }

    if (global_row < M && global_col < N_out) {
        if constexpr (UseBias) {
            if (B != nullptr) {
                Vec4 b_vec = Vec4::from_float4(
                    reinterpret_cast<const float4*>(&B[global_col])[0]
                );
                acc += b_vec;
            }
        }

        if constexpr (UseGeLU) {
            acc.x = gelu_stable(acc.x);
            acc.y = gelu_stable(acc.y);
            acc.z = gelu_stable(acc.z);
            acc.w = gelu_stable(acc.w);
        }

        reinterpret_cast<float4*>(&Y[global_row * N_out + global_col])[0] = acc;
    }
}

void launch_gemm_bias(
    const float* X,
    const float* W,
    const float* B,
    float* Y,
    int M,
    int K,
    int N_out,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(
        (N_out + TILE - 1) / TILE,
         (M + TILE - 1) / TILE
        );
    gemm_kernel<true, false><<<grid, block, 0, stream>>>(X, W, B, Y, M, K, N_out);
}

void launch_gemm(
    const float* X,
    const float* W,
    float* Y,
    int M,
    int K,
    int N_out,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((N_out + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_kernel<false, false><<<grid, block, 0, stream>>>(X, W, nullptr, Y, M, K, N_out);
}

void launch_gemm_gelu(
    const float* X,
    const float* W,
    const float* B,
    float* Y,
    int M,
    int K,
    int N_out,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((N_out + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_kernel<true, true><<<grid, block, 0, stream>>>(X, W, B, Y, M, K, N_out);
}