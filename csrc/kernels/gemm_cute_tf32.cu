#include "gemm_kernels.h"

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cutlass/numeric_conversion.h>

namespace vit_cute {
using namespace cute;

constexpr int kBlockM = 32;
constexpr int kBlockN = 32;
constexpr int kBlockK = 32;
constexpr int kThreads = 128;

using RoundTF32 = cutlass::NumericConverter<
    cutlass::tfloat32_t, float, cutlass::FloatRoundStyle::round_to_nearest>;

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

template <bool UseBias, bool UseGeLU>
__global__ void gemm_tf32x3(
    const float* X, const float* W, const float* bias, float* Y,
    int M, int K, int N
) {
    __shared__ float shared_a[kBlockM * kBlockK];
    __shared__ float shared_b[kBlockN * kBlockK];
    auto sA = make_tensor(make_smem_ptr(shared_a),
                         make_layout(Shape<Int<kBlockM>, Int<kBlockK>>{}, LayoutRight{}));
    auto sB = make_tensor(make_smem_ptr(shared_b),
                         make_layout(Shape<Int<kBlockN>, Int<kBlockK>>{}, LayoutRight{}));

    auto mma = make_tiled_mma(SM80_16x8x8_F32TF32TF32F32_TN{},
                              Layout<Shape<_2, _2, _1>>{});
    auto thread_mma = mma.get_slice(threadIdx.x);
    auto a_values = thread_mma.partition_A(sA);
    auto b_values = thread_mma.partition_B(sB);
    auto a_hi = thread_mma.make_fragment_A(a_values);
    auto a_lo = thread_mma.make_fragment_A(a_values);
    auto b_hi = thread_mma.make_fragment_B(b_values);
    auto b_lo = thread_mma.make_fragment_B(b_values);

    auto coordinates = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    auto output_coordinates = thread_mma.partition_C(coordinates);
    auto sum = thread_mma.make_fragment_C(output_coordinates);
    auto main_sum = thread_mma.make_fragment_C(output_coordinates);
    auto correction = thread_mma.make_fragment_C(output_coordinates);
    clear(sum);

    const int first_m = blockIdx.y * kBlockM;
    const int first_n = blockIdx.x * kBlockN;

    for (int first_k = 0; first_k < K; first_k += kBlockK) {
        for (int i = threadIdx.x; i < kBlockM * kBlockK; i += kThreads) {
            int m = first_m + i / kBlockK;
            int k = first_k + i % kBlockK;
            shared_a[i] = (m < M && k < K) ? X[int64_t(m) * K + k] : 0.0f;
        }
        for (int i = threadIdx.x; i < kBlockN * kBlockK; i += kThreads) {
            int n = first_n + i / kBlockK;
            int k = first_k + i % kBlockK;
            shared_b[i] = (n < N && k < K) ? W[int64_t(n) * K + k] : 0.0f;
        }
        __syncthreads();

        CUTE_UNROLL
        for (int i = 0; i < size(a_hi); ++i) {
            float value = a_values(i);
            a_hi(i) = RoundTF32{}(value);
            a_lo(i) = RoundTF32{}(__fsub_rn(value, float(a_hi(i))));
        }
        CUTE_UNROLL
        for (int i = 0; i < size(b_hi); ++i) {
            float value = b_values(i);
            b_hi(i) = RoundTF32{}(value);
            b_lo(i) = RoundTF32{}(__fsub_rn(value, float(b_hi(i))));
        }

        clear(main_sum);
        clear(correction);
        cute::gemm(mma, a_hi, b_hi, main_sum);
        cute::gemm(mma, a_hi, b_lo, correction);
        cute::gemm(mma, a_lo, b_hi, correction);
        CUTE_UNROLL
        for (int i = 0; i < size(sum); ++i) {
            sum(i) += main_sum(i) + correction(i);
        }
        __syncthreads();
    }

    CUTE_UNROLL
    for (int i = 0; i < size(sum); ++i) {
        auto coordinate = output_coordinates(i);
        int m = first_m + get<0>(coordinate);
        int n = first_n + get<1>(coordinate);
        if (m < M && n < N) {
            float value = sum(i);
            if constexpr (UseBias) {
                if (bias != nullptr) value += bias[n];
            }
            if constexpr (UseGeLU) value = gelu(value);
            Y[int64_t(m) * N + n] = value;
        }
    }
}

template <bool UseBias, bool UseGeLU>
void launch(const float* X, const float* W, const float* bias, float* Y,
            int M, int K, int N, cudaStream_t stream) {
    if (M == 0 || N == 0) return;
    dim3 grid((N + kBlockN - 1) / kBlockN, (M + kBlockM - 1) / kBlockM);
    gemm_tf32x3<UseBias, UseGeLU><<<grid, kThreads, 0, stream>>>(
        X, W, bias, Y, M, K, N);
}
}

void launch_gemm(const float* X, const float* W, float* Y,
                 int M, int K, int N, cudaStream_t stream) {
    vit_cute::launch<false, false>(X, W, nullptr, Y, M, K, N, stream);
}

void launch_gemm_bias(const float* X, const float* W, const float* bias, float* Y,
                     int M, int K, int N, cudaStream_t stream) {
    vit_cute::launch<true, false>(X, W, bias, Y, M, K, N, stream);
}

void launch_gemm_gelu(const float* X, const float* W, const float* bias, float* Y,
                     int M, int K, int N, cudaStream_t stream) {
    vit_cute::launch<true, true>(X, W, bias, Y, M, K, N, stream);
}
