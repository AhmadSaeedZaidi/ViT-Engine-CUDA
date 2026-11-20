#include "gemm_kernels.h"

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/swizzle.hpp>
#include <cutlass/numeric_conversion.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>

#ifndef VIT_CUTE_OVERLAP
#define VIT_CUTE_OVERLAP 1
#endif
#ifndef VIT_CUTE_SWIZZLE
#define VIT_CUTE_SWIZZLE 1
#endif
#ifndef VIT_CUTE_USE_BF16
#define VIT_CUTE_USE_BF16 0
#endif
#ifndef VIT_CUTE_BF16_PAIRS
#define VIT_CUTE_BF16_PAIRS 1
#endif

namespace vit_cute {
using namespace cute;

constexpr int kBlockM = 32;
constexpr int kBlockN = 32;
constexpr int kBlockK = 32;
constexpr int kThreads = 128;

using RoundTF32 = cutlass::NumericConverter<
    cutlass::tfloat32_t, float, cutlass::FloatRoundStyle::round_to_nearest>;

template <int Rows>
constexpr auto shared_layout() {
    auto row_major = make_layout(Shape<Int<Rows>, Int<kBlockK>>{}, LayoutRight{});
    if constexpr (VIT_CUTE_SWIZZLE && VIT_CUTE_USE_BF16 && VIT_CUTE_BF16_PAIRS) {
        return composition(Swizzle<2, 3, 2>{}, row_major);
    } else if constexpr (VIT_CUTE_SWIZZLE) {
        return composition(Swizzle<3, 2, 3>{}, row_major);
    } else {
        return row_major;
    }
}

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

template <int Rows>
__device__ void prefetch_tile(const float* input, float* shared,
                             int first_row, int first_k, int row_count, int K) {
    constexpr int vectors_per_row = kBlockK / 4;
    auto layout = shared_layout<Rows>();
    for (int i = threadIdx.x; i < Rows * vectors_per_row; i += kThreads) {
        int tile_row = i / vectors_per_row;
        int tile_k = (i % vectors_per_row) * 4;
        int row = first_row + tile_row;
        int k = first_k + tile_k;
        int valid = 0;
        const float* source = input;
        if (row < row_count && k < K) {
            source = input + int64_t(row) * K + k;
            valid = (K - k < 4) ? K - k : 4;
        }
        float* destination = shared + layout(tile_row, tile_k);
        if ((reinterpret_cast<uintptr_t>(source) & 15) == 0) {
            __pipeline_memcpy_async(destination, source, 16, (4 - valid) * sizeof(float));
        } else {
            CUTE_UNROLL
            for (int j = 0; j < 4; ++j) {
                __pipeline_memcpy_async(destination + j, j < valid ? source + j : input,
                                        4, j < valid ? 0 : 4);
            }
        }
    }
}

template <class Values, class Fragment>
__device__ void split_tf32(const Values& values, Fragment& hi, Fragment& lo) {
    CUTE_UNROLL
    for (int i = 0; i < size(hi); ++i) {
        float value = values(i);
        hi(i) = RoundTF32{}(value);
        lo(i) = RoundTF32{}(__fsub_rn(value, float(hi(i))));
    }
}

template <class Values, class Fragment>
__device__ void split_bf16(const Values& values, Fragment& hi, Fragment& mid, Fragment& lo) {
#if VIT_CUTE_BF16_PAIRS
    CUTE_UNROLL
    for (int i = 0; i < size(hi); i += 2) {
        float x, y;
        auto address = static_cast<uint32_t>(__cvta_generic_to_shared(&values(i)));
        asm("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(x), "=f"(y) : "r"(address) : "memory");

        auto high = __floats2bfloat162_rn(x, y);
        auto high_fp32 = __bfloat1622float2(high);
        float rx = __fsub_rn(x, high_fp32.x);
        float ry = __fsub_rn(y, high_fp32.y);
        auto middle = __floats2bfloat162_rn(rx, ry);
        auto middle_fp32 = __bfloat1622float2(middle);
        auto low = __floats2bfloat162_rn(__fsub_rn(rx, middle_fp32.x), __fsub_rn(ry, middle_fp32.y));

        using BFloat = cutlass::bfloat16_t;
        hi(i) = BFloat::bitcast(__bfloat16_as_ushort(__low2bfloat16(high)));
        hi(i + 1) = BFloat::bitcast(__bfloat16_as_ushort(__high2bfloat16(high)));
        mid(i) = BFloat::bitcast(__bfloat16_as_ushort(__low2bfloat16(middle)));
        mid(i + 1) = BFloat::bitcast(__bfloat16_as_ushort(__high2bfloat16(middle)));
        lo(i) = BFloat::bitcast(__bfloat16_as_ushort(__low2bfloat16(low)));
        lo(i + 1) = BFloat::bitcast(__bfloat16_as_ushort(__high2bfloat16(low)));
    }
#else
    CUTE_UNROLL
    for (int i = 0; i < size(hi); ++i) {
        float value = values(i);
        hi(i) = cutlass::bfloat16_t(value);
        float residual = __fsub_rn(value, float(hi(i)));
        mid(i) = cutlass::bfloat16_t(residual);
        lo(i) = cutlass::bfloat16_t(__fsub_rn(residual, float(mid(i))));
    }
#endif
}

template <bool UseBias, bool UseGeLU>
__global__ void gemm_emulated(
    const float* X, const float* W, const float* bias, float* Y,
    int M, int K, int N
) {
    __shared__ __align__(16) float shared_a[2][kBlockM * kBlockK];
    __shared__ __align__(16) float shared_b[2][kBlockN * kBlockK];

    using MmaOperation = std::conditional_t<VIT_CUTE_USE_BF16,
        SM80_16x8x16_F32BF16BF16F32_TN, SM80_16x8x8_F32TF32TF32F32_TN>;
    auto mma = make_tiled_mma(MmaOperation{}, Layout<Shape<_2, _2, _1>>{});
    auto thread_mma = mma.get_slice(threadIdx.x);
    auto coordinates = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    auto output_coordinates = thread_mma.partition_C(coordinates);
    auto sum = thread_mma.make_fragment_C(output_coordinates);
    auto main_sum = thread_mma.make_fragment_C(output_coordinates);
    auto correction = thread_mma.make_fragment_C(output_coordinates);
    clear(sum);

    const int first_m = blockIdx.y * kBlockM;
    const int first_n = blockIdx.x * kBlockN;

    if (K > 0) {
        prefetch_tile<kBlockM>(X, shared_a[0], first_m, 0, M, K);
        prefetch_tile<kBlockN>(W, shared_b[0], first_n, 0, N, K);
        __pipeline_commit();
    }

    for (int first_k = 0; first_k < K; first_k += kBlockK) {
        __pipeline_wait_prior(0);
        __syncthreads();

        int stage = (first_k / kBlockK) & 1;
        int next_k = first_k + kBlockK;
        if (next_k < K) {
            prefetch_tile<kBlockM>(X, shared_a[stage ^ 1], first_m, next_k, M, K);
            prefetch_tile<kBlockN>(W, shared_b[stage ^ 1], first_n, next_k, N, K);
            __pipeline_commit();
            if constexpr (!VIT_CUTE_OVERLAP) __pipeline_wait_prior(0);
        }

        auto sA = make_tensor(make_smem_ptr(shared_a[stage]), shared_layout<kBlockM>());
        auto sB = make_tensor(make_smem_ptr(shared_b[stage]), shared_layout<kBlockN>());
        auto a_values = thread_mma.partition_A(sA);
        auto b_values = thread_mma.partition_B(sB);
        auto a_hi = thread_mma.make_fragment_A(a_values);
        auto a_lo = thread_mma.make_fragment_A(a_values);
        auto b_hi = thread_mma.make_fragment_B(b_values);
        auto b_lo = thread_mma.make_fragment_B(b_values);

        clear(main_sum);
        clear(correction);
        if constexpr (VIT_CUTE_USE_BF16) {
            auto a_mid = thread_mma.make_fragment_A(a_values);
            auto b_mid = thread_mma.make_fragment_B(b_values);
            split_bf16(a_values, a_hi, a_mid, a_lo);
            split_bf16(b_values, b_hi, b_mid, b_lo);
            cute::gemm(mma, a_hi, b_lo, correction);
            cute::gemm(mma, a_mid, b_mid, correction);
            cute::gemm(mma, a_lo, b_hi, correction);
            cute::gemm(mma, a_hi, b_mid, correction);
            cute::gemm(mma, a_mid, b_hi, correction);
        } else {
            split_tf32(a_values, a_hi, a_lo);
            split_tf32(b_values, b_hi, b_lo);
            cute::gemm(mma, a_hi, b_lo, correction);
            cute::gemm(mma, a_lo, b_hi, correction);
        }
        cute::gemm(mma, a_hi, b_hi, main_sum);
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
    gemm_emulated<UseBias, UseGeLU><<<grid, kThreads, 0, stream>>>(
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
