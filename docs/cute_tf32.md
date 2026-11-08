# Readable CuTe TF32x3 GEMM for Ampere

`csrc/kernels/gemm_cute_tf32.cu` is an experimental replacement for `gemm.cu`.
It keeps FP32 inputs, bias, accumulators, and output, and preserves the three
existing launcher signatures. The original FP32 kernel remains the default.

The implementation uses two shared-memory buffers and asynchronous prefetch
with a 32 x 32 x 32 block tile and four warps. Aligned inputs use 16-byte
`cp.async` copies directly from global to shared memory. Unaligned rows use
4-byte asynchronous copies; partial tiles are zero-filled. There is no swizzle,
split-K, or autotuning.

## Arithmetic

Each operand is split into two TF32 values:

```text
x_hi = round_tf32(x)
x_lo = round_tf32(x - float(x_hi))

A * B ~= A_hi * B_hi + A_hi * B_lo + A_lo * B_hi
```

CuTe's `SM80_16x8x8_F32TF32TF32F32_TN` atom executes all three products on
Tensor Cores. Main products and residual corrections accumulate separately
within each 32-wide K tile, then contribute to an ordinary FP32 running sum.
This limits error from long Tensor Core accumulation chains. Bias and the
existing erf GELU are applied after the full reduction.

The low-times-low product is omitted. This improves accuracy over a single
TF32 product but does not guarantee IEEE FP32 results or identical accumulation
order. This experiment targets finite, ordinary inference values; subnormal,
overflow, infinity, and NaN behavior has not been made equivalent to FP32 GEMM.

Weights stay in their PyTorch `(N, K)` layout. Input loads are zero-filled at
the M/N/K boundaries, and output and bias accesses are individually predicated.
Zero-sized outputs return without launching; K=0 produces zero or the bias,
followed by GELU if requested.

## Run on Modal

From the repository root, after authenticating with `modal setup`:

```bash
modal run tools/modal_cute_gemm.py
```

The image build compiles the CMake CUDA library on CPU, links a small test
adapter, and compiles the original kernel for comparison. A single A10 runs
`tests/test_cute_gemm.py`, checks the binary for TF32 HMMA instructions, and
benchmarks both implementations and PyTorch FP32 `addmm`. Graph replay removes
Python/ctypes dispatch from the timing loop. Results are saved locally to
`build/cute_tf32_results.json`.

The focused adapter in `tools/gemm_probe.cu` calls the same public launchers
as the PyTorch wrappers; it avoids depending on the older Modal runner's
`ext/` paths or on the package's Python exports.

## Initial single-buffer run

On 2026-11-07, an NVIDIA A10 (SM86), CUDA 12.4, and PyTorch 2.5.1 passed all
33 focused checks. Tests cover all three launcher variants, partial M/N/K
tiles, empty dimensions, a two-layer MLP, and a non-default CUDA stream.
The compiled binary contains `HMMA.1688.F32.TF32` instructions.

Against an FP64 reference, the seeded accuracy probe measured RMSE
`2.73584e-6` for TF32x3 versus `4.75777e-3` for a single rounded TF32 product.
The general GEMM checks use `atol=1e-4, rtol=1e-5`; this is not a guarantee
for every input distribution or reduction size. Full-model accuracy was not
evaluated in this experiment.

Graph-replay timings in milliseconds (bias included, GELU disabled):

| M | K | N | CuTe TF32x3 | Original FP32 | PyTorch FP32 |
|---:|---:|---:|---:|---:|---:|
| 8 | 768 | 3072 | 0.143 | 0.060 | 0.031 |
| 197 | 768 | 2304 | 0.335 | 0.186 | 0.072 |
| 197 | 768 | 3072 | 0.384 | 0.262 | 0.092 |
| 197 | 3072 | 768 | 0.614 | 0.335 | 0.109 |

This implementation is about 1.5–2.4x slower than the original kernel on
these shapes. It is a working, readable Tensor Core example, not a performance
replacement. The original remains the default.

[Modal run and logs](https://modal.com/apps/ahmadsaeedzaidi/main/ap-hdLZf7vY51iN7kILuKiXTj)

## Double buffering and overlap

The single improvement is a two-stage async prefetch pipeline. Tile 0 is
prefetched before the loop. Each iteration waits for its current tile, issues
copies for the next tile into the other buffer, then performs the current
TF32x3 computation. A block barrier protects the consumed buffer before reuse.
The arithmetic, tile sizes, thread layout, and numerical tolerances are unchanged.

Run the two focused tests with:

```bash
modal run tools/modal_cute_pipeline.py
```

1. `test_double_buffer_reuse` runs under Compute Sanitizer Racecheck. It covers
   13 cases and 78 launches: odd/even tile counts, up to 33 K tiles, partial
   M/N/K tiles, unaligned input pointers, empty dimensions, all three launcher
   variants, and a non-default CUDA stream.
2. `test_async_overlap` checks 128-bit `LDGSTS` instructions and PTX prefetch
   ordering, compares results bit-for-bit with two controls, and benchmarks
   them using interleaved CUDA Graph replay timings (median of five samples).

The immediate-wait control is the same two-buffer kernel compiled with
`-DVIT_CUTE_OVERLAP=0`: it waits immediately after prefetch instead of deferring
the wait until the next iteration. The older single-buffer control comes from
commit `1d5431d5bdc35b73d285f6348b81685e515073fd`. This separates the benefit of
deferring the wait from the total change in the copy path. Instruction ordering
and A/B timings establish a functioning overlap path; they do not measure the
number of cycles during which memory transfer and computation overlap.

On 2026-11-08, both tests passed on an A10, with **0 Racecheck hazards**.
All benchmark outputs were bitwise equal to both controls and passed the
existing `atol=1e-4, rtol=1e-5` check against an FP64 reference.

Timings in milliseconds (bias included, GELU disabled):

| M | K | N | Pipeline | Immediate wait | Previous CuTe | Speedup vs previous |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 768 | 3072 | 0.1232 | 0.1320 | 0.1420 | 1.15x |
| 197 | 768 | 2304 | 0.3015 | 0.3227 | 0.3681 | 1.22x |
| 197 | 768 | 3072 | 0.3832 | 0.3886 | 0.4461 | 1.16x |
| 197 | 3072 | 768 | 0.5260 | 0.5416 | 0.6146 | 1.17x |

Deferring the wait contributes approximately 1–7% throughput improvement over
the immediate-wait control in this run. Results are saved to
`build/cute_pipeline_results.json`.

[Pipeline run and logs](https://modal.com/apps/ahmadsaeedzaidi/main/ap-ToUHQiQeOYhEeVXJeF91jO)

See [how we noticed the bank conflicts](bank_conflicts.md) for the next issue
identified by profiling. Swizzling remains to be implemented.

## Enable in a CMake build

```bash
cmake -S . -B build/cute -DVIT_CUTE_TF32=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/cute --target vit_cuda_kernels -j 4
```

With this option, CMake fetches the checksum-pinned CUTLASS v3.5.1 headers,
selects the CuTe source instead of the original GEMM, and targets SM80/SM86.
All operations that use the GEMM launchers, including QKV and MLP, then use
TF32x3. Use `-DVIT_CUTE_TF32=OFF` to select the original kernel again.

References:

- [NVIDIA CuTe GEMM tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html)
- [Pinned CuTe SM80 MMA atoms](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cute/arch/mma_sm80.hpp)
- [CUTLASS TF32 conversion](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cutlass/numeric_conversion.h)
