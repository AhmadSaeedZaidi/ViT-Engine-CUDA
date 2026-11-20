# CuTe TF32x3 and BF16x6 GEMM for Ampere

`csrc/kernels/gemm_cute_tf32.cu` is an experimental replacement for `gemm.cu`.
It keeps FP32 inputs, bias, accumulators, and output, and preserves the three
existing launcher signatures. The original FP32 kernel remains the default.

The implementation uses two shared-memory buffers and asynchronous prefetch
with a 32 x 32 x 32 block tile and four warps. Aligned inputs use 16-byte
`cp.async` copies directly from global to shared memory. Unaligned rows use
4-byte asynchronous copies, and partial tiles are zero-filled.
TF32x3 uses `Swizzle<3,2,3>` to XOR three row bits into the four-float
group index in shared memory. BF16x6 uses `Swizzle<2,3,2>` so four rows land
in separate eight-bank groups. The copy and MMA reads use the same layout
for each variant. Each copy remains contiguous and 16-byte aligned.

## Arithmetic

TF32x3 splits each operand into two TF32 values:

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

BF16x6 splits each value into a high BF16 component, a rounded BF16 residual
of the first subtraction, and a third BF16 component from the remaining
residual. It keeps the six products with the largest significance:

```text
A_hi*B_hi
+ A_hi*B_mid + A_mid*B_hi
+ A_hi*B_lo + A_mid*B_mid + A_lo*B_hi
```

The five correction products accumulate separately from the high product.
BF16x6 uses the SM80 m16n8k16 BF16 atom with FP32 accumulation. See
[Henry et al. for the BF16 decomposition](https://arxiv.org/abs/1904.06376).

Each BF16 MMA lane takes pairs of adjacent FP32 values from shared memory.
The BF16 layout spreads a half warp's 16 pairs across the 32 banks, and a
64-bit shared load issues each pair together. The three roundings use the
packed BF16 conversion. The comparison runner builds the previous scalar
BF16 path with `-DVIT_CUTE_BF16_PAIRS=0` for a bitwise and timing comparison.

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

See [how we noticed the bank conflicts](bank_conflicts.md) for the original
profile. The swizzled builds can be compared on the same A10 with:

```bash
modal run tools/modal_cute_pipeline.py --compare
```

This runs the GEMM correctness suite and Racecheck on both variants. It
compares bias GEMM with PyTorch FP32 `addmm` (TF32 disabled) and uses PyTorch
FP64 as an accuracy reference. It also compares with the unswizzled TF32x3
pipeline and measures shared-load bank conflicts with Nsight Compute.
Results are saved to `build/cute_precision_results.json`.

## First swizzle comparison on A10

Median latency in milliseconds for seeded random FP32 inputs with bias.
Numbers are from five interleaved CUDA Graph timing samples.

| M × K × N | TF32x3 unswizzled | TF32x3 swizzled | BF16x6 swizzled | PyTorch FP32 |
|---|---:|---:|---:|---:|
| 8 × 768 × 3072 | 0.123 | 0.118 | 0.047 | 0.030 |
| 197 × 768 × 2304 | 0.340 | 0.313 | 0.129 | 0.078 |
| 197 × 768 × 3072 | 0.429 | 0.396 | 0.168 | 0.097 |
| 197 × 3072 × 768 | 0.593 | 0.526 | 0.215 | 0.113 |

Both variants passed 33 correctness tests and the buffer reuse test under
Racecheck, which reported zero hazards for both. TF32x3 results were bitwise
identical before and after swizzling. For the `197 × 768 × 3072` shape,
the Nsight shared-load bank-conflict counter dropped from 14,458,104 to 8,429
for TF32x3. BF16x6 recorded 2,075,184 conflicts with the same swizzle.

Across all shapes, BF16x6 was 2.37–2.53× as fast as swizzled TF32x3 and
took 1.54–1.91× as long as PyTorch FP32. Its maximum absolute difference
from PyTorch FP32 was approximately `2.14e-4`. Its RMSE against FP64 was
lower than both TF32x3 and PyTorch FP32 on each shape in this run. These
measurements cover GEMM accuracy on random inputs.
[Modal run](https://modal.com/apps/ahmadsaeedzaidi/main/ap-fJOGEVT8sWDUQfZ1xAJKN1)

## BF16 paired shared loads on A10

The BF16 swizzle and paired loads reduced the shared-load bank-conflict
counter from 2,075,340 to 8,789 at `197 × 768 × 3072`. Outputs matched the
scalar BF16 build bit-for-bit on all four timed shapes. Both CuTe variants
passed 33 correctness tests each and the buffer reuse test with zero
Racecheck hazards.

Median bias GEMM latency in microseconds from the same run:

| M × K × N | BF16x6 scalar | BF16x6 paired | PyTorch FP32 |
|---|---:|---:|---:|
| 8 × 768 × 3072 | 47.1 | 40.6 | 30.5 |
| 197 × 768 × 2304 | 127.7 | 114.5 | 78.1 |
| 197 × 768 × 3072 | 167.8 | 157.3 | 103.7 |
| 197 × 3072 × 768 | 212.6 | 196.1 | 115.3 |

The paired path reduced latency by 6–14% and took 1.33–1.70× as long as
PyTorch FP32 with TF32 disabled.
[Modal run](https://modal.com/apps/ahmadsaeedzaidi/main/ap-VhQ7yQycWwOsm1odSkLaxn)

A follow-up tried a two-warp BF16 tile for `M <= 8`. It swapped the
operands to fit the MMA's eight-column dimension and stored directly in the
original output layout. For `8 × 768 × 3072`, it took 40.4 µs, compared
with 40.3 µs for the general paired kernel and 30.7 µs for PyTorch FP32.
It passed the numerical tests and Racecheck. The general paired kernel
remains the BF16 implementation.
[Small M experiment](https://modal.com/apps/ahmadsaeedzaidi/main/ap-krF2DwW9RRe7lcRahkexD8)

## Enable in a CMake build

```bash
cmake -S . -B build/cute -DVIT_CUTE_TF32=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/cute --target vit_cuda_kernels -j 4
cmake -S . -B build/cute-bf16 -DVIT_CUTE_BF16=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/cute-bf16 --target vit_cuda_kernels -j 4
```

With this option, CMake fetches the checksum-pinned CUTLASS v3.5.1 headers,
selects the CuTe source instead of the original GEMM, and targets SM80/SM86.
All operations that use the GEMM launchers, including QKV and MLP, use the
selected CuTe variant. Both options default to `OFF`, which selects the
original FP32 kernel. Enable one variant per build.

References:

- [NVIDIA CuTe GEMM tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html)
- [Pinned CuTe SM80 MMA atoms](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cute/arch/mma_sm80.hpp)
- [CUTLASS TF32 conversion](https://github.com/NVIDIA/cutlass/blob/v3.5.1/include/cutlass/numeric_conversion.h)
