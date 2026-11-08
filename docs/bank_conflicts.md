# How we noticed the bank conflicts

1. **Double buffering helped, but only a little.** It made our CuTe kernel
   1.15–1.22x faster. We needed to check what was still slowing it down.

2. **We looked at the shared-memory layout.** Both operand tiles have 32
   floats per row. `LayoutRight` puts element `(row, column)` at float offset
   `row * 32 + column`.

3. **We checked which bank each address uses.** Ampere has 32 shared-memory
   banks. For these float accesses, `bank = float_offset % 32`:

   | Element | Float offset | Bank |
   |---|---:|---:|
   | `(0, 0)` | 0 | 0 |
   | `(1, 0)` | 32 | 0 |
   | `(2, 0)` | 64 | 0 |

4. **That made the MMA reads suspicious.** CuTe assigns operand values from
   different rows to threads in a warp. If those threads read different
   addresses in the same bank together, the reads must be served separately.
   The layout was a clue; the row stride alone was not proof. Reading the
   exact same address together is a broadcast, not a bank conflict.

5. **We confirmed it with Nsight Compute.** We profiled one bias GEMM on A10
   with `M=197`, `K=768`, `N=3072`. The counter
   `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`
   reported **14,458,574 shared-load bank conflicts** for that launch.
   DRAM throughput was about 14% and Tensor-pipe activity about 16%.
   Profiling used `--clock-control none`; the GPU clocks were not locked.

6. **Passing correctness tests did not rule this out.** Racecheck reported
   zero hazards. That checks synchronization safety; bank conflicts slow
   down otherwise-correct memory accesses.

7. **The next step is swizzling.** Change how values are placed in shared
   memory so simultaneous MMA reads spread across banks. Copy destinations
   and MMA reads must use the same mapping, while preserving 16-byte copy
   alignment. Then rerun correctness checks and compare the conflict counter
   and timings. This commit does not implement the swizzle yet.

Reference: [NVIDIA's explanation of shared-memory banks](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#shared-memory-and-memory-banks).
