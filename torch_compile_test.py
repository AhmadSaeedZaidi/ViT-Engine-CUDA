#!/usr/bin/env python3
"""
Run ViTCUDA inference under torch.compile + torch.profiler to see:
  - Does torch.compile fuse layernorm+qkv_proj+flash_attn_2+gemm_bias?
  - Does Inductor replace custom CUDA ops?
  - What is the overall latency difference (compile vs no compile)?
  - Benchmark timm vs vit_cuda (both with and without compile)
"""
import os
import sys
import time
import json
import urllib.request
import math
import argparse

import torch
import torch.nn as nn

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from torchvision import transforms
from PIL import Image

import vit_cuda

# ----------------------------------------------------------------
# import the inference module so we can instantiate ViTCUDA and ViTBlockCUDA
# ----------------------------------------------------------------
sys.path.insert(0, "/repo")
from inference import ViTCUDA, ViTBlockCUDA
from benchmark import measure_latency

# ----------------------------------------------------------------
# Profiler runner
# ----------------------------------------------------------------
def run_profiler(model, inp, label, iterations=5, warmup=3):
    """Run forward passes under torch.profiler and print key averages."""
    from torch.profiler import profile, ProfilerActivity, record_function

    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(inp)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,  # less noise, faster
    ) as prof:
        with record_function(f"forward_{label}"):
            for i in range(iterations):
                inp_i = inp if i == 0 else torch.randn_like(inp)
                with torch.no_grad():
                    _ = model(inp_i)

    print(f"\n{'='*80}")
    print(f"PROFILER: {label}")
    print(f"{'='*80}")
    print("Sorted by CUDA time total:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=35))
    print("\nSorted by CPU time total:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    print(f"\nSelf CPU time total (top operators):")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))

    # Export trace
    trace_path = f"/tmp/trace_{label}.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace exported to: {trace_path}")

    # Count how many instances of custom ops survive after compile
    op_counts = {}
    all_keys = prof.key_averages().table(sort_by="count", row_limit=100)
    # We'll grep the key_averages programmatically instead
    for evt in prof.key_averages():
        name = evt.key
        if "cudaLaunchKernel" in name or "vit_cuda" in name:
            if name not in op_counts:
                op_counts[name] = evt.count

    print("\nRelevant CUDA kernel counts:")
    for k, v in sorted(op_counts.items()):
        print(f"  {k}: {v}")

    return prof


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    device = "cuda"
    batch = 1

    print("Building models...", flush=True)

    # --- timm reference ---
    import timm
    ref_model = timm.create_model("vit_base_patch16_224", pretrained=True).to(device).eval()

    # --- ViTCUDA (uncompiled) ---
    vit_orig = ViTCUDA().to(device).eval()

    # --- ViTCUDA with torch.compile ---
    # mode="reduce-overhead" is recommended when we have frequent small kernel
    # launches — it uses CUDA graphs to amortise launch overhead
    print("Applying torch.compile(mode='reduce-overhead') to ViTCUDA...", flush=True)
    vit_compiled = torch.compile(
        ViTCUDA().to(device).eval(),
        mode="reduce-overhead",
        fullgraph=False,  # allow graph breaks if Inductor doesn't support an op
    )

    # Create inputs
    # For the profiler we use ImageNet-sized random tensor (faster, no IO)
    inp_rand = torch.randn(batch, 3, 224, 224, device=device, dtype=torch.float32)

    # Warmup both models
    print("Warming up compiled model (first run triggers compilation)...", flush=True)
    with torch.no_grad():
        _ = vit_compiled(inp_rand)
    print("Compilation warmup done.", flush=True)
    torch.cuda.synchronize()

    # ----------------------------------------------------------------
    # Phase 1: Profiler on uncompiled ViTCUDA
    # ----------------------------------------------------------------
    print("\n=== PHASE 1: ViTCUDA (uncompiled) ===", flush=True)
    run_profiler(vit_orig, inp_rand, "vit_cuda_no_compile", iterations=5)

    torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # Phase 2: Profiler on compiled ViTCUDA
    # ----------------------------------------------------------------
    print("\n=== PHASE 2: ViTCUDA (torch.compile, reduce-overhead) ===", flush=True)
    run_profiler(vit_compiled, inp_rand, "vit_cuda_compile", iterations=5)

    torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # Phase 3: Profiler on compiled timm model
    # ----------------------------------------------------------------
    print("\n=== PHASE 3: timm ViT (torch.compile, reduce-overhead) ===", flush=True)
    ref_compiled = torch.compile(
        ref_model,
        mode="reduce-overhead",
        fullgraph=False,
    )
    with torch.no_grad():
        _ = ref_compiled(inp_rand)
    torch.cuda.synchronize()

    run_profiler(ref_compiled, inp_rand, "timm_compile", iterations=5)

    torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # Phase 4: Benchmark latency (all 4 variants)
    # ----------------------------------------------------------------
    print("\n=== PHASE 4: Latency Benchmarks ===", flush=True)
    bench_batch = 32
    inp_bench = torch.randn(bench_batch, 3, 224, 224, device=device, dtype=torch.float32)

    # Warmup compiled models with the larger batch
    with torch.no_grad():
        _ = vit_compiled(inp_bench)
        _ = ref_compiled(inp_bench)
    torch.cuda.synchronize()

    results = {}

    for name, model in [
        ("timm (no compile)", ref_model),
        ("timm (compile)", ref_compiled),
        ("vit_cuda (no compile)", vit_orig),
        ("vit_cuda (compile)", vit_compiled),
    ]:
        print(f"  Benchmarking {name}...", flush=True)
        best, avg = measure_latency(model, inp_bench, iterations=50, warmup=10)
        results[name] = (best, avg)

    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS (batch=32, 50 iterations)")
    print(f"{'='*80}")
    print(f"{'Model':<30} {'Best (ms)':>10} {'Avg (ms)':>10}")
    print("-" * 50)
    for name, (best, avg) in results.items():
        print(f"{name:<30} {best:>10.3f} {avg:>10.3f}")

    # Compute speedups
    timm_nc = results["timm (no compile)"][0]
    vit_nc = results["vit_cuda (no compile)"][0]
    timm_c = results["timm (compile)"][0]
    vit_c = results["vit_cuda (compile)"][0]

    print(f"\nSpeedups:")
    print(f"  vit_cuda vs timm (no compile): {timm_nc/vit_nc:.3f}x")
    print(f"  vit_cuda vs timm (compile):    {timm_c/vit_c:.3f}x")
    if vit_c > 0:
        print(f"  vit_cuda compile vs no compile: {vit_nc/vit_c:.3f}x")
    if timm_c > 0:
        print(f"  timm compile vs no compile:     {timm_nc/timm_c:.3f}x")

    print("\nDone.")


if __name__ == "__main__":
    main()