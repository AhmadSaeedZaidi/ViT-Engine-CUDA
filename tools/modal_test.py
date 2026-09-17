#!/usr/bin/env python3
"""
Modal app for running the vit_cuda test suite with PyTorch profiler integration.

    Compilation happens in the image layer (CPU time, no GPU allocation).
    Tests and benchmarks run on A10 GPU.

Usage:
    modal run modal_test.py
    MODAL_MODE=profile modal run modal_test.py
    MODAL_MODE=layerwise modal run modal_test.py
    MODAL_MODE=benchmark modal run modal_test.py
    MODAL_MODE=torch_compile modal run modal_test.py
    MODAL_KERNELS=layernorm,mlp modal run modal_test.py

Environment:
    Requires Modal CLI authentication: `modal login`
"""
import os
import sys
import subprocess

from modal import App, Image

IMAGE_TAG = "nvcr.io/nvidia/pytorch:23.10-py3"
REPO_PATH = "/home/ahmadsaeed/code/gpu/ViT-Engine-CUDA"

image = (
    Image.from_registry(IMAGE_TAG)
    .pip_install("timm", "torchvision", "pytest")
    .env({
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "",
    })
    .add_local_dir(REPO_PATH, "/repo", copy=True)
    .run_commands(
        f"cd /repo/ext && {sys.executable} setup.py build_ext --inplace",
        "rm -rf /repo/ext/build",
    )
)

app = App("vit-cuda-tests")


def _ensure_env():
    """Set PYTHONPATH so the prebuilt .so is importable."""
    ext_dir = "/repo/ext"
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get(
        "PYTORCH_CUDA_ALLOC_CONF", ""
    )
    env_pythonpath = os.environ.get("PYTHONPATH", "")
    if ext_dir not in env_pythonpath.split(os.pathsep):
        os.environ["PYTHONPATH"] = ext_dir + os.pathsep + env_pythonpath


def _run_tests(kernels: str, *, verbose: bool = True) -> None:
    os.environ["KERNELS"] = kernels
    cmd = (
        [sys.executable, "-m", "pytest", "tests/", "-v"]
        if verbose
        else [sys.executable, "-m", "pytest", "tests/"]
    )
    rc = subprocess.call(cmd, cwd="/repo")
    if rc != 0:
        raise RuntimeError(f"Tests failed with exit code {rc}")


# ---------------------------------------------------------------------------
# GPU functions — compilation baked into image, no build_extension() call
# ---------------------------------------------------------------------------
@app.function(gpu="A10", image=image, name="build_and_test", timeout=600)
def build_and_test(
    kernels: str = "patch_embed,pos_encoding,layernorm,mlp,classifier",
):
    _ensure_env()
    _run_tests(kernels=kernels, verbose=True)
    return "All tests passed"


@app.function(gpu="A10", image=image, name="profile_tests", timeout=600)
def profile_tests(
    kernels: str = "patch_embed,pos_encoding,layernorm,mlp,classifier",
):
    from torch.profiler import profile, ProfilerActivity, record_function

    _ensure_env()
    os.environ["KERNELS"] = kernels

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with record_function("vit_cuda_test_suite"):
            cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
            rc = subprocess.call(cmd, cwd="/repo")
            if rc != 0:
                raise RuntimeError(f"Tests failed with exit code {rc}")

    print("\n=== Profiler Results (sorted by CUDA time) ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    trace_path = "/tmp/trace.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace exported to: {trace_path}")

    print("\n=== Profiler Results (sorted by CPU time) ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    return "Tests passed and profiled"


@app.function(gpu="A10", image=image, name="profile_layerwise", timeout=600)
def profile_layerwise():
    from torch.profiler import profile, ProfilerActivity, record_function

    _ensure_env()
    os.environ["KERNELS"] = (
        "patch_embed,pos_encoding,layernorm,mlp,classifier,qkv,gemm_bias"
    )

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with record_function("layerwise_compare"):
            cmd = [
                sys.executable,
                "scripts/layerwise_compare.py",
                "--image",
                "tests/sample.jpg",
            ]
            rc = subprocess.call(cmd, cwd="/repo")
            if rc != 0:
                raise RuntimeError(
                    f"Layerwise comparison failed with exit code {rc}"
                )

    print("\n=== Profiler Results (sorted by CUDA time) ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    trace_path = "/tmp/layerwise_trace.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace exported to: {trace_path}")
    return "Layerwise comparison passed and profiled"


@app.function(gpu="A10", image=image, name="profile_benchmark", timeout=600)
def profile_benchmark():
    from torch.profiler import profile, ProfilerActivity, record_function

    _ensure_env()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with record_function("benchmark"):
            cmd = [sys.executable, "benchmark.py"]
            rc = subprocess.call(cmd, cwd="/repo")
            if rc != 0:
                raise RuntimeError(f"Benchmark failed with exit code {rc}")

    print("\n=== Profiler Results (sorted by CUDA time) ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    trace_path = "/tmp/benchmark_trace.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace exported to: {trace_path}")
    return "Benchmark completed and profiled"


@app.function(
    gpu="A10", image=image, name="profile_torch_compile", timeout=900
)
def profile_torch_compile():
    _ensure_env()

    cmd = [sys.executable, "/repo/torch_compile_test.py"]
    rc = subprocess.call(cmd, cwd="/repo")
    if rc != 0:
        raise RuntimeError(f"Torch compile test failed with exit code {rc}")

    return "Torch compile analysis completed"


@app.function(gpu="A10", image=image, name="benchmark_cuda_graph", timeout=600)
def benchmark_cuda_graph(batch: int = 32, iterations: int = 50, warmup: int = 10):
    _ensure_env()

    cmd = [
        sys.executable,
        "benchmark.py",
        "--batch", str(batch),
        "--iterations", str(iterations),
        "--warmup", str(warmup),
        "--graph",
        "--trace",
    ]
    rc = subprocess.call(cmd, cwd="/repo")
    if rc != 0:
        raise RuntimeError(f"CUDA graph benchmark failed with exit code {rc}")
    return "CUDA graph benchmark completed (eager vs graph, traces in /repo/results)"


# ---------------------------------------------------------------------------
# Local entry point — runs on your machine, routes to Modal
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    mode = os.environ.get("MODAL_MODE", "test").strip()
    kernels = os.environ.get(
        "MODAL_KERNELS",
        "patch_embed,pos_encoding,layernorm,mlp,classifier,qkv,gemm_bias,flash_attn",
    )

    dispatcher = {
        "test": lambda: build_and_test.remote(kernels=kernels),
        "profile": lambda: profile_tests.remote(kernels=kernels),
        "layerwise": lambda: profile_layerwise.remote(),
        "benchmark": lambda: profile_benchmark.remote(),
        "benchmark_graph": lambda: benchmark_cuda_graph.remote(),
        "torch_compile": lambda: profile_torch_compile.remote(),
    }

    result = dispatcher.get(mode, lambda: f"Unknown mode: {mode}")()
    print(result)