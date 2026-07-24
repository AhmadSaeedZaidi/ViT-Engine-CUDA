#!/usr/bin/env python3
"""
Modal app for running the vit_cuda test suite with PyTorch profiler integration.

Usage:
    modal run modal_test.py
    modal run modal_test.py -- --profile
    modal run modal_test.py -- --layerwise
    modal run modal_test.py -- --benchmark
    modal run modal_test.py -- --kernels patch_embed,layernorm

Environment:
    Requires Modal CLI authentication: `modal login`
"""
import os
import sys
import subprocess

from modal import App, Image

# ---------------------------------------------------------------------------
# Image configuration
# ---------------------------------------------------------------------------
# Use NVIDIA's PyTorch image which ships with CUDA toolkit (NVCC), PyTorch,
# and cuDNN pre-installed. This is required because ext/setup.py uses
# torch.utils.cpp_extension.CUDAExtension which needs NVCC at build time.
#
# A10 GPU = GA102 (Ampere, sm_86) — matches the hardcoded -gencode in setup.py.
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
)

# ---------------------------------------------------------------------------
# App definition
# ---------------------------------------------------------------------------
app = App("vit-cuda-tests")

# Volume for persisting profiler traces across runs
trace_volume = Image.from_name("vit-traces", create_if_missing=True) if False else None


# ---------------------------------------------------------------------------
# Helper: build the CUDA extension inside the container
# ---------------------------------------------------------------------------
def build_extension():
    """Build and install the vit_cuda extension from ext/."""
    ext_dir = "/repo/ext"
    print(f"Building extension in {ext_dir}...")
    # Build in-place so the .so lands in ext/ alongside the sources
    subprocess.check_call(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=ext_dir,
    )
    # Add ext dir to path so vit_cuda is importable
    if ext_dir not in sys.path:
        sys.path.insert(0, ext_dir)
    import vit_cuda
    print(f"vit_cuda loaded successfully from: {vit_cuda.__file__}")


# ---------------------------------------------------------------------------
# Helper: run the test suite
# ---------------------------------------------------------------------------
def run_test_suite(kernels: str, verbose: bool = True):
    """Run pytest against the tests/ directory."""
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["KERNELS"] = kernels
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    # Ensure vit_cuda is importable in the pytest subprocess
    ext_dir = "/repo/ext"
    os.environ["PYTHONPATH"] = ext_dir + os.pathsep + os.environ.get("PYTHONPATH", "")

    cmd = [sys.executable, "-m", "pytest", "tests/", "-v"] if verbose else [sys.executable, "-m", "pytest", "tests/"]
    rc = subprocess.call(cmd, cwd="/repo")
    if rc != 0:
        raise RuntimeError(f"Tests failed with exit code {rc}")


# ---------------------------------------------------------------------------
# Modal function: build + test without profiler
# ---------------------------------------------------------------------------
@app.function(
    gpu="A10",
    image=image,
    name="build_and_test",
    timeout=600,
)
def build_and_test(kernels: str = "patch_embed,pos_encoding,layernorm,mlp,classifier"):
    """Build the extension and run the test suite on Modal's A10 GPU."""
    build_extension()
    run_test_suite(kernels=kernels, verbose=True)
    return "All tests passed"


# ---------------------------------------------------------------------------
# Modal function: build + test with PyTorch profiler
# ---------------------------------------------------------------------------
@app.function(
    gpu="A10",
    image=image,
    name="profile_tests",
    timeout=600,
)
def profile_tests(kernels: str = "patch_embed,pos_encoding,layernorm,mlp,classifier"):
    """Build the extension, run tests under torch.profiler, export traces."""
    from torch.profiler import profile, ProfilerActivity, record_function

    build_extension()

    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["KERNELS"] = kernels
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    os.environ["PYTHONPATH"] = "/repo/ext" + os.pathsep + os.environ.get("PYTHONPATH", "")

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


# ---------------------------------------------------------------------------
# Modal function: run layerwise comparison with profiler
# ---------------------------------------------------------------------------
@app.function(
    gpu="A10",
    image=image,
    name="profile_layerwise",
    timeout=600,
)
def profile_layerwise():
    """Run the layerwise comparison script under torch.profiler."""
    from torch.profiler import profile, ProfilerActivity, record_function

    build_extension()

    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["KERNELS"] = "patch_embed,pos_encoding,layernorm,mlp,classifier,qkv,gemm_bias"
    os.environ["PYTHONPATH"] = "/repo/ext" + os.pathsep + os.environ.get("PYTHONPATH", "")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with record_function("layerwise_comparison"):
            cmd = [sys.executable, "scripts/layerwise_compare.py", "--image", "tests/sample.jpg"]
            rc = subprocess.call(cmd, cwd="/repo")
            if rc != 0:
                raise RuntimeError(f"Layerwise comparison failed with exit code {rc}")

    print("\n=== Profiler Results (sorted by CUDA time) ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    trace_path = "/tmp/layerwise_trace.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace exported to: {trace_path}")

    return "Layerwise comparison passed and profiled"


# ---------------------------------------------------------------------------
# Modal function: run benchmark with profiler
# ---------------------------------------------------------------------------
@app.function(
    gpu="A10",
    image=image,
    name="profile_benchmark",
    timeout=600,
)
def profile_benchmark():
    """Run the benchmark script under torch.profiler."""
    from torch.profiler import profile, ProfilerActivity, record_function

    build_extension()

    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["PYTHONPATH"] = "/repo/ext" + os.pathsep + os.environ.get("PYTHONPATH", "")

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


# ---------------------------------------------------------------------------
# Modal function: torch.compile profiler & benchmark
# ---------------------------------------------------------------------------
@app.function(
    gpu="A10",
    image=image,
    name="profile_torch_compile",
    timeout=900,
)
def profile_torch_compile():
    """Run the torch.compile analysis script (profiler + benchmark)."""
    build_extension()

    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["PYTHONPATH"] = "/repo/ext" + os.pathsep + os.environ.get("PYTHONPATH", "")

    cmd = [sys.executable, "/repo/torch_compile_test.py"]
    rc = subprocess.call(cmd, cwd="/repo")
    if rc != 0:
        raise RuntimeError(f"Torch compile test failed with exit code {rc}")

    return "Torch compile analysis completed"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    """
    Run the test suite on Modal.

    Use MODAL_MODE env var to select mode:
        (unset)     - build and test
        profile     - with torch.profiler
        layerwise   - layerwise comparison
        benchmark   - benchmark
        torch_compile - torch.compile profiler + benchmark

    Use MODAL_KERNELS env var to select kernel subset:
        patch_embed,pos_encoding,layernorm,mlp,classifier
    """
    mode = os.environ.get("MODAL_MODE", "").strip()
    kernels = os.environ.get("MODAL_KERNELS", "patch_embed,pos_encoding,layernorm,mlp,classifier")

    if mode == "torch_compile":
        result = profile_torch_compile.remote()
    elif mode == "layerwise":
        result = profile_layerwise.remote()
    elif mode == "benchmark":
        result = profile_benchmark.remote()
    elif mode == "profile":
        result = profile_tests.remote(kernels=kernels)
    else:
        result = build_and_test.remote(kernels=kernels)

    print(result)
