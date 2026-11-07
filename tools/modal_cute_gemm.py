"""Build and check the readable CuTe TF32x3 GEMM on one Modal A10.

    modal run tools/modal_cute_gemm.py

Compiles on CPU in the image build. The GPU function runs correctness checks,
verifies Tensor Core instructions, and compares against the original GEMM.
"""
import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
BUILD = "/opt/vit-cute-build"
PROBE = "-std=c++17 -O3 -arch=sm_86 -shared -Xcompiler=-fPIC -I/repo/csrc/include"

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .pip_install("cmake==3.30.5", "ninja", "pytest")
    .env({"TORCH_CUDA_ARCH_LIST": "8.6"})
    .add_local_file(REPO / "CMakeLists.txt", "/repo/CMakeLists.txt", copy=True)
    .add_local_dir(REPO / "csrc", "/repo/csrc", copy=True)
    .add_local_file(REPO / "tools/gemm_probe.cu", "/repo/tools/gemm_probe.cu", copy=True)
    .run_commands(
        f"cmake -S /repo -B {BUILD} -GNinja -DVIT_CUTE_TF32=ON -DCMAKE_BUILD_TYPE=Release",
        f"cmake --build {BUILD} --target vit_cuda_kernels -j 4",
        f"nvcc {PROBE} /repo/tools/gemm_probe.cu {BUILD}/libvit_cuda_kernels.a -o /opt/libgemm_cute.so",
        f"nvcc {PROBE} /repo/tools/gemm_probe.cu /repo/csrc/kernels/gemm.cu -o /opt/libgemm_simt.so",
    )
    .add_local_file(REPO / "tests/test_cute_gemm.py", "/repo/tests/test_cute_gemm.py")
)
app = modal.App("vit-cute-tf32-experiment")


@app.function(image=image, gpu="A10", cpu=4, memory=16384, timeout=600)
def check():
    import os
    import subprocess
    import sys

    import torch

    os.environ["VIT_GEMM_LIBRARY"] = "/opt/libgemm_cute.so"
    torch.backends.cuda.matmul.allow_tf32 = False
    subprocess.run([
        sys.executable, "-m", "pytest", "/repo/tests/test_cute_gemm.py", "-q", "-s",
    ], check=True, cwd="/repo")

    # Check the actual binary, rather than inferring Tensor Core use from timings.
    sass = subprocess.check_output(["cuobjdump", "--dump-sass", "/opt/libgemm_cute.so"], text=True)
    tensor_core_lines = [line.strip() for line in sass.splitlines() if "HMMA" in line and "TF32" in line]
    assert tensor_core_lines, "No TF32 Tensor Core instructions found"
    print("Tensor Core instruction:", tensor_core_lines[0])

    sys.path.insert(0, "/repo/tests")
    from test_cute_gemm import GemmLibrary

    cute = GemmLibrary("/opt/libgemm_cute.so")
    simt = GemmLibrary("/opt/libgemm_simt.so")

    def time_ms(operation):
        for _ in range(3):
            operation()
        torch.cuda.synchronize()
        # Replay batches to keep Python/ctypes dispatch out of kernel timings.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(10):
                operation()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / 100

    results = []
    for m, k, n in [(8, 768, 3072), (197, 768, 2304), (197, 768, 3072), (197, 3072, 768)]:
        torch.manual_seed(0)
        x = torch.randn(m, k, device="cuda")
        w = torch.randn(n, k, device="cuda")
        bias = torch.randn(n, device="cuda")
        out = torch.empty(m, n, device="cuda")
        row = {"M": m, "K": k, "N": n}
        row["cute_tf32x3_ms"] = time_ms(lambda: cute.launch(x, w, bias, out))
        row["original_fp32_ms"] = time_ms(lambda: simt.launch(x, w, bias, out))
        row["torch_fp32_ms"] = time_ms(lambda: torch.addmm(bias, x, w.t(), out=out))
        row["speedup_vs_original"] = row["original_fp32_ms"] / row["cute_tf32x3_ms"]
        results.append(row)
        print(json.dumps(row))

    return {
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "torch": str(torch.__version__),
        "tensor_core_instruction": tensor_core_lines[0],
        "benchmarks": results,
    }


@app.local_entrypoint()
def main():
    result = check.remote()
    path = REPO / "build" / "cute_tf32_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Results saved to {path}")
