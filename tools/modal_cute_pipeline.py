"""Run two tests on an A10: buffer reuse under Racecheck, then overlap A/B timing.

    modal run tools/modal_cute_pipeline.py
"""
from pathlib import Path
import subprocess

import modal

REPO = Path(__file__).resolve().parents[1]
BUILD = "/opt/vit-pipeline-build"
BASELINE_COMMIT = "1d5431d5bdc35b73d285f6348b81685e515073fd"
BASELINE = REPO / "build" / "gemm_cute_before_pipeline.cu"
if modal.is_local():
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_bytes(subprocess.check_output([
        "git", "show", f"{BASELINE_COMMIT}:csrc/kernels/gemm_cute_tf32.cu",
    ], cwd=REPO))

FLAGS = (
    "-std=c++17 -O3 --use_fast_math --expt-relaxed-constexpr -arch=sm_86 "
    "-shared -Xcompiler=-fPIC -I/repo/csrc/include "
    f"-I{BUILD}/_deps/cutlass_headers-src/include"
)
image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .pip_install("cmake==3.30.5", "ninja", "pytest")
    .env({"TORCH_CUDA_ARCH_LIST": "8.6"})
    .add_local_file(REPO / "CMakeLists.txt", "/repo/CMakeLists.txt", copy=True)
    .add_local_dir(REPO / "csrc", "/repo/csrc", copy=True)
    .add_local_file(REPO / "tools/gemm_probe.cu", "/repo/tools/gemm_probe.cu", copy=True)
    .add_local_file(BASELINE, "/opt/gemm_cute_before_pipeline.cu", copy=True)
    .run_commands(
        f"cmake -S /repo -B {BUILD} -GNinja -DVIT_CUTE_TF32=ON -DCMAKE_BUILD_TYPE=Release",
        f"cmake --build {BUILD} --target vit_cuda_kernels -j 4",
        f"nvcc {FLAGS} /repo/tools/gemm_probe.cu {BUILD}/libvit_cuda_kernels.a -o /opt/libgemm_pipeline.so",
        f"nvcc {FLAGS} -DVIT_CUTE_OVERLAP=0 /repo/tools/gemm_probe.cu /repo/csrc/kernels/gemm_cute_tf32.cu -o /opt/libgemm_immediate_wait.so",
        f"nvcc {FLAGS} /repo/tools/gemm_probe.cu /opt/gemm_cute_before_pipeline.cu -o /opt/libgemm_before.so",
    )
    .add_local_file(REPO / "tests/test_cute_gemm.py", "/repo/tests/test_cute_gemm.py")
    .add_local_file(REPO / "tests/test_cute_pipeline.py", "/repo/tests/test_cute_pipeline.py")
)
app = modal.App("vit-cute-pipeline")


@app.function(image=image, gpu="A10", cpu=4, memory=16384, timeout=900)
def check():
    import json
    import os
    import sys

    os.environ.update({
        "VIT_GEMM_LIBRARY": "/opt/libgemm_pipeline.so",
        "VIT_GEMM_SERIAL_LIBRARY": "/opt/libgemm_immediate_wait.so",
        "VIT_GEMM_BEFORE_LIBRARY": "/opt/libgemm_before.so",
    })
    tests = "/repo/tests/test_cute_pipeline.py"
    subprocess.run([
        "compute-sanitizer", "--tool", "racecheck", "--error-exitcode", "1",
        "--kernel-name", "kns=gemm_tf32x3",
        sys.executable, "-m", "pytest", f"{tests}::test_double_buffer_reuse", "-q", "-s",
    ], check=True, cwd="/repo")
    subprocess.run([
        sys.executable, "-m", "pytest", f"{tests}::test_async_overlap", "-q", "-s",
    ], check=True, cwd="/repo")
    result = json.loads(Path("/tmp/cute_pipeline_results.json").read_text())
    result["double_buffer_racecheck"] = "passed"
    result["baseline_commit"] = BASELINE_COMMIT
    return result


@app.local_entrypoint()
def main():
    import json

    result = check.remote()
    path = REPO / "build" / "cute_pipeline_results.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Results saved to {path}")
