"""Compare the two CuTe builds and PyTorch FP32 on the same CUDA inputs."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import re
from statistics import median
import subprocess
import sys

import torch

LIBRARIES = {
    "tf32x3_unswizzled": "/opt/libgemm_unswizzled.so",
    "tf32x3": "/opt/libgemm_pipeline.so",
    "bf16x6_scalar": "/opt/libgemm_bf16_scalar.so",
    "bf16x6": "/opt/libgemm_bf16.so",
}
SHAPES = [(8, 768, 3072), (197, 768, 2304), (197, 768, 3072), (197, 3072, 768)]
PROFILE_SHAPE = (197, 768, 3072)
CONFLICTS = "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"


def inputs(shape, seed):
    torch.manual_seed(seed)
    m, k, n = shape
    return tuple(torch.randn(size, device="cuda") for size in [(m, k), (n, k), (n,)])


def capture(operation):
    for _ in range(3):
        operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            operation()
    return graph


def time_graph(graph):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(10):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 100


def error_stats(actual, reference):
    error = (actual.double() - reference).abs()
    return {"max_abs": error.max().item(), "rmse": error.square().mean().sqrt().item()}


def compare_shape(shape, kernels):
    x, w, bias = inputs(shape, seed=0)
    pytorch = torch.addmm(bias, x, w.t())
    fp64 = torch.addmm(bias.double(), x.double(), w.double().t())
    outputs = {name: kernel(x, w, bias) for name, kernel in kernels.items()}
    torch.testing.assert_close(outputs["tf32x3"], outputs["tf32x3_unswizzled"], atol=0, rtol=0)
    torch.testing.assert_close(outputs["bf16x6"], outputs["bf16x6_scalar"], atol=0, rtol=0)
    for actual in outputs.values():
        torch.testing.assert_close(actual, fp64.float(), atol=1e-4, rtol=1e-5)

    out = torch.empty_like(pytorch)
    graphs = {
        name: capture(lambda: kernel.launch(x, w, bias, out))
        for name, kernel in kernels.items()
    }
    graphs["pytorch_fp32"] = capture(lambda: torch.addmm(bias, x, w.t(), out=out))
    samples = {name: [] for name in graphs}
    names = list(graphs)
    for trial in range(5):
        offset = trial % len(names)
        for name in names[offset:] + names[:offset]:
            samples[name].append(time_graph(graphs[name]))

    return {
        "shape_MKN": shape,
        "milliseconds": {name: median(values) for name, values in samples.items()},
        "error_vs_pytorch_fp32": {name: error_stats(outputs[name], pytorch)
                                  for name in ("tf32x3", "bf16x6")},
        "error_vs_fp64": {name: error_stats(out, fp64)
                          for name, out in {**outputs, "pytorch_fp32": pytorch}.items()},
    }


def profile(name):
    process = subprocess.run([
        "ncu", "--clock-control", "none", "--kernel-name", "regex:gemm_emulated",
        "--launch-count", "1", "--metrics", CONFLICTS, "--page", "raw", "--csv",
        sys.executable, __file__, "--profile-kernel", name,
    ], text=True, capture_output=True)
    for row in csv.DictReader(io.StringIO(process.stdout[process.stdout.find('"ID",') :])):
        if row.get("ID") == "0" and row.get(CONFLICTS):
            return {"shared_load_bank_conflicts": int(row[CONFLICTS].replace(",", ""))}
    return {"error": (process.stderr + process.stdout).strip()[-2000:]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-kernel", choices=LIBRARIES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    os.environ["VIT_GEMM_LIBRARY"] = LIBRARIES["tf32x3"]
    from test_cute_gemm import GemmLibrary

    if args.profile_kernel:
        GemmLibrary(LIBRARIES[args.profile_kernel])(*inputs(PROFILE_SHAPE, seed=0))
        torch.cuda.synchronize()
        return
    if args.output is None:
        parser.error("--output is required")

    kernels = {name: GemmLibrary(path) for name, path in LIBRARIES.items()}
    for name, suffix in [("tf32x3", "TF32"), ("bf16x6", "BF16")]:
        sass = subprocess.check_output([
            "cuobjdump", "--gpu-architecture", "sm_86", "--dump-sass", LIBRARIES[name],
        ], text=True)
        assert re.search(r"HMMA[^\n;]*" + suffix, sass), f"{name} Tensor Core instruction missing"
        assert re.search(r"LDGSTS[^\n;]*\.128", sass), f"{name} 128-bit async copy missing"
        if name == "bf16x6":
            assert re.search(r"\bLDS\.64\b", sass), "BF16x6 paired shared load missing"

    rows = []
    for shape in SHAPES:
        row = compare_shape(shape, kernels)
        rows.append(row)
        print(json.dumps(row), flush=True)
    profiles = {name: profile(name) for name in kernels}
    result = {
        "gpu": torch.cuda.get_device_name(),
        "pytorch": str(torch.__version__),
        "profile_shape_MKN": PROFILE_SHAPE,
        "profiles": profiles,
        "benchmarks": rows,
    }
    print(json.dumps(profiles), flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
