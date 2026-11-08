"""Two focused GPU tests for asynchronous prefetch and shared-buffer reuse."""
import json
import os
from pathlib import Path
import re
import statistics
import subprocess

import torch
import torch.nn.functional as F

from test_cute_gemm import GemmLibrary


def test_double_buffer_reuse():
    gemm = GemmLibrary(os.environ["VIT_GEMM_LIBRARY"])
    stream = torch.cuda.Stream()
    cases = [(35, k, 37) for k in (0, 1, 31, 32, 33, 64, 65, 96, 97, 257, 1025)]
    cases += [(0, 65, 37), (35, 65, 0)]
    launches = 0
    with torch.cuda.stream(stream):
        for case, (m, k, n) in enumerate(cases):
            offset = case % 2
            x = torch.randn(m * k + offset, device="cuda")[offset:].view(m, k)
            w = torch.randn(n * k + offset, device="cuda")[offset:].view(n, k)
            bias = torch.randn(n, device="cuda")
            for _ in range(2):
                x.normal_()
                w.normal_()
                product = x.double() @ w.double().t()
                for use_bias, use_gelu in [(False, False), (True, False), (True, True)]:
                    actual = gemm(x, w, bias, use_bias=use_bias, use_gelu=use_gelu)
                    expected = product + bias.double() if use_bias else product
                    if use_gelu:
                        expected = F.gelu(expected, approximate="none")
                    torch.testing.assert_close(actual, expected.float(), atol=1e-4, rtol=1e-5)
                    launches += 1
    stream.synchronize()
    print(f"Double-buffer reuse: {len(cases)} cases, {launches} launches on a non-default stream")


def test_async_overlap():
    pipeline = GemmLibrary(os.environ["VIT_GEMM_LIBRARY"])
    immediate_wait = GemmLibrary(os.environ["VIT_GEMM_SERIAL_LIBRARY"])
    before = GemmLibrary(os.environ["VIT_GEMM_BEFORE_LIBRARY"])

    sass = subprocess.check_output([
        "cuobjdump", "--gpu-architecture", "sm_86", "--dump-sass", os.environ["VIT_GEMM_LIBRARY"],
    ], text=True)
    assert re.search(r"LDGSTS[^;\n]*\.128", sass), "16-byte asynchronous copies missing"
    assert "HMMA.1688.F32.TF32" in sass
    ptx = subprocess.check_output(["cuobjdump", "--dump-ptx", os.environ["VIT_GEMM_LIBRARY"]], text=True)
    entry = next(part for part in ptx.split(".visible .entry ") if "gemm_tf32x3ILb1ELb0" in part[:200])
    wait = entry.index("cp.async.wait_group 0;")
    prefetch = entry.index("cp.async.commit_group;", wait)
    compute = entry.index("mma.sync.aligned", prefetch)
    assert "cp.async.wait_group" not in entry[prefetch:compute], "Prefetch is waited on before compute"

    def make_graph(operation):
        for _ in range(3):
            operation()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(10):
                operation()
        return graph

    def elapsed(graph):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
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
        actual = pipeline(x, w, bias)
        torch.testing.assert_close(actual, immediate_wait(x, w, bias), atol=0, rtol=0)
        torch.testing.assert_close(actual, before(x, w, bias), atol=0, rtol=0)
        reference = (x.double() @ w.double().t() + bias.double()).float()
        torch.testing.assert_close(actual, reference, atol=1e-4, rtol=1e-5)

        out = torch.empty_like(actual)
        graphs = {
            "pipeline": make_graph(lambda: pipeline.launch(x, w, bias, out)),
            "immediate_wait": make_graph(lambda: immediate_wait.launch(x, w, bias, out)),
            "before": make_graph(lambda: before.launch(x, w, bias, out)),
        }
        samples = {name: [] for name in graphs}
        names = list(graphs)
        for repeat in range(5):
            for name in names[repeat % 3:] + names[:repeat % 3]:
                samples[name].append(elapsed(graphs[name]))
        row = {"M": m, "K": k, "N": n}
        row.update({f"{name}_ms": statistics.median(values) for name, values in samples.items()})
        row["speedup_vs_before"] = row["before_ms"] / row["pipeline_ms"]
        row["speedup_vs_immediate_wait"] = row["immediate_wait_ms"] / row["pipeline_ms"]
        results.append(row)
        print(json.dumps(row))

    Path("/tmp/cute_pipeline_results.json").write_text(json.dumps({
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "async_copy_128bit": True,
        "prefetch_before_compute_without_wait": True,
        "bitwise_equal_to_controls": True,
        "benchmarks": results,
    }, indent=2))
