import argparse
import os
import time

import torch
import timm

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from vit_cuda.model import VITCUDA


def measure_latency(model, inp, iterations=50, warmup=10, use_graph=False):
    model.eval()
    with torch.no_grad():
        if use_graph:
            model.capture_graph(inp, num_warmup=warmup)
        for _ in range(warmup):
            model(inp)

    # Record all events on the stream, then sync once at the end so the GPU
    # pipeline stays full (no per-iteration synchronize()).
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    with torch.no_grad():
        for i in range(iterations):
            starts[i].record()
            model(inp)
            ends[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    times.sort()
    return times[0], sum(times) / len(times)


def export_trace(model, inp, iterations, use_graph, trace_path):
    from torch.profiler import profile, ProfilerActivity

    model.eval()
    if use_graph:
        model.capture_graph(inp, num_warmup=3)

    with torch.no_grad():
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            for _ in range(iterations):
                model(inp)

    print(f"\n=== Profiler ({'CUDA graph' if use_graph else 'eager'}) ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

    prof.export_chrome_trace(trace_path)
    print(f"Chrome trace exported to: {trace_path}")
    return prof


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ViTCUDA vs timm, optionally with CUDA graph capture"
    )
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--graph", action="store_true", help="enable CUDA graph capture")
    parser.add_argument("--eager", action="store_true", help="also run eager for comparison")
    parser.add_argument("--trace", action="store_true", help="export torch profiler chrome traces")
    parser.add_argument("--trace-dir", type=str, default="results")
    args = parser.parse_args()

    device = "cuda"
    inp = torch.randn(args.batch, 3, 224, 224, device=device, dtype=torch.float32)

    ref = timm.create_model("vit_base_patch16_224", pretrained=False).to(device).eval()
    vit = ViTCUDA().to(device).eval()

    print("Warming and measuring...")
    ref_best, ref_avg = measure_latency(
        ref, inp, iterations=args.iterations, warmup=args.warmup
    )
    print(f"Reference timm ViT  best: {ref_best:.3f} ms  avg: {ref_avg:.3f} ms")

    modes = []
    if args.graph or args.eager:
        modes.append(("eager", False))
    if args.graph:
        modes.append(("cuda_graph", True))
    if not modes:
        modes.append(("eager", False))

    results = {}
    for label, use_graph in modes:
        m = ViTCUDA().to(device).eval()
        best, avg = measure_latency(
            m, inp, iterations=args.iterations, warmup=args.warmup, use_graph=use_graph
        )
        results[label] = best
        print(f"ViTCUDA {label:<10} best: {best:.3f} ms  avg: {avg:.3f} ms  speedup vs timm: {ref_best / best:.2f}x")

    if "eager" in results and "cuda_graph" in results:
        print(f"\nCUDA graph vs eager: {results['eager'] / results['cuda_graph']:.2f}x speedup")

    if args.trace:
        os.makedirs(args.trace_dir, exist_ok=True)
        for label, use_graph in modes:
            m = ViTCUDA().to(device).eval()
            trace_path = os.path.join(args.trace_dir, f"trace_{label}.json")
            export_trace(
                m,
                inp,
                iterations=max(3, args.iterations // 10),
                use_graph=use_graph,
                trace_path=trace_path,
            )


if __name__ == "__main__":
    main()
