import time
import torch
import timm

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from inference import ViTCUDA


def measure_latency(model, inp, iterations=50, warmup=10):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(inp)

    torch.cuda.synchronize()
    times = []
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    for _ in range(iterations):
        starter.record()
        with torch.no_grad():
            _ = model(inp)
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))

    times.sort()
    best_ms = times[0]
    avg_ms = sum(times) / len(times)
    return best_ms, avg_ms


def main():
    device = 'cuda'
    batch = 32
    inp = torch.randn(batch, 3, 224, 224, device=device, dtype=torch.float32)

    # timm reference model (PyTorch)
    ref = timm.create_model('vit_base_patch16_224', pretrained=False).to(device).eval()

    # custom compiled model
    vit = ViTCUDA().to(device).eval()

    print('Warming and measuring...')
    ref_best, ref_avg = measure_latency(ref, inp, iterations=50, warmup=10)
    vit_best, vit_avg = measure_latency(vit, inp, iterations=50, warmup=10)

    print(f'Reference timm ViT best latency: {ref_best:.3f} ms (avg: {ref_avg:.3f} ms)')
    print(f'ViTCUDA best latency: {vit_best:.3f} ms (avg: {vit_avg:.3f} ms)')
    if vit_best > 0:
        print(f'Speedup (timm / vit_cuda): {ref_best / vit_best:.2f}x')


if __name__ == '__main__':
    main()
