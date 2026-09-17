import os

import pytest
import torch
import timm

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

kernels_env = os.getenv('KERNELS', 'patch_embed')
if 'end_to_end' not in kernels_env.split(','):
    pytest.skip('end_to_end tests disabled via KERNELS env', allow_module_level=True)

try:
    import vit_cuda
except Exception:
    pytest.skip('vit_cuda extension not available', allow_module_level=True)

from vit_cuda.model import ViTCUDA


def test_end_to_end_batch_matches_timm():
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available on this runner')

    torch.manual_seed(0)
    B = 8  # batch>1 guards against hardcoded-batch regressions (Phase-5 bug class)
    C, H, W = 3, 224, 224

    # Random weights (no pretrained download); same state for both models
    ref = timm.create_model('vit_base_patch16_224', pretrained=False)
    state = ref.state_dict()

    ref = ref.to('cuda').eval()
    vit = ViTCUDA(state_dict=state, pretrained=False).to('cuda').eval()

    x = torch.randn(B, C, H, W, dtype=torch.float32, device='cuda')

    with torch.no_grad():
        ref_out = ref(x)
        vit_out = vit(x)

    assert vit_out.shape == ref_out.shape
    max_abs = (vit_out - ref_out).abs().max().item()
    assert max_abs < 1e-2, f"max abs diff too large: {max_abs}"


def test_end_to_end_cuda_graph_matches_eager():
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available on this runner')

    torch.manual_seed(1)
    B = 4
    x = torch.randn(B, 3, 224, 224, dtype=torch.float32, device='cuda')

    state = timm.create_model('vit_base_patch16_224', pretrained=False).state_dict()
    vit = ViTCUDA(state_dict=state, pretrained=False).to('cuda').eval()

    with torch.no_grad():
        eager_out = vit(x)

    vit.capture_graph(x, num_warmup=3)

    with torch.no_grad():
        graph_out = vit(x)

    max_abs = (eager_out - graph_out).abs().max().item()
    assert max_abs < 1e-5, f"graph vs eager mismatch: {max_abs}"
