import os
import pytest
import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

kernels_env = os.getenv('KERNELS', 'patch_embed')
if 'qkv' not in kernels_env.split(','):
    pytest.skip('qkv tests disabled via KERNELS env', allow_module_level=True)

try:
    import vit_cuda
except Exception:
    pytest.skip('vit_cuda extension not available', allow_module_level=True)


def test_qkv_proj_correctness():
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available on this runner')

    torch.manual_seed(0)

    B = 2
    N = 197
    E = 768

    X_cpu = torch.randn(B, N, E, dtype=torch.float32)
    W_cpu = torch.randn(3 * E, E, dtype=torch.float32)
    B_cpu = torch.randn(3 * E, dtype=torch.float32)

    X = X_cpu.cuda()
    W = W_cpu.cuda()
    B = B_cpu.cuda()

    q, k, v = vit_cuda.qkv_proj(X, W, B)

    assert q.shape == (B, N, E)
    assert k.shape == (B, N, E)
    assert v.shape == (B, N, E)

    qkv_reference = F.linear(X_cpu, W_cpu, B_cpu)
    qkv_reference = qkv_reference.view(B, N, 3, E)
    q_expected = qkv_reference[:, :, 0, :].contiguous()
    k_expected = qkv_reference[:, :, 1, :].contiguous()
    v_expected = qkv_reference[:, :, 2, :].contiguous()

    assert torch.allclose(q, q_expected, atol=1e-4)
    assert torch.allclose(k, k_expected, atol=1e-4)
    assert torch.allclose(v, v_expected, atol=1e-4)


def test_gemm_bias_correctness():
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available on this runner')

    torch.manual_seed(0)

    M = 8
    K = 768
    N_out = 3072

    X_cpu = torch.randn(M, K, dtype=torch.float32)
    W_cpu = torch.randn(N_out, K, dtype=torch.float32)
    B_cpu = torch.randn(N_out, dtype=torch.float32)

    X = X_cpu.cuda()
    W = W_cpu.cuda()
    B = B_cpu.cuda()

    Y = vit_cuda.gemm_bias(X, W, B)

    expected = X_cpu.matmul(W_cpu.t()) + B_cpu

    assert torch.allclose(Y, expected, atol=1e-4)