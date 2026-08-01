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

    qc, kc, vc = q.cpu(), k.cpu(), v.cpu()

    qkv_ref = X_cpu.matmul(W_cpu.t()) + B_cpu
    q_ref = qkv_ref[:, :, 0:768]
    k_ref = qkv_ref[:, :, 768:1536]
    v_ref = qkv_ref[:, :, 1536:2304]

    assert torch.allclose(qc, q_ref, atol=1e-4), "Q mismatch"
    assert torch.allclose(kc, k_ref, atol=1e-4), "K mismatch"
    assert torch.allclose(vc, v_ref, atol=1e-4), "V mismatch"


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

    assert bool(torch.allclose(Y.cpu(), expected, atol=1e-4))