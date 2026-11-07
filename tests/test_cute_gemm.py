"""GPU checks for the opt-in kernel. Run with tools/modal_cute_gemm.py."""
import ctypes
import os

import pytest
import torch
import torch.nn.functional as F

if not os.environ.get("VIT_GEMM_LIBRARY"):
    pytest.skip("set VIT_GEMM_LIBRARY to the GEMM probe .so", allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GemmLibrary:
    def __init__(self, path):
        self.library = ctypes.CDLL(path)
        self.run = self.library.run_gemm
        self.run.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_bool, ctypes.c_bool, ctypes.c_void_p,
        ]
        self.run.restype = ctypes.c_int

    def launch(self, x, w, bias, out, *, use_bias=True, use_gelu=False):
        status = self.run(
            x.data_ptr(), w.data_ptr(), bias.data_ptr() if bias is not None else 0,
            out.data_ptr(), x.shape[0], x.shape[1], w.shape[0],
            use_bias, use_gelu, torch.cuda.current_stream().cuda_stream,
        )
        assert status == 0, f"CUDA launch failed with error {status}"

    def __call__(self, x, w, bias=None, *, use_bias=True, use_gelu=False):
        out = torch.full((x.shape[0], w.shape[0]), float("nan"), device=x.device)
        self.launch(x, w, bias, out, use_bias=use_bias, use_gelu=use_gelu)
        return out


@pytest.fixture(scope="module")
def gemm():
    return GemmLibrary(os.environ["VIT_GEMM_LIBRARY"])


@pytest.mark.parametrize("shape", [
    (1, 1, 1), (3, 63, 67), (31, 33, 31), (32, 32, 32), (35, 65, 37),
    (8, 768, 3072), (394, 768, 2304),
    (0, 17, 19), (3, 17, 0), (3, 0, 7),
])
@pytest.mark.parametrize("use_bias,use_gelu", [(False, False), (True, False), (True, True)])
def test_gemm(gemm, shape, use_bias, use_gelu):
    torch.manual_seed(0)
    m, k, n = shape
    x = torch.randn(m, k, device="cuda")
    w = torch.randn(n, k, device="cuda")
    bias = torch.randn(n, device="cuda")
    actual = gemm(x, w, bias, use_bias=use_bias, use_gelu=use_gelu)
    expected = x.double() @ w.double().t()
    if use_bias:
        expected += bias.double()
    if use_gelu:
        expected = F.gelu(expected, approximate="none")
    torch.testing.assert_close(actual, expected.float(), atol=1e-4, rtol=1e-5)


def test_residual_correction(gemm):
    """Three TF32 products should substantially improve on one rounded product."""
    torch.manual_seed(1)
    x = torch.randn(127, 257, device="cuda")
    w = torch.randn(129, 257, device="cuda")
    reference = x.double() @ w.double().t()
    actual = gemm(x, w, use_bias=False)

    def round_tf32(tensor):
        bits = tensor.view(torch.int32)
        # Round to nearest, ties to even; these inputs are finite normal values.
        return ((bits + 0xFFF + ((bits >> 13) & 1)) & -8192).view(torch.float32)

    single_tf32 = round_tf32(x).double() @ round_tf32(w).double().t()
    error = (actual.double() - reference).square().mean().sqrt().item()
    single_error = (single_tf32 - reference).square().mean().sqrt().item()
    print(f"TF32x3 RMSE={error:.6g}; single TF32 RMSE={single_error:.6g}")
    assert error < single_error / 100


def test_mlp_chain(gemm):
    """Exercise the same two launchers and tolerances as the existing MLP test."""
    torch.manual_seed(0)
    x = torch.randn(3, 768, device="cuda")
    w1 = torch.randn(3072, 768, device="cuda")
    w2 = torch.randn(768, 3072, device="cuda")
    b1 = torch.randn(3072, device="cuda")
    b2 = torch.randn(768, device="cuda")
    hidden = gemm(x, w1, b1, use_gelu=True)
    out = gemm(hidden, w2, b2)
    hidden_ref = F.gelu(F.linear(x, w1, b1), approximate="none")
    out_ref = F.linear(hidden_ref, w2, b2)
    torch.testing.assert_close(hidden, hidden_ref, atol=5e-4, rtol=1e-5)
    torch.testing.assert_close(out, out_ref, atol=3e-2, rtol=3e-3)


def test_current_stream(gemm):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn(19, 77, device="cuda")
        w = torch.randn(35, 77, device="cuda")
        bias = torch.randn(35, device="cuda")
        actual = gemm(x, w, bias)
        expected = F.linear(x, w, bias)
    stream.synchronize()
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-5)
