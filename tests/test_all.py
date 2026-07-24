#!/usr/bin/env python3
"""
Build the editable extension and run the full test-suite.

Usage: python tests/test_all.py

This script intentionally does NOT define pytest tests — it is a runnable
test-runner script. When executed it will:
  - Rebuild the `ext` extension with `--no-build-isolation`
  - Ensure `pytest` is installed
  - Run `pytest -q`

"""
import os
import sys
import subprocess
from pathlib import Path

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '0'


def run(cmd, check=True):
    print('RUN:', ' '.join(cmd))
    rc = subprocess.call(cmd)
    if check and rc != 0:
        raise SystemExit(rc)
    return rc


def ensure_editable_install(repo_root):
    py = sys.executable
    ext_path = str(repo_root / 'ext')
    run([py, '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'])
    run([py, '-m', 'pip', 'install', '-e', ext_path, '--no-build-isolation'])


def ensure_pytest():
    try:
        import pytest  # noqa: F401
    except Exception:
        run([sys.executable, '-m', 'pip', 'install', 'pytest'])


def main():
    repo_root = Path(__file__).resolve().parent.parent

    print('Repository root:', repo_root)
    print('Python executable:', sys.executable)
    # Ensure kernel tests run: enable all kernels by default for test_all
    os.environ['KERNELS'] = os.environ.get('KERNELS', 'patch_embed,pos_encoding,qkv,gemm_bias')
    print('KERNELS=', os.environ['KERNELS'])

    # 1) Build/install extension so tests import the latest .pyd
    ensure_editable_install(repo_root)

    # 2) Ensure pytest exists
    ensure_pytest()

    # 3) Run pytest against the tests directory
    test_cmd = [sys.executable, '-m', 'pytest', '-q']
    rc = subprocess.call(test_cmd)
    if rc != 0:
        print('Tests failed with exit code', rc)
    else:
        print('All tests passed')
    raise SystemExit(rc)


if __name__ == '__main__':
    main()
