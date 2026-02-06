#!/usr/bin/env python3
"""
Preflight environment checks for Who Are You (USENIX 2022) reproduction.

This script verifies:
  - Python version compatibility
  - CUDA toolkit visibility (nvcc)
  - NUMBA_CUDA_NVVM / NUMBA_CUDA_LIBDEVICE paths
  - numba / llvmlite / numpy versions
  - numba CUDA availability

Run:
  python check_env.py
"""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path


def run_cmd(cmd: list[str]) -> str:
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        return (result.stdout + result.stderr).strip()
    except Exception as exc:
        return f"[error] {exc}"


def print_kv(key: str, value: str) -> None:
    print(f"{key}: {value}")


def check_python() -> None:
    ver = sys.version.split()[0]
    major, minor = sys.version_info[:2]
    print_kv("python", ver)
    if major == 3 and minor >= 13:
        print("WARNING: Python 3.13 is not supported by numba. Use Python 3.12.")


def check_cuda_env() -> None:
    cuda_home = os.environ.get("CUDA_HOME", "")
    nvvm = os.environ.get("NUMBA_CUDA_NVVM", "")
    libdevice = os.environ.get("NUMBA_CUDA_LIBDEVICE", "")
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")

    print_kv("CUDA_HOME", cuda_home or "(unset)")
    print_kv("NUMBA_CUDA_NVVM", nvvm or "(unset)")
    print_kv("NUMBA_CUDA_LIBDEVICE", libdevice or "(unset)")
    print_kv("LD_LIBRARY_PATH", ld_path or "(unset)")

    if nvvm:
        nvvm_path = Path(nvvm)
        if not nvvm_path.exists():
            print(f"ERROR: NUMBA_CUDA_NVVM path does not exist: {nvvm}")
    if libdevice:
        libdevice_path = Path(libdevice)
        if not libdevice_path.exists():
            print(f"ERROR: NUMBA_CUDA_LIBDEVICE path does not exist: {libdevice}")
        else:
            bc_files = list(libdevice_path.glob("*.bc"))
            if not bc_files:
                print(f"WARNING: No .bc files found under {libdevice_path}")


def check_cuda_toolkit() -> None:
    nvcc_out = run_cmd(["nvcc", "--version"])
    print_kv("nvcc --version", nvcc_out or "(not found)")

    cuda_dirs = sorted(Path("/usr/local").glob("cuda-*"))
    if cuda_dirs:
        print("CUDA toolkits under /usr/local:")
        for d in cuda_dirs:
            print(f"  - {d}")
    else:
        print("WARNING: No CUDA toolkits found under /usr/local")

    if "not found" in nvcc_out.lower():
        print("INFO: If you only have the NVIDIA driver, install CUDA toolkit via:")
        print("      bash scripts/create_env.sh")


def check_numba_stack() -> None:
    try:
        import numpy as np  # noqa: F401
        import numba  # noqa: F401
        import llvmlite  # noqa: F401

        print_kv("numpy", np.__version__)
        print_kv("numba", numba.__version__)
        print_kv("llvmlite", llvmlite.__version__)

        try:
            from numba import cuda
            from numba.cuda.cudadrv import libs

            nvvm_path = libs.get_cudalib("nvvm")
            libdevice_path = libs.get_libdevice()
            print_kv("numba nvvm", str(nvvm_path))
            print_kv("numba libdevice", str(libdevice_path))
            print_kv("cuda available", str(cuda.is_available()))
        except Exception as exc:
            print(f"ERROR: numba CUDA check failed: {exc}")
    except Exception as exc:
        print(f"ERROR: numba stack import failed: {exc}")


def main() -> None:
    print("== Python ==")
    check_python()
    print("\n== CUDA Env ==")
    check_cuda_env()
    print("\n== CUDA Toolkit ==")
    check_cuda_toolkit()
    print("\n== Numba Stack ==")
    check_numba_stack()

    print("\nIf you see NVVM errors, ensure NUMBA_CUDA_NVVM points to CUDA 12.4/12.9:")
    print("  export CUDA_HOME=/usr/local/cuda-12.4")
    print("  export NUMBA_CUDA_NVVM=$CUDA_HOME/nvvm/lib64/libnvvm.so")
    print("  export NUMBA_CUDA_LIBDEVICE=$CUDA_HOME/nvvm/libdevice")
    print("\nFor conda-based CUDA toolkit setup:")
    print("  bash scripts/create_env.sh")
    print("  export NUMBA_CUDA_NVVM=$CONDA_PREFIX/nvvm/lib64/libnvvm.so")
    print("  export NUMBA_CUDA_LIBDEVICE=$CONDA_PREFIX/nvvm/libdevice")
    print("  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/nvvm/lib64:$LD_LIBRARY_PATH")


if __name__ == "__main__":
    main()
