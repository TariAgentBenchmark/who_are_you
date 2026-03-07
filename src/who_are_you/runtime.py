from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(slots=True)
class RuntimeConfig:
    device: str
    float_dtype: torch.dtype
    complex_dtype: torch.dtype


def _mps_available() -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def resolve_runtime(device: str = "auto") -> RuntimeConfig:
    requested = device.lower()
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported device '{device}', expected auto, cpu, cuda, or mps")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
        return RuntimeConfig(device="cuda", float_dtype=torch.float64, complex_dtype=torch.complex128)

    if requested == "mps":
        if not _mps_available():
            raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is False")
        return RuntimeConfig(device="mps", float_dtype=torch.float32, complex_dtype=torch.complex64)

    if requested == "cpu":
        return RuntimeConfig(device="cpu", float_dtype=torch.float64, complex_dtype=torch.complex128)

    if torch.cuda.is_available():
        return RuntimeConfig(device="cuda", float_dtype=torch.float64, complex_dtype=torch.complex128)
    if _mps_available():
        return RuntimeConfig(device="mps", float_dtype=torch.float32, complex_dtype=torch.complex64)
    return RuntimeConfig(device="cpu", float_dtype=torch.float64, complex_dtype=torch.complex128)


def runtime_status(device: str = "auto") -> dict[str, Any]:
    try:
        runtime = resolve_runtime(device)
        resolved_device = runtime.device
        error = None
    except Exception as exc:
        resolved_device = "unavailable"
        error = str(exc)

    return {
        "torch_available": True,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": _mps_available(),
        "requested_device": device,
        "resolved_backend": "torch",
        "resolved_device": resolved_device,
        "error": error,
    }
