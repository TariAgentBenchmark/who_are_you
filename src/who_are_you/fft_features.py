from __future__ import annotations

import torch

from who_are_you.runtime import resolve_runtime


def magnitude_spectrum(
    waveform: torch.Tensor,
    sample_rate_hz: int,
    max_frequency_hz: float,
    bin_stride: int = 1,
    device: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    runtime = resolve_runtime(device)
    stride = max(1, int(bin_stride))

    signal = waveform.to(dtype=runtime.float_dtype, device=runtime.device)
    if signal.ndim != 1:
        raise ValueError("waveform must be 1-D")

    window = torch.hamming_window(
        len(signal),
        periodic=False,
        dtype=runtime.float_dtype,
        device=runtime.device,
    )
    windowed = signal * window
    magnitudes = torch.abs(torch.fft.rfft(windowed))
    frequencies = torch.fft.rfftfreq(
        len(windowed),
        d=1.0 / sample_rate_hz,
        device=runtime.device,
    ).to(dtype=runtime.float_dtype)
    mask = frequencies <= max_frequency_hz
    frequencies = frequencies[mask][::stride]
    magnitudes = magnitudes[mask][::stride]
    if magnitudes.numel():
        scale = magnitudes.max()
        if float(scale.item()) > 0.0:
            magnitudes = magnitudes / scale
    return frequencies, magnitudes
