from __future__ import annotations

import torch

from who_are_you.runtime import resolve_runtime


def tube_length_cm(
    sample_rate_hz: int,
    speed_of_sound_cm_per_s: float = 34_300.0,
) -> float:
    period_s = 1.0 / sample_rate_hz
    return period_s * speed_of_sound_cm_per_s / 2.0


def clamp_reflection_coefficients(
    values,
    eps: float = 1e-6,
    device: str = "auto",
) -> torch.Tensor:
    runtime = resolve_runtime(device)
    if isinstance(values, torch.Tensor):
        tensor = values.to(dtype=runtime.float_dtype, device=runtime.device)
    else:
        tensor = torch.as_tensor(values, dtype=runtime.float_dtype, device=runtime.device)
    return torch.clamp(tensor, -1.0 + eps, 1.0 - eps)


def recover_cross_sectional_areas(
    reflection_coefficients,
    initial_area_cm2: float = 3.7,
    device: str = "auto",
) -> list[float]:
    coefficients = clamp_reflection_coefficients(reflection_coefficients, device=device)
    areas = [float(initial_area_cm2)]
    for coefficient in coefficients:
        next_area = areas[-1] * (1.0 + float(coefficient.item())) / (1.0 - float(coefficient.item()))
        areas.append(float(next_area))
    return areas


def magnitude_response(
    frequencies_hz,
    reflection_coefficients,
    sample_rate_hz: int,
    speed_of_sound_cm_per_s: float = 34_300.0,
    glottis_reflection_coefficient: float = 1.0,
    mouth_reflection_coefficient: float = 1.0,
    device: str = "auto",
) -> torch.Tensor:
    runtime = resolve_runtime(device)
    segment_length_cm = tube_length_cm(
        sample_rate_hz=sample_rate_hz,
        speed_of_sound_cm_per_s=speed_of_sound_cm_per_s,
    )

    frequencies = torch.as_tensor(frequencies_hz, dtype=runtime.float_dtype, device=runtime.device)
    coefficients = clamp_reflection_coefficients(reflection_coefficients, device=runtime.device)
    omega = (2.0 * torch.pi * frequencies).to(dtype=runtime.float_dtype)
    phase_scale = torch.tensor(
        -2.0 * segment_length_cm / speed_of_sound_cm_per_s,
        dtype=runtime.complex_dtype,
        device=runtime.device,
    )
    phase = torch.exp(phase_scale * (1j * omega.to(dtype=runtime.complex_dtype)))

    a = torch.ones_like(omega, dtype=runtime.complex_dtype, device=runtime.device)
    b = torch.zeros_like(omega, dtype=runtime.complex_dtype, device=runtime.device)
    c = torch.zeros_like(omega, dtype=runtime.complex_dtype, device=runtime.device)
    d = torch.ones_like(omega, dtype=runtime.complex_dtype, device=runtime.device)

    forward_gain = 0.5 * (1.0 + glottis_reflection_coefficient)
    for coefficient in coefficients:
        coefficient_complex = coefficient.to(dtype=runtime.complex_dtype)
        new_a = a - b * coefficient_complex * phase
        new_b = (-a * coefficient_complex) + (b * phase)
        new_c = c - d * coefficient_complex * phase
        new_d = (-c * coefficient_complex) + (d * phase)
        a, b, c, d = new_a, new_b, new_c, new_d
        forward_gain *= 1.0 + float(coefficient.item())

    tail_0 = a + b * mouth_reflection_coefficient
    tail_1 = c + d * mouth_reflection_coefficient
    denominator = tail_0 - glottis_reflection_coefficient * tail_1
    denominator = torch.where(
        torch.abs(denominator) < 1e-12,
        torch.full_like(denominator, 1e-12 + 0j),
        denominator,
    )
    numerator_scale = torch.tensor(
        -segment_length_cm * len(coefficients) / speed_of_sound_cm_per_s,
        dtype=runtime.complex_dtype,
        device=runtime.device,
    )
    numerator = forward_gain * torch.exp(numerator_scale * (1j * omega.to(dtype=runtime.complex_dtype)))
    response = numerator / denominator
    return torch.nan_to_num(torch.abs(response), nan=0.0, posinf=0.0, neginf=0.0)
