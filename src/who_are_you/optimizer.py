from __future__ import annotations

from dataclasses import dataclass

import torch

from who_are_you.runtime import resolve_runtime
from who_are_you.transfer_function import clamp_reflection_coefficients, magnitude_response


@dataclass(slots=True)
class CoordinateSearchResult:
    reflection_coefficients: list[float]
    error: float
    iterations: int


def integrated_absolute_error(
    target_magnitude: torch.Tensor,
    estimated_magnitude: torch.Tensor,
) -> float:
    return float(torch.abs(target_magnitude - estimated_magnitude).sum().item())


def coordinate_search(
    target_magnitude,
    frequencies_hz,
    sample_rate_hz: int,
    num_coefficients: int,
    step: float = 0.05,
    tolerance: float = 1e-4,
    max_iterations: int = 200,
    device: str = "auto",
) -> CoordinateSearchResult:
    runtime = resolve_runtime(device)
    target = torch.as_tensor(target_magnitude, dtype=runtime.float_dtype, device=runtime.device)
    frequencies = torch.as_tensor(frequencies_hz, dtype=runtime.float_dtype, device=runtime.device)
    coefficients = torch.zeros(num_coefficients, dtype=runtime.float_dtype, device=runtime.device)

    current_estimate = magnitude_response(
        frequencies_hz=frequencies,
        reflection_coefficients=coefficients,
        sample_rate_hz=sample_rate_hz,
        device=runtime.device,
    )
    current_error = integrated_absolute_error(target, current_estimate)

    for iteration in range(1, max_iterations + 1):
        improved = False
        for index in range(num_coefficients):
            best_error = current_error
            best_candidate = None
            for direction in (-1.0, 1.0):
                candidate = coefficients.clone()
                candidate[index] += direction * step
                candidate = clamp_reflection_coefficients(candidate, device=runtime.device)
                estimate = magnitude_response(
                    frequencies_hz=frequencies,
                    reflection_coefficients=candidate,
                    sample_rate_hz=sample_rate_hz,
                    device=runtime.device,
                )
                error = integrated_absolute_error(target, estimate)
                if error + tolerance < best_error:
                    best_error = error
                    best_candidate = candidate

            if best_candidate is not None:
                coefficients = best_candidate
                current_error = best_error
                improved = True

        if not improved:
            step *= 0.5
            if step < tolerance:
                break

    return CoordinateSearchResult(
        reflection_coefficients=coefficients.detach().cpu().tolist(),
        error=current_error,
        iterations=iteration,
    )
