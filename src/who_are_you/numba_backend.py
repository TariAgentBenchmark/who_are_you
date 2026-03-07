from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass(slots=True)
class NumbaSearchResult:
    reflection_coefficients: np.ndarray
    tract_areas_cm2: list[float]
    error: float
    iterations: int


@njit(cache=True)
def tube_length_cm(sample_rate_hz: int, speed_of_sound_cm_per_s: float) -> float:
    return (1.0 / sample_rate_hz) * speed_of_sound_cm_per_s / 2.0


@njit(cache=True)
def _clamp_reflection_coefficients(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    output = np.empty_like(values)
    for index in range(values.shape[0]):
        value = values[index]
        if value < -1.0 + eps:
            output[index] = -1.0 + eps
        elif value > 1.0 - eps:
            output[index] = 1.0 - eps
        else:
            output[index] = value
    return output


@njit(cache=True)
def _integrated_absolute_error(target: np.ndarray, estimate: np.ndarray) -> float:
    total = 0.0
    for index in range(target.shape[0]):
        total += abs(target[index] - estimate[index])
    return total


@njit(cache=True)
def magnitude_response(
    frequencies_hz: np.ndarray,
    reflection_coefficients: np.ndarray,
    sample_rate_hz: int,
    speed_of_sound_cm_per_s: float = 34_300.0,
    glottis_reflection_coefficient: float = 1.0,
    mouth_reflection_coefficient: float = 1.0,
) -> np.ndarray:
    coefficients = _clamp_reflection_coefficients(reflection_coefficients)
    segment_length = tube_length_cm(sample_rate_hz, speed_of_sound_cm_per_s)
    omega = 2.0 * math.pi * frequencies_hz
    phase = np.exp((-2.0 * segment_length / speed_of_sound_cm_per_s) * 1j * omega)

    a = np.ones(omega.shape[0], dtype=np.complex128)
    b = np.zeros(omega.shape[0], dtype=np.complex128)
    c = np.zeros(omega.shape[0], dtype=np.complex128)
    d = np.ones(omega.shape[0], dtype=np.complex128)

    forward_gain = 0.5 * (1.0 + glottis_reflection_coefficient)
    for coeff_index in range(coefficients.shape[0]):
        coefficient = coefficients[coeff_index]
        new_a = a - b * coefficient * phase
        new_b = (-a * coefficient) + (b * phase)
        new_c = c - d * coefficient * phase
        new_d = (-c * coefficient) + (d * phase)
        a = new_a
        b = new_b
        c = new_c
        d = new_d
        forward_gain *= 1.0 + coefficient

    tail_0 = a + b * mouth_reflection_coefficient
    tail_1 = c + d * mouth_reflection_coefficient
    denominator = tail_0 - glottis_reflection_coefficient * tail_1
    for index in range(denominator.shape[0]):
        if abs(denominator[index]) < 1e-12:
            denominator[index] = 1e-12 + 0j

    numerator = forward_gain * np.exp(
        (-segment_length * coefficients.shape[0] / speed_of_sound_cm_per_s) * 1j * omega
    )
    response = numerator / denominator
    magnitudes = np.abs(response)
    for index in range(magnitudes.shape[0]):
        if not np.isfinite(magnitudes[index]):
            magnitudes[index] = 0.0
    return magnitudes


@njit(cache=True)
def coordinate_search(
    target_magnitude: np.ndarray,
    frequencies_hz: np.ndarray,
    sample_rate_hz: int,
    num_coefficients: int,
    step: float = 0.05,
    tolerance: float = 1e-4,
    max_iterations: int = 200,
) -> tuple[np.ndarray, float, int]:
    coefficients = np.zeros(num_coefficients, dtype=np.float64)
    current_estimate = magnitude_response(
        frequencies_hz=frequencies_hz,
        reflection_coefficients=coefficients,
        sample_rate_hz=sample_rate_hz,
    )
    current_error = _integrated_absolute_error(target_magnitude, current_estimate)

    for iteration in range(1, max_iterations + 1):
        improved = False
        for index in range(num_coefficients):
            best_error = current_error
            best_candidate = coefficients.copy()
            for direction in (-1.0, 1.0):
                candidate = coefficients.copy()
                candidate[index] += direction * step
                candidate = _clamp_reflection_coefficients(candidate)
                estimate = magnitude_response(
                    frequencies_hz=frequencies_hz,
                    reflection_coefficients=candidate,
                    sample_rate_hz=sample_rate_hz,
                )
                error = _integrated_absolute_error(target_magnitude, estimate)
                if error + tolerance < best_error:
                    best_error = error
                    best_candidate = candidate

            changed = False
            for coeff_index in range(coefficients.shape[0]):
                if best_candidate[coeff_index] != coefficients[coeff_index]:
                    changed = True
                    break
            if changed:
                coefficients = best_candidate
                current_error = best_error
                improved = True

        if not improved:
            step *= 0.5
            if step < tolerance:
                return coefficients, current_error, iteration

    return coefficients, current_error, max_iterations


def recover_cross_sectional_areas(
    reflection_coefficients: np.ndarray,
    initial_area_cm2: float = 3.7,
) -> list[float]:
    coefficients = np.clip(np.asarray(reflection_coefficients, dtype=np.float64), -1.0 + 1e-6, 1.0 - 1e-6)
    areas = [float(initial_area_cm2)]
    for coefficient in coefficients:
        next_area = areas[-1] * (1.0 + float(coefficient)) / (1.0 - float(coefficient))
        areas.append(float(next_area))
    return areas


def magnitude_spectrum(
    waveform: np.ndarray,
    sample_rate_hz: int,
    max_frequency_hz: float,
    bin_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    stride = max(1, int(bin_stride))
    signal = waveform.astype(np.float64)
    windowed = signal * np.hamming(signal.shape[0])
    magnitudes = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(signal.shape[0], d=1.0 / sample_rate_hz)
    mask = frequencies <= max_frequency_hz
    frequencies = frequencies[mask][::stride]
    magnitudes = magnitudes[mask][::stride]
    if magnitudes.size:
        scale = float(magnitudes.max())
        if scale > 0.0:
            magnitudes = magnitudes / scale
    return frequencies, magnitudes


def estimate_vocal_tract_numba(
    waveform: np.ndarray,
    sample_rate_hz: int,
    max_frequency_hz: float,
    num_coefficients: int,
    step: float,
    tolerance: float,
    max_iterations: int,
    bin_stride: int,
    initial_area_cm2: float,
) -> NumbaSearchResult:
    frequencies_hz, magnitudes = magnitude_spectrum(
        waveform=waveform,
        sample_rate_hz=sample_rate_hz,
        max_frequency_hz=max_frequency_hz,
        bin_stride=bin_stride,
    )
    coefficients, error, iterations = coordinate_search(
        target_magnitude=magnitudes,
        frequencies_hz=frequencies_hz,
        sample_rate_hz=sample_rate_hz,
        num_coefficients=num_coefficients,
        step=step,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return NumbaSearchResult(
        reflection_coefficients=coefficients,
        tract_areas_cm2=recover_cross_sectional_areas(coefficients, initial_area_cm2=initial_area_cm2),
        error=float(error),
        iterations=int(iterations),
    )
