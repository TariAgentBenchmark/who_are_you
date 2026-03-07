from __future__ import annotations

from dataclasses import dataclass

from who_are_you.config import ReproductionConfig
from who_are_you.fft_features import magnitude_spectrum
from who_are_you.optimizer import coordinate_search
from who_are_you.transfer_function import recover_cross_sectional_areas
from who_are_you.windows import WindowedBigram


@dataclass(slots=True)
class TractEstimate:
    bigram: str
    window_index: int
    tract_areas_cm2: list[float]
    reflection_coefficients: list[float]
    error: float


def estimate_vocal_tract(
    windowed_bigram: WindowedBigram,
    config: ReproductionConfig,
) -> TractEstimate:
    frequencies_hz, magnitudes = magnitude_spectrum(
        waveform=windowed_bigram.waveform,
        sample_rate_hz=config.sample_rate_hz,
        max_frequency_hz=config.max_frequency_hz,
        bin_stride=config.fft_bin_stride,
    )
    search = coordinate_search(
        target_magnitude=magnitudes,
        frequencies_hz=frequencies_hz,
        sample_rate_hz=config.sample_rate_hz,
        num_coefficients=config.num_tract_segments - 1,
        step=config.coordinate_search_step,
        tolerance=config.coordinate_search_tolerance,
        max_iterations=config.coordinate_search_max_iterations,
    )
    tract_areas_cm2 = recover_cross_sectional_areas(
        reflection_coefficients=search.reflection_coefficients,
        initial_area_cm2=config.initial_glottis_area_cm2,
    )
    return TractEstimate(
        bigram=windowed_bigram.bigram,
        window_index=windowed_bigram.window_index,
        tract_areas_cm2=tract_areas_cm2,
        reflection_coefficients=search.reflection_coefficients,
        error=search.error,
    )
