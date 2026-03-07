from __future__ import annotations

from dataclasses import dataclass

from who_are_you.config import ReproductionConfig
from who_are_you.numba_backend import estimate_vocal_tract_numba
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
    search = estimate_vocal_tract_numba(
        waveform=windowed_bigram.waveform,
        sample_rate_hz=config.sample_rate_hz,
        max_frequency_hz=config.max_frequency_hz,
        num_coefficients=config.num_tract_segments - 1,
        step=config.coordinate_search_step,
        tolerance=config.coordinate_search_tolerance,
        max_iterations=config.coordinate_search_max_iterations,
        bin_stride=config.fft_bin_stride,
        initial_area_cm2=config.initial_glottis_area_cm2,
    )
    return TractEstimate(
        bigram=windowed_bigram.bigram,
        window_index=windowed_bigram.window_index,
        tract_areas_cm2=search.tract_areas_cm2,
        reflection_coefficients=search.reflection_coefficients.tolist(),
        error=search.error,
    )
