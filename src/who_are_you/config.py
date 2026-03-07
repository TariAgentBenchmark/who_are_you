from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class ReproductionConfig:
    sample_rate_hz: int = 16_000
    speed_of_sound_cm_per_s: float = 34_300.0
    num_sampled_speakers: int = 300
    feature_extraction_speakers: int = 51
    evaluation_speakers: int = 249
    bigram_window_size: int = 565
    bigram_window_overlap: int = 115
    num_tract_segments: int = 15
    glottis_reflection_coefficient: float = 1.0
    mouth_reflection_coefficient: float = 1.0
    initial_glottis_area_cm2: float = 3.7
    max_frequency_hz: float = 5_000.0
    fft_bin_stride: int = 1
    coordinate_search_step: float = 0.05
    coordinate_search_tolerance: float = 1e-4
    coordinate_search_max_iterations: int = 200
    max_sentence_pairs: int | None = None
    speaker_seed: int = 1337

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
