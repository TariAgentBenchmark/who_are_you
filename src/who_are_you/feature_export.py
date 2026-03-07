from __future__ import annotations

import csv
from pathlib import Path

from who_are_you.config import ReproductionConfig
from who_are_you.corpus import UtteranceRecord
from who_are_you.tract_reconstruction import TractEstimate
from who_are_you.windows import WindowedBigram


def csv_fieldnames(config: ReproductionConfig) -> list[str]:
    reflection_fields = [
        f"reflection_{index:02d}" for index in range(config.num_tract_segments - 1)
    ]
    tract_fields = [
        f"tract_area_{index:02d}" for index in range(config.num_tract_segments)
    ]
    return [
        "speaker_id",
        "dialect",
        "sentence_id",
        "label",
        "wav_path",
        "word",
        "bigram",
        "window_index",
        "start_sample",
        "end_sample",
        "error",
        *reflection_fields,
        *tract_fields,
    ]


def feature_csv_path(output_dir: Path, utterance: UtteranceRecord) -> Path:
    speaker_dir = output_dir / utterance.speaker_id
    return speaker_dir / f"{utterance.sentence_id}__{utterance.label}.csv"


def build_feature_row(
    utterance: UtteranceRecord,
    windowed_bigram: WindowedBigram,
    estimate: TractEstimate,
) -> dict[str, str | int | float]:
    row: dict[str, str | int | float] = {
        "speaker_id": utterance.speaker_id,
        "dialect": utterance.dialect,
        "sentence_id": utterance.sentence_id,
        "label": utterance.label,
        "wav_path": str(utterance.wav_path),
        "word": windowed_bigram.word,
        "bigram": estimate.bigram,
        "window_index": estimate.window_index,
        "start_sample": windowed_bigram.start_sample,
        "end_sample": windowed_bigram.end_sample,
        "error": estimate.error,
    }
    for index, value in enumerate(estimate.reflection_coefficients):
        row[f"reflection_{index:02d}"] = value
    for index, value in enumerate(estimate.tract_areas_cm2):
        row[f"tract_area_{index:02d}"] = value
    return row


def write_utterance_feature_csv(
    output_dir: Path,
    utterance: UtteranceRecord,
    config: ReproductionConfig,
    rows: list[dict[str, str | int | float]],
) -> Path:
    path = feature_csv_path(output_dir, utterance)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fieldnames(config))
        writer.writeheader()
        writer.writerows(rows)
    return path
