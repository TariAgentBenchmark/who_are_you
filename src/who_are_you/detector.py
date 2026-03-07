from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from who_are_you.config import ReproductionConfig
from who_are_you.corpus import SpeakerBundle, discover_speakers, load_speaker_utterances
from who_are_you.metrics import BinaryMetrics
from who_are_you.tract_reconstruction import estimate_vocal_tract
from who_are_you.windows import extract_windowed_bigrams


FeatureKey = tuple[str, int, int]


@dataclass(slots=True)
class FeatureObservation:
    key: FeatureKey
    label: str
    value: float
    speaker_id: str
    sentence_id: str


@dataclass(slots=True)
class OrganicRange:
    minimum: float
    maximum: float
    sample_count: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "minimum": self.minimum,
            "maximum": self.maximum,
            "sample_count": self.sample_count,
        }


@dataclass(slots=True)
class IdealFeature:
    bigram: str
    window_index: int
    tract_position: int
    threshold: float
    direction: str
    precision: float
    recall: float
    sample_count: int

    @property
    def key(self) -> FeatureKey:
        return (self.bigram, self.window_index, self.tract_position)

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "bigram": self.bigram,
            "window_index": self.window_index,
            "tract_position": self.tract_position,
            "threshold": self.threshold,
            "direction": self.direction,
            "precision": self.precision,
            "recall": self.recall,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, float | int | str]) -> "IdealFeature":
        return cls(
            bigram=str(payload["bigram"]),
            window_index=int(payload["window_index"]),
            tract_position=int(payload["tract_position"]),
            threshold=float(payload["threshold"]),
            direction=str(payload["direction"]),
            precision=float(payload["precision"]),
            recall=float(payload["recall"]),
            sample_count=int(payload["sample_count"]),
        )


@dataclass(slots=True)
class DetectorModel:
    config: ReproductionConfig
    sampled_speakers: list[str]
    feature_speakers: list[str]
    evaluation_speakers: list[str]
    organic_ranges: dict[FeatureKey, OrganicRange]
    ideal_features: list[IdealFeature]

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config.to_dict(),
            "sampled_speakers": self.sampled_speakers,
            "feature_speakers": self.feature_speakers,
            "evaluation_speakers": self.evaluation_speakers,
            "organic_ranges": [
                {
                    "bigram": key[0],
                    "window_index": key[1],
                    "tract_position": key[2],
                    **value.to_dict(),
                }
                for key, value in sorted(self.organic_ranges.items())
            ],
            "ideal_features": [feature.to_dict() for feature in self.ideal_features],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "DetectorModel":
        config = ReproductionConfig(**payload["config"])
        organic_ranges: dict[FeatureKey, OrganicRange] = {}
        for item in payload["organic_ranges"]:
            key = (item["bigram"], int(item["window_index"]), int(item["tract_position"]))
            organic_ranges[key] = OrganicRange(
                minimum=float(item["minimum"]),
                maximum=float(item["maximum"]),
                sample_count=int(item["sample_count"]),
            )
        ideal_features = [IdealFeature.from_dict(item) for item in payload["ideal_features"]]
        return cls(
            config=config,
            sampled_speakers=list(payload["sampled_speakers"]),
            feature_speakers=list(payload["feature_speakers"]),
            evaluation_speakers=list(payload["evaluation_speakers"]),
            organic_ranges=organic_ranges,
            ideal_features=ideal_features,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "DetectorModel":
        return cls.from_dict(json.loads(path.read_text()))


def split_speakers(
    bundles: list[SpeakerBundle],
    sample_size: int,
    feature_extraction_speakers: int,
    seed: int,
) -> tuple[list[SpeakerBundle], list[SpeakerBundle], list[SpeakerBundle]]:
    if sample_size > len(bundles):
        raise ValueError(f"requested {sample_size} speakers, but only {len(bundles)} paired speakers exist")
    rng = random.Random(seed)
    sampled = rng.sample(bundles, sample_size)
    sampled = sorted(sampled, key=lambda item: item.speaker_id)
    feature = sampled[:feature_extraction_speakers]
    evaluation = sampled[feature_extraction_speakers:]
    return sampled, feature, evaluation


def _observations_for_speaker(bundle: SpeakerBundle, config: ReproductionConfig) -> list[FeatureObservation]:
    observations: list[FeatureObservation] = []
    utterances = load_speaker_utterances(bundle, max_sentence_pairs=config.max_sentence_pairs)
    utterance_progress = tqdm(
        utterances,
        desc=f"{bundle.speaker_id} utterances",
        unit="utt",
        leave=False,
    )
    for utterance in utterance_progress:
        utterance_progress.set_postfix_str(f"{utterance.label}:{utterance.sentence_id}")
        windowed_bigrams = extract_windowed_bigrams(
            utterance=utterance,
            window_size=config.bigram_window_size,
            overlap=config.bigram_window_overlap,
        )
        window_progress = tqdm(
            windowed_bigrams,
            desc=f"{bundle.speaker_id}:{utterance.sentence_id}",
            unit="window",
            leave=False,
        )
        for windowed_bigram in window_progress:
            window_progress.set_postfix_str(
                f"{utterance.label} {windowed_bigram.bigram}#{windowed_bigram.window_index}"
            )
            estimate = estimate_vocal_tract(windowed_bigram, config)
            for tract_position, value in enumerate(estimate.tract_areas_cm2):
                observations.append(
                    FeatureObservation(
                        key=(estimate.bigram, estimate.window_index, tract_position),
                        label=utterance.label,
                        value=float(value),
                        speaker_id=utterance.speaker_id,
                        sentence_id=utterance.sentence_id,
                    )
                )
        window_progress.close()
    utterance_progress.close()
    return observations


def _build_organic_ranges(
    observations: list[FeatureObservation],
) -> dict[FeatureKey, OrganicRange]:
    grouped: dict[FeatureKey, list[float]] = defaultdict(list)
    for observation in observations:
        if observation.label == "organic":
            grouped[observation.key].append(observation.value)
    return {
        key: OrganicRange(minimum=min(values), maximum=max(values), sample_count=len(values))
        for key, values in grouped.items()
        if values
    }


def _search_best_threshold(
    organic_values: list[float],
    deepfake_values: list[float],
    min_precision: float,
    min_recall: float,
) -> IdealFeature | None:
    if not organic_values or not deepfake_values:
        return None

    sorted_values = sorted(set(organic_values + deepfake_values))
    candidates: list[float] = []
    if sorted_values:
        candidates.append(sorted_values[0] - 1e-6)
        for left, right in zip(sorted_values, sorted_values[1:]):
            candidates.append((left + right) / 2.0)
        candidates.append(sorted_values[-1] + 1e-6)
    best: tuple[float, float, float, float, str] | None = None
    for direction in ("lt", "gt"):
        for threshold in candidates:
            if direction == "lt":
                tp = sum(value < threshold for value in deepfake_values)
                fp = sum(value < threshold for value in organic_values)
            else:
                tp = sum(value > threshold for value in deepfake_values)
                fp = sum(value > threshold for value in organic_values)

            if tp == 0:
                continue
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / len(deepfake_values)
            if precision < min_precision or recall < min_recall:
                continue

            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            candidate = (f1, precision, recall, threshold, direction)
            if best is None or candidate > best:
                best = candidate

    if best is None:
        return None
    _, precision, recall, threshold, direction = best
    return IdealFeature(
        bigram="",
        window_index=0,
        tract_position=0,
        threshold=float(threshold),
        direction=direction,
        precision=float(precision),
        recall=float(recall),
        sample_count=len(organic_values) + len(deepfake_values),
    )


def _build_ideal_features(
    observations: list[FeatureObservation],
    min_precision: float,
    min_recall: float,
) -> list[IdealFeature]:
    grouped: dict[FeatureKey, dict[str, list[float]]] = defaultdict(lambda: {"organic": [], "deepfake": []})
    for observation in observations:
        grouped[observation.key][observation.label].append(observation.value)

    candidates: list[IdealFeature] = []
    for key, values in grouped.items():
        feature = _search_best_threshold(
            organic_values=values["organic"],
            deepfake_values=values["deepfake"],
            min_precision=min_precision,
            min_recall=min_recall,
        )
        if feature is None:
            continue
        feature.bigram = key[0]
        feature.window_index = key[1]
        feature.tract_position = key[2]
        candidates.append(feature)

    if not candidates:
        return []

    mean_weight = sum(feature.sample_count for feature in candidates) / len(candidates)
    filtered = [feature for feature in candidates if feature.sample_count >= mean_weight]
    return filtered or candidates


def build_detector(
    organic_root: Path,
    generated_root: Path,
    config: ReproductionConfig,
    seed: int = 1337,
    sample_size: int | None = None,
    feature_extraction_speakers: int | None = None,
    min_precision: float = 0.9,
    min_recall: float = 0.9,
) -> DetectorModel:
    bundles = discover_speakers(organic_root=organic_root, generated_root=generated_root)
    sample_size = sample_size or config.num_sampled_speakers
    feature_extraction_speakers = feature_extraction_speakers or config.feature_extraction_speakers
    sampled, feature, evaluation = split_speakers(
        bundles=bundles,
        sample_size=sample_size,
        feature_extraction_speakers=feature_extraction_speakers,
        seed=seed,
    )

    observations: list[FeatureObservation] = []
    feature_progress = tqdm(feature, desc="build speakers", unit="speaker")
    for bundle in feature_progress:
        feature_progress.set_postfix_str(bundle.speaker_id)
        speaker_observations = _observations_for_speaker(bundle, config)
        observations.extend(speaker_observations)
        feature_progress.set_postfix_str(
            f"{bundle.speaker_id} obs={len(speaker_observations)} total={len(observations)}"
        )
    feature_progress.close()

    organic_ranges = _build_organic_ranges(observations)
    ideal_features = _build_ideal_features(
        observations=observations,
        min_precision=min_precision,
        min_recall=min_recall,
    )
    return DetectorModel(
        config=config,
        sampled_speakers=[bundle.speaker_id for bundle in sampled],
        feature_speakers=[bundle.speaker_id for bundle in feature],
        evaluation_speakers=[bundle.speaker_id for bundle in evaluation],
        organic_ranges=organic_ranges,
        ideal_features=ideal_features,
    )


def _predict_deepfake_from_ranges(
    observations: list[FeatureObservation],
    organic_ranges: dict[FeatureKey, OrganicRange],
) -> tuple[bool, int, int]:
    deepfake_votes = 0
    matched = 0
    for observation in observations:
        organic_range = organic_ranges.get(observation.key)
        if organic_range is None:
            continue
        matched += 1
        if observation.value < organic_range.minimum or observation.value > organic_range.maximum:
            deepfake_votes += 1
    return (deepfake_votes > matched / 2.0) if matched else False, deepfake_votes, matched


def _predict_deepfake_from_ideal(
    observations: list[FeatureObservation],
    ideal_features: list[IdealFeature],
) -> tuple[bool, int, int]:
    feature_map = {feature.key: feature for feature in ideal_features}
    deepfake_votes = 0
    matched = 0
    for observation in observations:
        feature = feature_map.get(observation.key)
        if feature is None:
            continue
        matched += 1
        if feature.direction == "lt":
            flagged = observation.value < feature.threshold
        else:
            flagged = observation.value > feature.threshold
        if flagged:
            deepfake_votes += 1
    return (deepfake_votes > matched / 2.0) if matched else False, deepfake_votes, matched


def _evaluate_bundle(
    bundle: SpeakerBundle,
    model: DetectorModel,
    mode: str,
) -> tuple[bool, bool]:
    config = model.config
    speaker_utterances = load_speaker_utterances(bundle, max_sentence_pairs=config.max_sentence_pairs)
    by_label = {"organic": [], "deepfake": []}
    utterance_progress = tqdm(
        speaker_utterances,
        desc=f"{bundle.speaker_id} utterances",
        unit="utt",
        leave=False,
    )
    for utterance in utterance_progress:
        utterance_progress.set_postfix_str(f"{utterance.label}:{utterance.sentence_id}")
        windowed_bigrams = extract_windowed_bigrams(
            utterance=utterance,
            window_size=config.bigram_window_size,
            overlap=config.bigram_window_overlap,
        )
        window_progress = tqdm(
            windowed_bigrams,
            desc=f"{bundle.speaker_id}:{utterance.sentence_id}",
            unit="window",
            leave=False,
        )
        for windowed_bigram in window_progress:
            window_progress.set_postfix_str(
                f"{utterance.label} {windowed_bigram.bigram}#{windowed_bigram.window_index}"
            )
            estimate = estimate_vocal_tract(windowed_bigram, config)
            for tract_position, value in enumerate(estimate.tract_areas_cm2):
                by_label[utterance.label].append(
                    FeatureObservation(
                        key=(estimate.bigram, estimate.window_index, tract_position),
                        label=utterance.label,
                        value=float(value),
                        speaker_id=utterance.speaker_id,
                        sentence_id=utterance.sentence_id,
                    )
                )
        window_progress.close()
    utterance_progress.close()

    if mode == "range":
        organic_pred, _, _ = _predict_deepfake_from_ranges(by_label["organic"], model.organic_ranges)
        deepfake_pred, _, _ = _predict_deepfake_from_ranges(by_label["deepfake"], model.organic_ranges)
    else:
        organic_pred, _, _ = _predict_deepfake_from_ideal(by_label["organic"], model.ideal_features)
        deepfake_pred, _, _ = _predict_deepfake_from_ideal(by_label["deepfake"], model.ideal_features)

    return organic_pred, deepfake_pred


def evaluate_detector(
    organic_root: Path,
    generated_root: Path,
    model: DetectorModel,
    mode: str = "ideal",
) -> dict[str, object]:
    bundles = {
        bundle.speaker_id: bundle
        for bundle in discover_speakers(organic_root=organic_root, generated_root=generated_root)
    }
    tp = fp = tn = fn = 0
    eval_progress = tqdm(model.evaluation_speakers, desc="eval speakers", unit="speaker")
    for speaker_id in eval_progress:
        bundle = bundles[speaker_id]
        eval_progress.set_postfix_str(speaker_id)
        organic_pred, deepfake_pred = _evaluate_bundle(bundle, model, mode=mode)
        if organic_pred:
            fp += 1
        else:
            tn += 1
        if deepfake_pred:
            tp += 1
        else:
            fn += 1
        eval_progress.set_postfix_str(
            f"{speaker_id} org={'df' if organic_pred else 'org'} syn={'df' if deepfake_pred else 'org'}"
        )
    eval_progress.close()

    metrics = BinaryMetrics(
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
    )
    return {
        "mode": mode,
        "num_ideal_features": len(model.ideal_features),
        "num_organic_ranges": len(model.organic_ranges),
        "metrics": metrics.to_dict(),
    }
