from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from who_are_you.bigrams import build_in_word_bigrams
from who_are_you.corpus import discover_speakers, load_speaker_utterances
from who_are_you.detector import DetectorModel


def _speaker_ids_for_split(model: DetectorModel, speaker_split: str) -> list[str]:
    if speaker_split == "sampled":
        return model.sampled_speakers
    if speaker_split == "feature":
        return model.feature_speakers
    if speaker_split == "evaluation":
        return model.evaluation_speakers
    if speaker_split == "all":
        return []
    raise ValueError(f"unsupported speaker_split: {speaker_split}")


def analyze_ideal_feature_coverage(
    organic_root: Path,
    generated_root: Path,
    model: DetectorModel,
    speaker_split: str = "sampled",
    max_sentence_pairs: int | None = None,
) -> dict[str, int | float | str]:
    bundles = discover_speakers(organic_root=organic_root, generated_root=generated_root)
    bundle_by_speaker = {bundle.speaker_id: bundle for bundle in bundles}

    if speaker_split == "all":
        selected_bundles = bundles
    else:
        selected_speaker_ids = _speaker_ids_for_split(model, speaker_split)
        selected_bundles = [bundle_by_speaker[speaker_id] for speaker_id in selected_speaker_ids]

    ideal_bigrams = {feature.bigram for feature in model.ideal_features}
    observed_unique_bigrams: set[str] = set()
    covered_unique_bigrams: set[str] = set()
    total_bigram_occurrences = 0
    covered_bigram_occurrences = 0
    utterance_count = 0

    speaker_progress = tqdm(selected_bundles, desc="coverage speakers", unit="speaker")
    for bundle in speaker_progress:
        speaker_progress.set_postfix_str(bundle.speaker_id)
        utterances = load_speaker_utterances(
            bundle,
            max_sentence_pairs=max_sentence_pairs if max_sentence_pairs is not None else model.config.max_sentence_pairs,
        )
        utterance_progress = tqdm(
            utterances,
            desc=f"{bundle.speaker_id} utterances",
            unit="utt",
            leave=False,
        )
        for utterance in utterance_progress:
            utterance_progress.set_postfix_str(f"{utterance.label}:{utterance.sentence_id}")
            utterance_count += 1
            bigrams = build_in_word_bigrams(utterance.phonemes)
            total_bigram_occurrences += len(bigrams)
            for bigram in bigrams:
                observed_unique_bigrams.add(bigram.label)
                if bigram.label in ideal_bigrams:
                    covered_bigram_occurrences += 1
                    covered_unique_bigrams.add(bigram.label)
        utterance_progress.close()
    speaker_progress.close()

    coverage_ratio = (
        covered_bigram_occurrences / total_bigram_occurrences if total_bigram_occurrences else 0.0
    )
    unique_bigram_coverage_ratio = (
        len(covered_unique_bigrams) / len(observed_unique_bigrams) if observed_unique_bigrams else 0.0
    )

    return {
        "speaker_split": speaker_split,
        "speaker_count": len(selected_bundles),
        "utterance_count": utterance_count,
        "ideal_feature_count": len(model.ideal_features),
        "ideal_bigram_count": len(ideal_bigrams),
        "observed_unique_bigram_count": len(observed_unique_bigrams),
        "covered_unique_bigram_count": len(covered_unique_bigrams),
        "unique_bigram_coverage_ratio": unique_bigram_coverage_ratio,
        "total_bigram_occurrences": total_bigram_occurrences,
        "covered_bigram_occurrences": covered_bigram_occurrences,
        "occurrence_coverage_ratio": coverage_ratio,
    }
