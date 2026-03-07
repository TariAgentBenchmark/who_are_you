from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from who_are_you.bigrams import PhonemeSpan


@dataclass(slots=True)
class WordSpan:
    text: str
    start_sample: int
    end_sample: int


@dataclass(slots=True)
class UtteranceRecord:
    speaker_id: str
    dialect: str
    sentence_id: str
    label: str
    wav_path: Path
    text: str
    words: list[WordSpan]
    phonemes: list[PhonemeSpan]


@dataclass(slots=True)
class SpeakerBundle:
    speaker_id: str
    dialect: str
    organic_dir: Path
    generated_dir: Path


def _parse_txt(path: Path) -> str:
    line = path.read_text().strip()
    if not line:
        return ""
    parts = line.split(maxsplit=2)
    return parts[2] if len(parts) == 3 else ""


def _parse_word_spans(path: Path) -> list[WordSpan]:
    spans: list[WordSpan] = []
    for line in path.read_text().splitlines():
        start, end, text = line.split(maxsplit=2)
        spans.append(WordSpan(text=text, start_sample=int(start), end_sample=int(end)))
    return spans


def _parse_phonemes(path: Path) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for line in path.read_text().splitlines():
        start, end, symbol = line.split(maxsplit=2)
        spans.append((int(start), int(end), symbol))
    return spans


def _match_word(words: list[WordSpan], start_sample: int, end_sample: int) -> str:
    midpoint = (start_sample + end_sample) // 2
    for word in words:
        if word.start_sample <= midpoint < word.end_sample:
            return word.text

    overlap_best = ""
    overlap_size = -1
    for word in words:
        overlap = min(end_sample, word.end_sample) - max(start_sample, word.start_sample)
        if overlap > overlap_size:
            overlap_size = overlap
            overlap_best = word.text
    return overlap_best


def _build_phoneme_spans(
    raw_phonemes: Iterable[tuple[int, int, str]],
    words: list[WordSpan],
) -> list[PhonemeSpan]:
    spans: list[PhonemeSpan] = []
    for start_sample, end_sample, symbol in raw_phonemes:
        spans.append(
            PhonemeSpan(
                symbol=symbol,
                start_sample=start_sample,
                end_sample=end_sample,
                word=_match_word(words, start_sample, end_sample),
            )
        )
    return spans


def read_waveform(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        waveform = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        channels = wav_file.getnchannels()
        if channels > 1:
            waveform = waveform.reshape(-1, channels).mean(axis=1)
        waveform = waveform / 32768.0
        return waveform


def load_utterance(directory: Path, speaker_id: str, dialect: str, sentence_id: str, label: str) -> UtteranceRecord:
    words = _parse_word_spans(directory / f"{sentence_id}.WRD")
    raw_phonemes = _parse_phonemes(directory / f"{sentence_id}.PHN")
    phonemes = _build_phoneme_spans(raw_phonemes, words)
    return UtteranceRecord(
        speaker_id=speaker_id,
        dialect=dialect,
        sentence_id=sentence_id,
        label=label,
        wav_path=directory / f"{sentence_id}.wav",
        text=_parse_txt(directory / f"{sentence_id}.TXT"),
        words=words,
        phonemes=phonemes,
    )


def _organic_speaker_dirs(root: Path) -> dict[str, Path]:
    return {path.name: path for path in root.iterdir() if path.is_dir()}


def _generated_speaker_dirs(root: Path) -> dict[str, tuple[str, Path]]:
    directories: dict[str, tuple[str, Path]] = {}
    for dialect_dir in root.iterdir():
        if not dialect_dir.is_dir():
            continue
        for speaker_dir in dialect_dir.iterdir():
            if speaker_dir.is_dir():
                directories[speaker_dir.name] = (dialect_dir.name, speaker_dir)
    return directories


def discover_speakers(organic_root: Path, generated_root: Path) -> list[SpeakerBundle]:
    organic_dirs = _organic_speaker_dirs(organic_root)
    generated_dirs = _generated_speaker_dirs(generated_root)
    speakers = sorted(set(organic_dirs) & set(generated_dirs))
    return [
        SpeakerBundle(
            speaker_id=speaker_id,
            dialect=generated_dirs[speaker_id][0],
            organic_dir=organic_dirs[speaker_id],
            generated_dir=generated_dirs[speaker_id][1],
        )
        for speaker_id in speakers
    ]


def sentence_ids(directory: Path) -> list[str]:
    return sorted(path.stem for path in directory.glob("*.wav"))


def load_speaker_utterances(
    bundle: SpeakerBundle,
    max_sentence_pairs: int | None = None,
) -> list[UtteranceRecord]:
    organic_ids = set(sentence_ids(bundle.organic_dir))
    generated_ids = set(sentence_ids(bundle.generated_dir))
    paired_sentence_ids = sorted(organic_ids & generated_ids)
    if max_sentence_pairs is not None:
        paired_sentence_ids = paired_sentence_ids[:max_sentence_pairs]
    utterances: list[UtteranceRecord] = []
    for sentence_id in paired_sentence_ids:
        utterances.append(
            load_utterance(
                directory=bundle.organic_dir,
                speaker_id=bundle.speaker_id,
                dialect=bundle.dialect,
                sentence_id=sentence_id,
                label="organic",
            )
        )
        utterances.append(
            load_utterance(
                directory=bundle.generated_dir,
                speaker_id=bundle.speaker_id,
                dialect=bundle.dialect,
                sentence_id=sentence_id,
                label="deepfake",
            )
        )
    return utterances
