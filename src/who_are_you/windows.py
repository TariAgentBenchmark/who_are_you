from __future__ import annotations

from dataclasses import dataclass

import torch

from who_are_you.bigrams import build_in_word_bigrams
from who_are_you.corpus import UtteranceRecord, read_waveform


@dataclass(slots=True)
class WindowedBigram:
    speaker_id: str
    sentence_id: str
    label: str
    word: str
    bigram: str
    window_index: int
    start_sample: int
    end_sample: int
    waveform: torch.Tensor


def _window_starts(
    start_sample: int,
    end_sample: int,
    window_size: int,
    overlap: int,
) -> list[int]:
    hop = max(1, window_size - overlap)
    if end_sample <= start_sample:
        return [start_sample]

    max_start = max(start_sample, end_sample - window_size)
    starts = [start_sample]
    next_start = start_sample
    while next_start < max_start:
        next_start = min(next_start + hop, max_start)
        if next_start != starts[-1]:
            starts.append(next_start)
    return starts


def _slice_with_padding(
    waveform: torch.Tensor,
    start_sample: int,
    window_size: int,
) -> torch.Tensor:
    window = torch.zeros(window_size, dtype=torch.float32)
    src_start = max(0, start_sample)
    src_end = min(len(waveform), start_sample + window_size)
    if src_end > src_start:
        dst_start = src_start - start_sample
        dst_end = dst_start + (src_end - src_start)
        window[dst_start:dst_end] = waveform[src_start:src_end]
    return window


def extract_windowed_bigrams(
    utterance: UtteranceRecord,
    window_size: int,
    overlap: int,
) -> list[WindowedBigram]:
    waveform = read_waveform(utterance.wav_path)
    results: list[WindowedBigram] = []
    for bigram in build_in_word_bigrams(utterance.phonemes):
        for window_index, start_sample in enumerate(
            _window_starts(
                start_sample=bigram.start_sample,
                end_sample=bigram.end_sample,
                window_size=window_size,
                overlap=overlap,
            )
        ):
            window = _slice_with_padding(waveform, start_sample=start_sample, window_size=window_size)
            results.append(
                WindowedBigram(
                    speaker_id=utterance.speaker_id,
                    sentence_id=utterance.sentence_id,
                    label=utterance.label,
                    word=bigram.word,
                    bigram=bigram.label,
                    window_index=window_index,
                    start_sample=start_sample,
                    end_sample=start_sample + window_size,
                    waveform=window,
                )
            )
    return results
