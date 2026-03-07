from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(slots=True)
class PhonemeSpan:
    symbol: str
    start_sample: int
    end_sample: int
    word: str


@dataclass(slots=True)
class BigramSpan:
    first: str
    second: str
    word: str
    start_sample: int
    end_sample: int
    first_start_sample: int
    first_end_sample: int
    second_start_sample: int
    second_end_sample: int

    @property
    def label(self) -> str:
        return f"{self.first}-{self.second}"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["label"] = self.label
        return payload


def build_in_word_bigrams(phonemes: Iterable[PhonemeSpan]) -> list[BigramSpan]:
    spans = list(phonemes)
    bigrams: list[BigramSpan] = []
    for current, nxt in zip(spans, spans[1:]):
        if current.word != nxt.word:
            continue
        bigrams.append(
            BigramSpan(
                first=current.symbol,
                second=nxt.symbol,
                word=current.word,
                start_sample=current.start_sample,
                end_sample=nxt.end_sample,
                first_start_sample=current.start_sample,
                first_end_sample=current.end_sample,
                second_start_sample=nxt.start_sample,
                second_end_sample=nxt.end_sample,
            )
        )
    return bigrams
