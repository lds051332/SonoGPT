"""Tokenizer validation metrics used before model training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sonogpt.tokenizer.base import SonoTokenizer


@dataclass(frozen=True)
class TokenizerValidationResult:
    text_count: int
    token_count: int
    unknown_count: int
    unknown_rate: float
    min_tokens: int
    max_tokens: int
    mean_tokens: float


def validate_tokenizer(
    tokenizer: SonoTokenizer, texts: Iterable[str]
) -> TokenizerValidationResult:
    """Require exact round trips and report token-length/OOV statistics."""

    text_rows = tuple(texts)
    if not text_rows:
        raise ValueError("at least one validation text is required")

    lengths: list[int] = []
    unknown_count = 0
    for text in text_rows:
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids)
        if decoded != text:
            raise ValueError(f"tokenizer round trip failed for: {text!r}")
        lengths.append(len(token_ids))
        unknown_count += token_ids.count(tokenizer.unk_id)

    token_count = sum(lengths)
    return TokenizerValidationResult(
        text_count=len(text_rows),
        token_count=token_count,
        unknown_count=unknown_count,
        unknown_rate=unknown_count / token_count if token_count else 0.0,
        min_tokens=min(lengths),
        max_tokens=max(lengths),
        mean_tokens=round(token_count / len(lengths), 3),
    )
