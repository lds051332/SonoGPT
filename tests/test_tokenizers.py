from __future__ import annotations

from pathlib import Path

import pytest

from sonogpt.data.semantic_generator import sample_semantic_cases
from sonogpt.data.split import build_generate_splits
from sonogpt.tokenizer.character import CharacterTokenizer
from sonogpt.tokenizer.sentencepiece_bpe import SentencePieceBPETokenizer
from sonogpt.tokenizer.validation import validate_tokenizer


def _training_texts() -> tuple[str, ...]:
    splits = build_generate_splits(sample_semantic_cases(20, seed=51), seed=52)
    return tuple(
        text
        for sample in splits.train
        for text in (sample.input, sample.target)
    )


def test_character_tokenizer_is_reversible_with_unseen_unicode() -> None:
    tokenizer = CharacterTokenizer.train(_training_texts())
    validation = validate_tokenizer(
        tokenizer,
        (
            "甲状腺右叶见8×6mm结节。",
            "训练中未出现的探及🙂仍可往返。",
        ),
    )

    assert validation.unknown_count == 0
    assert tokenizer.pad_id == 0
    assert tokenizer.bos_id == 1
    assert tokenizer.eos_id == 2


def test_character_tokenizer_save_load_is_stable(tmp_path: Path) -> None:
    tokenizer = CharacterTokenizer.train(_training_texts())
    tokenizer_path = tmp_path / "character.json"
    file_hash = tokenizer.save(tokenizer_path)
    restored = CharacterTokenizer.load(tokenizer_path)

    assert restored.vocabulary == tokenizer.vocabulary
    assert restored.content_sha256 == tokenizer.content_sha256
    assert len(file_hash) == 64


def test_validation_rejects_non_reversible_unknown_characters() -> None:
    tokenizer = CharacterTokenizer.train(("甲状腺",), byte_fallback=False)

    with pytest.raises(ValueError, match="round trip failed"):
        validate_tokenizer(tokenizer, ("甲状腺🙂",))


def test_sentencepiece_bpe_round_trip_and_byte_fallback(tmp_path: Path) -> None:
    training_texts = _training_texts()
    tokenizer = SentencePieceBPETokenizer.train(
        training_texts,
        model_path=tmp_path / "sonogpt_bpe.model",
        vocab_size=512,
    )
    repeated = SentencePieceBPETokenizer.train(
        training_texts,
        model_path=tmp_path / "repeat" / "sonogpt_bpe.model",
        vocab_size=512,
    )
    validation = validate_tokenizer(
        tokenizer,
        (
            training_texts[0],
            "未见过的探及🙂字符也必须保留。",
            "数字0.8×0.6mm与JSON{\"a\":1}",
        ),
    )

    assert tokenizer.vocab_size <= 512
    assert tokenizer.vocab_size >= 320
    assert validation.unknown_count == 0
    assert len(tokenizer.model_sha256) == 64
    assert repeated.model_sha256 == tokenizer.model_sha256
