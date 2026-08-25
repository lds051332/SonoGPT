from __future__ import annotations

import pytest

from sonogpt.data.dataset import (
    IGNORE_INDEX,
    SequenceTooLongError,
    build_generate_prompt_ids,
    collate_encoded_samples,
    encode_generate_sample,
)
from sonogpt.data.semantic_generator import sample_semantic_cases
from sonogpt.data.split import build_generate_splits
from sonogpt.tokenizer.character import CharacterTokenizer


def _samples_and_tokenizer():
    splits = build_generate_splits(sample_semantic_cases(20, seed=61), seed=62)
    texts = (
        text
        for sample in splits.train
        for text in (sample.input, sample.target)
    )
    return splits.train, CharacterTokenizer.train(texts)


def test_generate_encoding_masks_prompt_and_keeps_target() -> None:
    samples, tokenizer = _samples_and_tokenizer()
    sample = samples[0]
    encoded = encode_generate_sample(sample, tokenizer, max_seq_len=512)

    assert all(
        label == IGNORE_INDEX for label in encoded.labels[: encoded.target_start]
    )
    assert encoded.input_ids[encoded.target_start - 1] == tokenizer.target_id
    assert encoded.labels[encoded.target_start] == tokenizer.encode(sample.target)[0]
    assert encoded.input_ids[-1] == tokenizer.eos_id
    assert encoded.labels[-1] == tokenizer.eos_id
    assert encoded.input_ids[: encoded.target_start] == tuple(
        build_generate_prompt_ids(sample.input, tokenizer)
    )


def test_encoder_refuses_to_silently_truncate() -> None:
    samples, tokenizer = _samples_and_tokenizer()

    with pytest.raises(SequenceTooLongError, match="needs"):
        encode_generate_sample(samples[0], tokenizer, max_seq_len=32)


def test_collation_right_pads_ids_mask_and_labels() -> None:
    samples, tokenizer = _samples_and_tokenizer()
    encoded = [
        encode_generate_sample(sample, tokenizer, max_seq_len=512)
        for sample in (samples[0], samples[1])
    ]
    batch = collate_encoded_samples(encoded, pad_id=tokenizer.pad_id)

    assert batch.input_ids.shape[0] == 2
    for row, item in enumerate(encoded):
        length = len(item.input_ids)
        assert batch.attention_mask[row, :length].all()
        assert not batch.attention_mask[row, length:].any()
        assert (batch.labels[row, length:] == IGNORE_INDEX).all()
        assert (batch.input_ids[row, length:] == tokenizer.pad_id).all()
