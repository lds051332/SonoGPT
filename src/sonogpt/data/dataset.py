"""Causal-LM encoding for the forward JSON-to-report task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from sonogpt.data.renderers import GeneratedSample
from sonogpt.tokenizer.base import SonoTokenizer

IGNORE_INDEX = -100


class SequenceTooLongError(ValueError):
    """Raised instead of silently truncating a target sequence."""


@dataclass(frozen=True)
class EncodedGenerateSample:
    sample_id: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    target_start: int


@dataclass(frozen=True)
class CausalLMBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor

    def to(self, device: torch.device | str) -> "CausalLMBatch":
        return CausalLMBatch(
            input_ids=self.input_ids.to(device),
            labels=self.labels.to(device),
            attention_mask=self.attention_mask.to(device),
        )


def encode_generate_sample(
    sample: GeneratedSample,
    tokenizer: SonoTokenizer,
    *,
    max_seq_len: int,
) -> EncodedGenerateSample:
    """Encode one row and mask all prompt/control positions from the loss."""

    if sample.task != "generate":
        raise ValueError("only the generate task is supported in V1")
    prompt_ids = (
        [
            tokenizer.bos_id,
            tokenizer.task_generate_id,
            tokenizer.input_id,
        ]
        + tokenizer.encode(sample.input)
        + [tokenizer.target_id]
    )
    target_ids = tokenizer.encode(sample.target) + [tokenizer.eos_id]
    input_ids = tuple(prompt_ids + target_ids)
    if len(input_ids) > max_seq_len:
        raise SequenceTooLongError(
            f"sample {sample.sample_id} needs {len(input_ids)} tokens, "
            f"but max_seq_len is {max_seq_len}"
        )

    labels = (IGNORE_INDEX,) * len(prompt_ids) + tuple(target_ids)
    return EncodedGenerateSample(
        sample_id=sample.sample_id,
        input_ids=input_ids,
        labels=labels,
        target_start=len(prompt_ids),
    )


def collate_encoded_samples(
    samples: Sequence[EncodedGenerateSample], *, pad_id: int
) -> CausalLMBatch:
    """Pad a non-empty batch dynamically while preserving the loss mask."""

    if not samples:
        raise ValueError("cannot collate an empty batch")
    max_length = max(len(sample.input_ids) for sample in samples)
    batch_size = len(samples)
    input_ids = torch.full(
        (batch_size, max_length), pad_id, dtype=torch.long
    )
    labels = torch.full(
        (batch_size, max_length), IGNORE_INDEX, dtype=torch.long
    )
    attention_mask = torch.zeros(
        (batch_size, max_length), dtype=torch.bool
    )

    for row, sample in enumerate(samples):
        length = len(sample.input_ids)
        input_ids[row, :length] = torch.tensor(sample.input_ids, dtype=torch.long)
        labels[row, :length] = torch.tensor(sample.labels, dtype=torch.long)
        attention_mask[row, :length] = True

    return CausalLMBatch(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
    )
