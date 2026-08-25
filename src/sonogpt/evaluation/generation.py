"""Greedy generate-task decoding with explicit length errors."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sonogpt.data.dataset import (
    IGNORE_INDEX,
    EncodedGenerateSample,
    SequenceTooLongError,
    build_generate_prompt_ids,
    collate_encoded_samples,
)
from sonogpt.model.gpt import GenerationLimitError, SonoGPT
from sonogpt.tokenizer.base import SonoTokenizer


@dataclass(frozen=True)
class GeneratedReport:
    text: str
    eos_finished: bool
    generated_token_count: int
    error: str | None = None


@dataclass(frozen=True)
class TeacherForcedMetrics:
    loss: float
    token_accuracy: float
    target_token_count: int


def encode_generate_pair(
    *,
    example_id: str,
    input_text: str,
    target_text: str,
    tokenizer: SonoTokenizer,
    max_seq_len: int,
) -> EncodedGenerateSample:
    prompt_ids = build_generate_prompt_ids(input_text, tokenizer)
    target_ids = tokenizer.encode(target_text) + [tokenizer.eos_id]
    input_ids = tuple(prompt_ids + target_ids)
    if len(input_ids) > max_seq_len:
        raise SequenceTooLongError(
            f"sample {example_id} needs {len(input_ids)} tokens, "
            f"but max_seq_len is {max_seq_len}"
        )
    labels = (IGNORE_INDEX,) * len(prompt_ids) + tuple(target_ids)
    return EncodedGenerateSample(
        sample_id=example_id,
        input_ids=input_ids,
        labels=labels,
        target_start=len(prompt_ids),
    )


@torch.no_grad()
def teacher_forced_metrics(
    model: SonoGPT,
    samples: tuple[EncodedGenerateSample, ...],
    *,
    pad_id: int,
    device: torch.device,
    batch_size: int = 8,
) -> TeacherForcedMetrics:
    """Return target-token loss and accuracy without generating."""

    if not samples:
        raise ValueError("cannot score an empty evaluation set")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    for start in range(0, len(samples), batch_size):
        batch = collate_encoded_samples(
            samples[start : start + batch_size], pad_id=pad_id
        ).to(device)
        output = model(
            batch.input_ids,
            labels=batch.labels,
            attention_mask=batch.attention_mask,
        )
        if output.loss is None:
            raise RuntimeError("model did not return an evaluation loss")
        shifted_labels = batch.labels[:, 1:]
        target_mask = shifted_labels != IGNORE_INDEX
        target_count = int(target_mask.sum())
        predictions = output.logits[:, :-1].argmax(dim=-1)
        total_loss += float(output.loss) * target_count
        total_correct += int(((predictions == shifted_labels) & target_mask).sum())
        total_tokens += target_count
    if was_training:
        model.train()
    return TeacherForcedMetrics(
        loss=total_loss / total_tokens,
        token_accuracy=total_correct / total_tokens,
        target_token_count=total_tokens,
    )


@torch.no_grad()
def generate_report(
    model: SonoGPT,
    tokenizer: SonoTokenizer,
    input_text: str,
    *,
    device: torch.device,
) -> GeneratedReport:
    """Greedy-decode one report and fail loudly instead of truncating."""

    prompt_ids = build_generate_prompt_ids(input_text, tokenizer)
    max_seq_len = model.config.max_seq_len
    if len(prompt_ids) >= max_seq_len:
        raise SequenceTooLongError(
            f"generate prompt needs {len(prompt_ids)} tokens, "
            f"but max_seq_len is {max_seq_len}"
        )
    was_training = model.training
    model.eval()
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    try:
        generated = model.generate(
            prompt,
            max_new_tokens=max_seq_len - len(prompt_ids),
            eos_id=tokenizer.eos_id,
        )
    except GenerationLimitError as error:
        if was_training:
            model.train()
        return GeneratedReport(
            text="",
            eos_finished=False,
            generated_token_count=0,
            error=str(error),
        )
    continuation = generated[0, len(prompt_ids) :].tolist()
    if tokenizer.eos_id in continuation:
        continuation = continuation[: continuation.index(tokenizer.eos_id)]
    text = tokenizer.decode(continuation, skip_special_tokens=True)
    if was_training:
        model.train()
    return GeneratedReport(
        text=text,
        eos_finished=True,
        generated_token_count=len(continuation),
    )
