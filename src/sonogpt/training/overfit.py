"""Small-set overfitting diagnostic for the training pipeline."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch
from torch.nn.utils import clip_grad_norm_

from sonogpt.data.dataset import CausalLMBatch, IGNORE_INDEX
from sonogpt.model.gpt import SonoGPT
from sonogpt.training.reproducibility import set_reproducible_seed


@dataclass(frozen=True)
class OverfitResult:
    steps: int
    sample_count: int
    parameter_count: int
    initial_loss: float
    final_loss: float
    initial_token_accuracy: float
    final_token_accuracy: float
    elapsed_seconds: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@torch.no_grad()
def evaluate_target_metrics(
    model: SonoGPT,
    batch: CausalLMBatch,
    *,
    evaluation_batch_size: int = 8,
) -> tuple[float, float]:
    """Return target-token cross entropy and teacher-forced accuracy."""

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    for start in range(0, batch.input_ids.shape[0], evaluation_batch_size):
        end = start + evaluation_batch_size
        input_ids = batch.input_ids[start:end]
        labels = batch.labels[start:end]
        attention_mask = batch.attention_mask[start:end]
        output = model(
            input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        shifted_labels = labels[:, 1:]
        target_mask = shifted_labels != IGNORE_INDEX
        target_count = int(target_mask.sum())
        if output.loss is None:
            raise RuntimeError("model did not return a loss")
        total_loss += float(output.loss) * target_count
        predictions = output.logits[:, :-1].argmax(dim=-1)
        total_correct += int(
            ((predictions == shifted_labels) & target_mask).sum()
        )
        total_tokens += target_count
    return total_loss / total_tokens, total_correct / total_tokens


def run_overfit(
    model: SonoGPT,
    batch: CausalLMBatch,
    *,
    steps: int = 400,
    batch_size: int = 4,
    learning_rate: float = 3e-3,
    weight_decay: float = 0.0,
    seed: int = 20260824,
    device: str = "cpu",
) -> OverfitResult:
    """Train repeatedly on a tiny fixed set to expose pipeline defects."""

    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch_size must be positive")
    if batch.input_ids.shape[0] < 1:
        raise ValueError("overfit batch must contain at least one sample")
    set_reproducible_seed(seed)
    model.to(device)
    device_batch = batch.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    initial_loss, initial_accuracy = evaluate_target_metrics(
        model, device_batch
    )
    random_generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(
        device_batch.input_ids.shape[0], generator=random_generator
    )
    cursor = 0
    started = time.perf_counter()
    model.train()

    for _ in range(steps):
        if cursor + batch_size > len(order):
            order = torch.randperm(
                device_batch.input_ids.shape[0], generator=random_generator
            )
            cursor = 0
        indices = order[cursor : cursor + batch_size].to(device)
        cursor += batch_size
        optimizer.zero_grad(set_to_none=True)
        output = model(
            device_batch.input_ids.index_select(0, indices),
            labels=device_batch.labels.index_select(0, indices),
            attention_mask=device_batch.attention_mask.index_select(0, indices),
        )
        if output.loss is None:
            raise RuntimeError("model did not return a loss")
        output.loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    elapsed_seconds = time.perf_counter() - started
    final_loss, final_accuracy = evaluate_target_metrics(model, device_batch)
    return OverfitResult(
        steps=steps,
        sample_count=device_batch.input_ids.shape[0],
        parameter_count=model.count_parameters(),
        initial_loss=round(initial_loss, 6),
        final_loss=round(final_loss, 6),
        initial_token_accuracy=round(initial_accuracy, 6),
        final_token_accuracy=round(final_accuracy, 6),
        elapsed_seconds=round(elapsed_seconds, 3),
    )
