"""Reproducible AdamW training loop for the JSON-to-report model."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_

from sonogpt.data.dataset import (
    IGNORE_INDEX,
    CausalLMBatch,
    EncodedGenerateSample,
    collate_encoded_samples,
)
from sonogpt.model.gpt import SonoGPT
from sonogpt.training.checkpoint import (
    capture_random_state,
    load_checkpoint,
    restore_random_state,
    save_checkpoint,
    write_latest_pointer,
)
from sonogpt.training.config import TrainingConfig
from sonogpt.training.reproducibility import set_reproducible_seed
from sonogpt.training.scheduler import WarmupCosineScheduler

BATCH_STREAM_VERSION = "1.0.0"


@dataclass
class TrainingState:
    global_step: int = 0
    micro_batches_seen: int = 0
    examples_seen: int = 0
    target_tokens_seen: int = 0
    skipped_optimizer_steps: int = 0
    best_validation_loss: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TrainingState":
        return cls(**dict(payload))  # type: ignore[arg-type]


@dataclass(frozen=True)
class EvaluationMetrics:
    loss: float
    token_accuracy: float
    target_token_count: int
    sample_count: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingStepMetrics:
    global_step: int
    loss: float
    learning_rate: float
    gradient_norm: float
    example_count: int
    target_token_count: int
    elapsed_seconds: float
    optimizer_step_applied: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingRuntimeMetrics:
    attempted_optimizer_steps: int
    applied_optimizer_steps: int
    example_count: int
    target_token_count: int
    training_step_seconds: float

    @property
    def examples_per_second(self) -> float:
        if self.training_step_seconds == 0:
            return 0.0
        return self.example_count / self.training_step_seconds

    @property
    def target_tokens_per_second(self) -> float:
        if self.training_step_seconds == 0:
            return 0.0
        return self.target_token_count / self.training_step_seconds

    def to_dict(self) -> dict[str, int | float]:
        return {
            **asdict(self),
            "examples_per_second": self.examples_per_second,
            "target_tokens_per_second": self.target_tokens_per_second,
        }


@dataclass(frozen=True)
class TrainingResult:
    state: TrainingState
    final_validation: EvaluationMetrics
    runtime: TrainingRuntimeMetrics


class DeterministicBatchStream:
    """Infinite shuffled index stream with serializable epoch/cursor state."""

    def __init__(self, dataset_size: int, batch_size: int, seed: int):
        if dataset_size <= 0 or batch_size <= 0:
            raise ValueError("dataset_size and batch_size must be positive")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.position = 0
        self._order = self._order_for_epoch(self.epoch)

    def _order_for_epoch(self, epoch: int) -> tuple[int, ...]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + epoch)
        return tuple(
            torch.randperm(self.dataset_size, generator=generator).tolist()
        )

    def next_indices(self) -> tuple[int, ...]:
        indices: list[int] = []
        while len(indices) < self.batch_size:
            if self.position == self.dataset_size:
                self.epoch += 1
                self.position = 0
                self._order = self._order_for_epoch(self.epoch)
            take = min(
                self.batch_size - len(indices),
                self.dataset_size - self.position,
            )
            indices.extend(self._order[self.position : self.position + take])
            self.position += take
        return tuple(indices)

    def state_dict(self) -> dict[str, object]:
        return {
            "batch_stream_version": BATCH_STREAM_VERSION,
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "epoch": self.epoch,
            "position": self.position,
        }

    def load_state_dict(self, payload: Mapping[str, object]) -> None:
        expected = {
            "batch_stream_version": BATCH_STREAM_VERSION,
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "seed": self.seed,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"batch stream checkpoint mismatch for {key}")
        epoch = payload.get("epoch")
        position = payload.get("position")
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch < 0
            or isinstance(position, bool)
            or not isinstance(position, int)
            or not 0 <= position <= self.dataset_size
        ):
            raise ValueError("invalid batch stream position")
        self.epoch = epoch
        self.position = position
        self._order = self._order_for_epoch(self.epoch)


def build_adamw_optimizer(
    model: SonoGPT, config: TrainingConfig
) -> torch.optim.AdamW:
    decay_parameters: list[torch.nn.Parameter] = []
    no_decay_parameters: list[torch.nn.Parameter] = []
    for _, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2:
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)
    parameter_groups = [
        {"params": decay_parameters, "weight_decay": config.weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        parameter_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.adam_epsilon,
    )


class Trainer:
    def __init__(
        self,
        model: SonoGPT,
        train_samples: Sequence[EncodedGenerateSample],
        validation_samples: Sequence[EncodedGenerateSample],
        *,
        pad_id: int,
        config: TrainingConfig,
        device: str | torch.device,
        run_identity: Mapping[str, str] | None = None,
    ):
        if not train_samples or not validation_samples:
            raise ValueError("training and validation samples must be non-empty")
        self.config = config
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        self.train_samples = tuple(train_samples)
        self.validation_samples = tuple(validation_samples)
        self.pad_id = pad_id
        self.run_identity = dict(sorted((run_identity or {}).items()))

        set_reproducible_seed(
            self.config.seed, deterministic=self.config.deterministic
        )
        self.model = model.to(self.device)
        self.optimizer = build_adamw_optimizer(self.model, self.config)
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            max_steps=self.config.max_steps,
            warmup_steps=self.config.warmup_steps,
            min_learning_rate=self.config.min_learning_rate,
        )
        self.amp_dtype = self._resolve_amp_dtype()
        self.scaler_enabled = self.amp_dtype == torch.float16
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.scaler_enabled
        )
        self.batch_stream = DeterministicBatchStream(
            len(self.train_samples),
            self.config.micro_batch_size,
            self.config.seed,
        )
        self.state = TrainingState()

    def _resolve_amp_dtype(self) -> torch.dtype | None:
        if self.device.type != "cuda" or self.config.amp_dtype == "none":
            return None
        if self.config.amp_dtype == "bfloat16":
            if not torch.cuda.is_bf16_supported():
                raise ValueError("this CUDA device does not support bfloat16")
            return torch.bfloat16
        return torch.float16

    def _autocast(self) -> Any:
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
        )

    def _collate_indices(self, indices: Sequence[int]) -> CausalLMBatch:
        return collate_encoded_samples(
            [self.train_samples[index] for index in indices],
            pad_id=self.pad_id,
        )

    @staticmethod
    def _target_token_count(batch: CausalLMBatch) -> int:
        return int((batch.labels[:, 1:] != IGNORE_INDEX).sum())

    def train_step(self) -> TrainingStepMetrics:
        started = time.perf_counter()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        micro_batches = [
            self._collate_indices(self.batch_stream.next_indices())
            for _ in range(self.config.gradient_accumulation_steps)
        ]
        target_counts = [
            self._target_token_count(batch) for batch in micro_batches
        ]
        total_target_tokens = sum(target_counts)
        if total_target_tokens <= 0:
            raise RuntimeError("gradient accumulation contains no target tokens")

        weighted_loss = 0.0
        for batch, target_count in zip(
            micro_batches, target_counts, strict=True
        ):
            device_batch = batch.to(self.device)
            with self._autocast():
                output = self.model(
                    device_batch.input_ids,
                    labels=device_batch.labels,
                    attention_mask=device_batch.attention_mask,
                )
                if output.loss is None:
                    raise RuntimeError("model did not return a training loss")
                loss_weight = target_count / total_target_tokens
                backward_loss = output.loss * loss_weight
            weighted_loss += float(output.loss.detach()) * loss_weight
            self.scaler.scale(backward_loss).backward()

        if self.scaler_enabled:
            self.scaler.unscale_(self.optimizer)
        gradient_norm_tensor = clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.gradient_clip_norm,
        )
        gradient_norm = float(gradient_norm_tensor)
        if not math.isfinite(gradient_norm) and not self.scaler_enabled:
            raise RuntimeError("non-finite gradient norm without AMP scaler")

        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        optimizer_step_applied = True
        if self.scaler_enabled:
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            optimizer_step_applied = self.scaler.get_scale() >= scale_before
        else:
            self.optimizer.step()

        self.state.micro_batches_seen += len(micro_batches)
        self.state.examples_seen += (
            len(micro_batches) * self.config.micro_batch_size
        )
        self.state.target_tokens_seen += total_target_tokens
        if optimizer_step_applied:
            self.scheduler.step()
            self.state.global_step += 1
        else:
            self.state.skipped_optimizer_steps += 1

        return TrainingStepMetrics(
            global_step=self.state.global_step,
            loss=weighted_loss,
            learning_rate=learning_rate,
            gradient_norm=gradient_norm,
            example_count=(
                len(micro_batches) * self.config.micro_batch_size
            ),
            target_token_count=total_target_tokens,
            elapsed_seconds=time.perf_counter() - started,
            optimizer_step_applied=optimizer_step_applied,
        )

    @torch.no_grad()
    def evaluate(self) -> EvaluationMetrics:
        was_training = self.model.training
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        for start in range(
            0,
            len(self.validation_samples),
            self.config.evaluation_batch_size,
        ):
            samples = self.validation_samples[
                start : start + self.config.evaluation_batch_size
            ]
            batch = collate_encoded_samples(samples, pad_id=self.pad_id).to(
                self.device
            )
            with self._autocast():
                output = self.model(
                    batch.input_ids,
                    labels=batch.labels,
                    attention_mask=batch.attention_mask,
                )
            if output.loss is None:
                raise RuntimeError("model did not return a validation loss")
            shifted_labels = batch.labels[:, 1:]
            target_mask = shifted_labels != IGNORE_INDEX
            target_count = int(target_mask.sum())
            predictions = output.logits[:, :-1].argmax(dim=-1)
            total_loss += float(output.loss) * target_count
            total_correct += int(
                ((predictions == shifted_labels) & target_mask).sum()
            )
            total_tokens += target_count
        if was_training:
            self.model.train()
        return EvaluationMetrics(
            loss=total_loss / total_tokens,
            token_accuracy=total_correct / total_tokens,
            target_token_count=total_tokens,
            sample_count=len(self.validation_samples),
        )

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "training_state": self.state.to_dict(),
            "batch_stream_state": self.batch_stream.state_dict(),
            "random_state": capture_random_state(),
            "model_config": self.model.config.to_dict(),
            "training_config": self.config.to_dict(),
            "run_identity": self.run_identity,
            "runtime": {
                "device_type": self.device.type,
                "amp_dtype": self.config.amp_dtype,
                "amp_enabled": self.amp_dtype is not None,
                "scaler_enabled": self.scaler_enabled,
            },
        }

    def save_training_checkpoint(self, path: Path) -> str:
        return save_checkpoint(path, self.checkpoint_payload())

    def _move_optimizer_state_to_device(self) -> None:
        for parameter_state in self.optimizer.state.values():
            for key, value in parameter_state.items():
                if isinstance(value, torch.Tensor):
                    parameter_state[key] = value.to(self.device)

    def load_training_checkpoint(self, path: Path) -> None:
        payload = load_checkpoint(path)
        expected = {
            "model_config": self.model.config.to_dict(),
            "training_config": self.config.to_dict(),
            "run_identity": self.run_identity,
            "runtime": {
                "device_type": self.device.type,
                "amp_dtype": self.config.amp_dtype,
                "amp_enabled": self.amp_dtype is not None,
                "scaler_enabled": self.scaler_enabled,
            },
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"checkpoint mismatch for {key}")

        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self._move_optimizer_state_to_device()
        self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        self.scaler.load_state_dict(payload["scaler_state_dict"])
        self.state = TrainingState.from_dict(payload["training_state"])
        self.batch_stream.load_state_dict(payload["batch_stream_state"])
        if self.scheduler.step_number != self.state.global_step:
            raise ValueError("scheduler and training global steps differ")
        restore_random_state(payload["random_state"])

    def fit(
        self,
        *,
        checkpoint_directory: Path,
        event_callback: Callable[[dict[str, object]], None] | None = None,
        progress_callback: (
            Callable[[str, int | None, int | None, str | None], None] | None
        ) = None,
    ) -> TrainingResult:
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        final_validation: EvaluationMetrics | None = None
        attempted_optimizer_steps = 0
        applied_optimizer_steps = 0
        example_count = 0
        target_token_count = 0
        training_step_seconds = 0.0
        if progress_callback is not None:
            progress_callback(
                "training",
                self.state.global_step,
                self.config.max_steps,
                "optimizer updates",
            )

        while self.state.global_step < self.config.max_steps:
            step_metrics = self.train_step()
            attempted_optimizer_steps += 1
            example_count += step_metrics.example_count
            target_token_count += step_metrics.target_token_count
            training_step_seconds += step_metrics.elapsed_seconds
            if progress_callback is not None:
                progress_callback(
                    "training",
                    self.state.global_step,
                    self.config.max_steps,
                    (
                        f"loss={step_metrics.loss:.6f}, "
                        f"lr={step_metrics.learning_rate:.8f}"
                    ),
                )
            if not step_metrics.optimizer_step_applied:
                if event_callback is not None:
                    event_callback(
                        {"event": "optimizer_step_skipped", **step_metrics.to_dict()}
                    )
                continue
            applied_optimizer_steps += 1

            should_evaluate = (
                self.state.global_step % self.config.evaluation_interval == 0
                or self.state.global_step == self.config.max_steps
            )
            if should_evaluate:
                if progress_callback is not None:
                    progress_callback(
                        "validation",
                        self.state.global_step,
                        self.config.max_steps,
                        f"{len(self.validation_samples)} samples",
                    )
                final_validation = self.evaluate()
                if (
                    self.state.best_validation_loss is None
                    or final_validation.loss
                    < self.state.best_validation_loss
                ):
                    self.state.best_validation_loss = final_validation.loss
                if progress_callback is not None:
                    progress_callback(
                        "training",
                        self.state.global_step,
                        self.config.max_steps,
                        f"validation_loss={final_validation.loss:.6f}",
                    )

            should_log = (
                self.state.global_step % self.config.log_interval == 0
                or should_evaluate
            )
            if should_log and event_callback is not None:
                event: dict[str, object] = {
                    "event": "train_step",
                    **step_metrics.to_dict(),
                    "state": self.state.to_dict(),
                }
                if final_validation is not None and should_evaluate:
                    event["validation"] = final_validation.to_dict()
                event_callback(event)

            should_checkpoint = (
                self.state.global_step % self.config.checkpoint_interval == 0
                or self.state.global_step == self.config.max_steps
            )
            if should_checkpoint:
                checkpoint_path = checkpoint_directory / (
                    f"step_{self.state.global_step:08d}.pt"
                )
                if progress_callback is not None:
                    progress_callback(
                        "checkpoint",
                        self.state.global_step,
                        self.config.max_steps,
                        checkpoint_path.name,
                    )
                checkpoint_sha256 = self.save_training_checkpoint(
                    checkpoint_path
                )
                write_latest_pointer(
                    checkpoint_directory,
                    checkpoint_path,
                    checkpoint_sha256,
                )
                if event_callback is not None:
                    event_callback(
                        {
                            "event": "checkpoint",
                            "global_step": self.state.global_step,
                            "path": checkpoint_path.name,
                            "sha256": checkpoint_sha256,
                        }
                    )
                if progress_callback is not None:
                    progress_callback(
                        "training",
                        self.state.global_step,
                        self.config.max_steps,
                        "checkpoint saved",
                    )

        if final_validation is None:
            if progress_callback is not None:
                progress_callback(
                    "validation",
                    self.state.global_step,
                    self.config.max_steps,
                    f"{len(self.validation_samples)} samples",
                )
            final_validation = self.evaluate()
        return TrainingResult(
            state=TrainingState.from_dict(self.state.to_dict()),
            final_validation=final_validation,
            runtime=TrainingRuntimeMetrics(
                attempted_optimizer_steps=attempted_optimizer_steps,
                applied_optimizer_steps=applied_optimizer_steps,
                example_count=example_count,
                target_token_count=target_token_count,
                training_step_seconds=training_step_seconds,
            ),
        )
