"""Versioned configuration for reproducible full-model training."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

TRAINING_CONFIG_VERSION = "1.0.0"
AmpDType = Literal["none", "float16", "bfloat16"]


@dataclass(frozen=True)
class TrainingConfig:
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_steps: int = 5000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    adam_epsilon: float = 1e-8
    gradient_clip_norm: float = 1.0
    evaluation_interval: int = 100
    evaluation_batch_size: int = 8
    checkpoint_interval: int = 100
    log_interval: int = 10
    amp_dtype: AmpDType = "float16"
    seed: int = 20260824
    deterministic: bool = True

    def __post_init__(self) -> None:
        positive_integers = {
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_steps": self.max_steps,
            "evaluation_interval": self.evaluation_interval,
            "evaluation_batch_size": self.evaluation_batch_size,
            "checkpoint_interval": self.checkpoint_interval,
            "log_interval": self.log_interval,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in positive_integers.values()
        ):
            raise ValueError("training count fields must be positive integers")
        if (
            isinstance(self.warmup_steps, bool)
            or not isinstance(self.warmup_steps, int)
            or not 0 <= self.warmup_steps <= self.max_steps
        ):
            raise ValueError("warmup_steps must be between 0 and max_steps")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")

        finite_values = {
            "learning_rate": self.learning_rate,
            "min_learning_rate": self.min_learning_rate,
            "weight_decay": self.weight_decay,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "adam_epsilon": self.adam_epsilon,
            "gradient_clip_norm": self.gradient_clip_norm,
        }
        if any(not math.isfinite(value) for value in finite_values.values()):
            raise ValueError("training numeric fields must be finite")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.min_learning_rate <= self.learning_rate:
            raise ValueError(
                "min_learning_rate must be between 0 and learning_rate"
            )
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("Adam betas must be in [0, 1)")
        if self.adam_epsilon <= 0 or self.gradient_clip_norm <= 0:
            raise ValueError(
                "adam_epsilon and gradient_clip_norm must be positive"
            )
        if self.amp_dtype not in {"none", "float16", "bfloat16"}:
            raise ValueError(f"unsupported amp_dtype: {self.amp_dtype}")

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> dict[str, object]:
        return {"config_version": TRAINING_CONFIG_VERSION, **asdict(self)}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "TrainingConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.pop("config_version", None) != TRAINING_CONFIG_VERSION:
            raise ValueError("unsupported training config version")
        return cls(**payload)


def apply_runtime_overrides(
    config: TrainingConfig,
    *,
    max_steps: int | None = None,
    micro_batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
) -> TrainingConfig:
    updates: dict[str, object] = {}
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("max_steps override must be positive")
        updates.update(
            {
                "max_steps": max_steps,
                "warmup_steps": min(config.warmup_steps, max_steps),
                "evaluation_interval": min(
                    config.evaluation_interval, max_steps
                ),
                "checkpoint_interval": min(
                    config.checkpoint_interval, max_steps
                ),
                "log_interval": min(config.log_interval, max_steps),
            }
        )
    if micro_batch_size is not None:
        updates["micro_batch_size"] = micro_batch_size
    if gradient_accumulation_steps is not None:
        updates["gradient_accumulation_steps"] = (
            gradient_accumulation_steps
        )
    return replace(config, **updates) if updates else config
