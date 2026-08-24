"""Warmup and cosine learning-rate scheduling with explicit state."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

SCHEDULER_VERSION = "1.0.0"


class WarmupCosineScheduler:
    """Set the learning rate for the next optimizer update.

    ``step_number`` is the number of completed optimizer updates. The initial
    rates therefore correspond to update index zero.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        max_steps: int,
        warmup_steps: int,
        min_learning_rate: float,
    ):
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not 0 <= warmup_steps <= max_steps:
            raise ValueError("warmup_steps must be between 0 and max_steps")
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.max_learning_rates = tuple(
            float(group["lr"]) for group in optimizer.param_groups
        )
        if not self.max_learning_rates:
            raise ValueError("optimizer must contain parameter groups")
        if not 0 <= min_learning_rate <= min(self.max_learning_rates):
            raise ValueError(
                "min_learning_rate must not exceed optimizer learning rates"
            )
        self.min_learning_rate = min_learning_rate
        self.step_number = 0
        self._apply_rates_for_step(self.step_number)

    def _rate_for_step(self, step_number: int, max_rate: float) -> float:
        if self.warmup_steps and step_number < self.warmup_steps:
            return max_rate * (step_number + 1) / self.warmup_steps
        if step_number >= self.max_steps:
            return self.min_learning_rate
        decay_steps = self.max_steps - self.warmup_steps
        if decay_steps == 0:
            return self.min_learning_rate
        progress = (step_number - self.warmup_steps) / decay_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_learning_rate + (
            max_rate - self.min_learning_rate
        ) * cosine

    def _apply_rates_for_step(self, step_number: int) -> None:
        for group, max_rate in zip(
            self.optimizer.param_groups,
            self.max_learning_rates,
            strict=True,
        ):
            group["lr"] = self._rate_for_step(step_number, max_rate)

    def step(self) -> None:
        self.step_number += 1
        self._apply_rates_for_step(self.step_number)

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, object]:
        return {
            "scheduler_version": SCHEDULER_VERSION,
            "step_number": self.step_number,
            "max_steps": self.max_steps,
            "warmup_steps": self.warmup_steps,
            "min_learning_rate": self.min_learning_rate,
            "max_learning_rates": list(self.max_learning_rates),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        expected = {
            "scheduler_version": SCHEDULER_VERSION,
            "max_steps": self.max_steps,
            "warmup_steps": self.warmup_steps,
            "min_learning_rate": self.min_learning_rate,
            "max_learning_rates": list(self.max_learning_rates),
        }
        for key, value in expected.items():
            if state_dict.get(key) != value:
                raise ValueError(f"scheduler checkpoint mismatch for {key}")
        step_number = state_dict.get("step_number")
        if (
            isinstance(step_number, bool)
            or not isinstance(step_number, int)
            or not 0 <= step_number <= self.max_steps
        ):
            raise ValueError("invalid scheduler step_number")
        self.step_number = step_number
        self._apply_rates_for_step(self.step_number)
