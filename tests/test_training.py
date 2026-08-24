from __future__ import annotations

import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from sonogpt.data.dataset import EncodedGenerateSample
from sonogpt.model.config import SonoGPTConfig
from sonogpt.model.gpt import SonoGPT
from sonogpt.training.checkpoint import (
    resolve_latest_pointer,
    semantic_checkpoint_sha256,
)
from sonogpt.training.config import TrainingConfig, apply_runtime_overrides
from sonogpt.training.progress import ProgressReporter
from sonogpt.training.reproducibility import set_reproducible_seed
from sonogpt.training.scheduler import WarmupCosineScheduler
from sonogpt.training.telemetry import NvidiaSmiMonitor
from sonogpt.training.trainer import DeterministicBatchStream, Trainer


def _encoded_samples(count: int = 8) -> tuple[EncodedGenerateSample, ...]:
    samples = []
    for index in range(count):
        token = 4 + index % 6
        input_ids = (1, 3, token, 10 + index % 4, 2)
        samples.append(
            EncodedGenerateSample(
                sample_id=f"sample-{index}",
                input_ids=input_ids,
                labels=(-100, -100, token, 10 + index % 4, 2),
                target_start=2,
            )
        )
    return tuple(samples)


def _training_config(*, max_steps: int = 4) -> TrainingConfig:
    return TrainingConfig(
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=max_steps,
        learning_rate=1e-2,
        min_learning_rate=1e-3,
        warmup_steps=1,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        evaluation_interval=2,
        evaluation_batch_size=4,
        checkpoint_interval=2,
        log_interval=1,
        amp_dtype="none",
        seed=123,
    )


def _trainer(
    config: TrainingConfig,
    *,
    run_identity: Mapping[str, str] | None = None,
) -> Trainer:
    set_reproducible_seed(config.seed)
    model = SonoGPT(
        SonoGPTConfig(
            vocab_size=16,
            max_seq_len=8,
            n_layers=1,
            n_heads=2,
            d_model=16,
            d_ff=32,
            dropout=0.2,
        )
    )
    samples = _encoded_samples()
    return Trainer(
        model,
        samples,
        samples[:4],
        pad_id=0,
        config=config,
        device="cpu",
        run_identity=run_identity or {"dataset": "test"},
    )


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, Mapping):
        assert isinstance(right, Mapping)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        assert isinstance(right, Sequence)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_warmup_cosine_scheduler_has_explicit_recoverable_state() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = WarmupCosineScheduler(
        optimizer,
        max_steps=4,
        warmup_steps=2,
        min_learning_rate=0.1,
    )

    assert scheduler.get_last_lr() == pytest.approx([0.5])
    scheduler.step()
    assert scheduler.get_last_lr() == pytest.approx([1.0])
    scheduler.step()
    assert scheduler.get_last_lr() == pytest.approx([1.0])
    state = scheduler.state_dict()
    scheduler.step()
    assert scheduler.get_last_lr() == pytest.approx([0.55])

    restored_optimizer = torch.optim.SGD([parameter], lr=1.0)
    restored = WarmupCosineScheduler(
        restored_optimizer,
        max_steps=4,
        warmup_steps=2,
        min_learning_rate=0.1,
    )
    restored.load_state_dict(state)
    assert restored.step_number == 2
    assert restored.get_last_lr() == pytest.approx([1.0])


def test_batch_stream_continues_exactly_after_state_restore() -> None:
    stream = DeterministicBatchStream(dataset_size=5, batch_size=3, seed=17)
    first_two = [stream.next_indices() for _ in range(2)]
    state = stream.state_dict()
    expected_next = [stream.next_indices() for _ in range(4)]

    restored = DeterministicBatchStream(
        dataset_size=5, batch_size=3, seed=17
    )
    restored.load_state_dict(state)

    assert len(first_two) == 2
    assert [restored.next_indices() for _ in range(4)] == expected_next


def test_checkpoint_resume_matches_uninterrupted_training_exactly(
    tmp_path: Path,
) -> None:
    config = _training_config(max_steps=4)
    uninterrupted = _trainer(config)
    for _ in range(4):
        uninterrupted.train_step()
    uninterrupted_path = tmp_path / "uninterrupted.pt"
    uninterrupted.save_training_checkpoint(uninterrupted_path)

    interrupted = _trainer(config)
    for _ in range(2):
        interrupted.train_step()
    checkpoint_path = tmp_path / "step_00000002.pt"
    interrupted.save_training_checkpoint(checkpoint_path)

    resumed = _trainer(config)
    resumed.load_training_checkpoint(checkpoint_path)
    for _ in range(2):
        resumed.train_step()
    resumed_path = tmp_path / "resumed.pt"
    resumed.save_training_checkpoint(resumed_path)

    assert resumed.state.to_dict() == uninterrupted.state.to_dict()
    assert (
        resumed.batch_stream.state_dict()
        == uninterrupted.batch_stream.state_dict()
    )
    assert (
        resumed.scheduler.state_dict()
        == uninterrupted.scheduler.state_dict()
    )
    _assert_nested_equal(
        resumed.model.state_dict(),
        uninterrupted.model.state_dict(),
    )
    _assert_nested_equal(
        resumed.optimizer.state_dict(),
        uninterrupted.optimizer.state_dict(),
    )
    assert semantic_checkpoint_sha256(
        resumed_path
    ) == semantic_checkpoint_sha256(uninterrupted_path)


def test_checkpoint_restores_python_numpy_and_torch_rng(tmp_path: Path) -> None:
    config = _training_config(max_steps=2)
    trainer = _trainer(config)
    trainer.train_step()
    checkpoint_path = tmp_path / "rng.pt"
    trainer.save_training_checkpoint(checkpoint_path)

    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(4)

    restored = _trainer(config)
    restored.load_training_checkpoint(checkpoint_path)

    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    torch.testing.assert_close(torch.rand(4), expected_torch, rtol=0, atol=0)


def test_fit_writes_verified_latest_pointer_and_validation(
    tmp_path: Path,
) -> None:
    trainer = _trainer(_training_config(max_steps=2))

    result = trainer.fit(checkpoint_directory=tmp_path)
    latest_path = resolve_latest_pointer(tmp_path)

    assert latest_path.name == "step_00000002.pt"
    assert result.state.global_step == 2
    assert result.final_validation.target_token_count > 0
    assert 0 <= result.final_validation.token_accuracy <= 1
    assert result.runtime.applied_optimizer_steps == 2
    assert result.runtime.example_count == 8
    assert result.runtime.target_tokens_per_second > 0


def test_checkpoint_rejects_different_run_identity(tmp_path: Path) -> None:
    config = _training_config(max_steps=2)
    checkpoint_path = tmp_path / "identity.pt"
    _trainer(config, run_identity={"dataset": "first"}).save_training_checkpoint(
        checkpoint_path
    )

    incompatible = _trainer(config, run_identity={"dataset": "second"})
    with pytest.raises(ValueError, match="run_identity"):
        incompatible.load_training_checkpoint(checkpoint_path)


def test_nvidia_smi_telemetry_row_parsing() -> None:
    sample = NvidiaSmiMonitor.parse_query_row(
        "72, 96, 3120, 84.5\n",
        elapsed_seconds=3.25,
    )

    assert sample.temperature_c == 72
    assert sample.utilization_percent == 96
    assert sample.memory_used_mib == 3120
    assert sample.power_w == 84.5
    assert sample.elapsed_seconds == 3.25


def test_progress_reporter_emits_heartbeats_eta_and_stall_warning() -> None:
    events: list[dict[str, object]] = []
    reporter = ProgressReporter(
        events.append,
        heartbeat_interval_seconds=0.01,
        stall_warning_seconds=0.03,
    )
    reporter.start()
    reporter.update(
        "training",
        completed_steps=0,
        total_steps=10,
        detail="test",
    )
    time.sleep(0.04)
    assert reporter.snapshot().status == "warning_possible_stall"

    reporter.update(
        "training",
        completed_steps=5,
        total_steps=10,
        detail="test",
    )
    snapshot = reporter.snapshot()
    assert snapshot.status == "running"
    assert snapshot.percent == 50
    assert snapshot.steps_per_second is not None
    assert snapshot.eta_seconds is not None

    reporter.close(status="completed")
    assert events[-1]["status"] == "completed"


def test_training_cli_overrides_preserve_effective_batch_size() -> None:
    effective = apply_runtime_overrides(
        TrainingConfig(),
        max_steps=100,
        micro_batch_size=8,
        gradient_accumulation_steps=4,
    )

    assert effective.max_steps == 100
    assert effective.warmup_steps == 100
    assert effective.micro_batch_size == 8
    assert effective.gradient_accumulation_steps == 4
    assert effective.effective_batch_size == 32
