"""Training, checkpointing, diagnostics, and reproducibility helpers."""

from sonogpt.training.checkpoint import (
    CHECKPOINT_VERSION,
    load_checkpoint,
    resolve_latest_pointer,
    save_checkpoint,
    semantic_checkpoint_sha256,
)
from sonogpt.training.config import TrainingConfig, apply_runtime_overrides
from sonogpt.training.overfit import (
    OverfitResult,
    evaluate_target_metrics,
    run_overfit,
)
from sonogpt.training.progress import ProgressReporter, ProgressSnapshot
from sonogpt.training.reproducibility import set_reproducible_seed
from sonogpt.training.scheduler import WarmupCosineScheduler
from sonogpt.training.telemetry import GpuTelemetrySample, NvidiaSmiMonitor
from sonogpt.training.trainer import (
    DeterministicBatchStream,
    EvaluationMetrics,
    Trainer,
    TrainingResult,
    TrainingRuntimeMetrics,
    TrainingState,
    TrainingStepMetrics,
    build_adamw_optimizer,
)

__all__ = [
    "CHECKPOINT_VERSION",
    "DeterministicBatchStream",
    "EvaluationMetrics",
    "GpuTelemetrySample",
    "NvidiaSmiMonitor",
    "OverfitResult",
    "ProgressReporter",
    "ProgressSnapshot",
    "Trainer",
    "TrainingConfig",
    "TrainingResult",
    "TrainingRuntimeMetrics",
    "TrainingState",
    "TrainingStepMetrics",
    "WarmupCosineScheduler",
    "apply_runtime_overrides",
    "build_adamw_optimizer",
    "evaluate_target_metrics",
    "load_checkpoint",
    "resolve_latest_pointer",
    "run_overfit",
    "save_checkpoint",
    "semantic_checkpoint_sha256",
    "set_reproducible_seed",
]
