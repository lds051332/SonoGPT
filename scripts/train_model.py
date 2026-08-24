"""Train the candidate SonoGPT model with exact checkpoint recovery."""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import torch

from sonogpt.data.dataset import EncodedGenerateSample, encode_generate_sample
from sonogpt.data.freeze import verify_freeze_record
from sonogpt.data.manifest import sha256_file, verify_manifest
from sonogpt.data.renderers import GeneratedSample
from sonogpt.model.config import SonoGPTConfig
from sonogpt.model.gpt import SonoGPT
from sonogpt.tokenizer.sentencepiece_bpe import SentencePieceBPETokenizer
from sonogpt.training.checkpoint import resolve_latest_pointer
from sonogpt.training.config import TrainingConfig, apply_runtime_overrides
from sonogpt.training.progress import ProgressReporter
from sonogpt.training.reproducibility import set_reproducible_seed
from sonogpt.training.telemetry import NvidiaSmiMonitor
from sonogpt.training.trainer import Trainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "artifacts" / "training" / "sonogpt_16m_m3"
)
DEFAULT_FREEZE_RECORD = (
    PROJECT_ROOT
    / "data"
    / "releases"
    / "synthetic_v1_5k_frozen_v1.freeze.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "synthetic_v1_5k_candidate_v2",
    )
    parser.add_argument(
        "--data-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "manifests"
        / "synthetic_v1_5k_candidate_v2.manifest.json",
    )
    parser.add_argument(
        "--freeze-record",
        type=Path,
        default=DEFAULT_FREEZE_RECORD,
    )
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "tokenizers"
        / "sonogpt_bpe_1807_candidate_v2"
        / "sonogpt_bpe.model",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "model"
        / "sonogpt_16m_candidate.json",
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "training"
        / "sonogpt_16m_m3.json",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a concrete device such as cuda:0",
    )
    parser.add_argument(
        "--resume",
        help="checkpoint path, or 'latest' in the output directory",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument(
        "--max-gpu-temperature",
        type=float,
        default=82.0,
        help="trigger the configured thermal action at this temperature",
    )
    parser.add_argument(
        "--thermal-resume-temperature",
        type=float,
        default=75.0,
    )
    parser.add_argument(
        "--thermal-action",
        choices=("cooldown", "abort"),
        default="cooldown",
    )
    parser.add_argument(
        "--max-cooldown-seconds",
        type=float,
        default=600.0,
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run two CPU-friendly steps on a small data view",
    )
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--stall-warning-seconds",
        type=float,
        default=60.0,
    )
    return parser.parse_args()


def _load_samples(path: Path) -> tuple[GeneratedSample, ...]:
    return tuple(
        GeneratedSample(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def _encode_samples(
    samples: tuple[GeneratedSample, ...],
    tokenizer: SentencePieceBPETokenizer,
    *,
    max_seq_len: int,
    phase: str,
    progress: ProgressReporter,
) -> tuple[EncodedGenerateSample, ...]:
    encoded = []
    total = len(samples)
    progress.update(
        phase,
        completed_steps=0,
        total_steps=total,
        detail="causal LM encoding",
    )
    for index, sample in enumerate(samples, start=1):
        encoded.append(
            encode_generate_sample(
                sample,
                tokenizer,
                max_seq_len=max_seq_len,
            )
        )
        if index % 250 == 0 or index == total:
            progress.update(
                phase,
                completed_steps=index,
                total_steps=total,
                detail="causal LM encoding",
            )
    return tuple(encoded)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def _effective_training_config(
    config: TrainingConfig, args: argparse.Namespace
) -> TrainingConfig:
    effective = config
    if args.smoke_test:
        effective = replace(
            config,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
            max_steps=2,
            warmup_steps=1,
            evaluation_interval=1,
            evaluation_batch_size=4,
            checkpoint_interval=1,
            log_interval=1,
        )
    return apply_runtime_overrides(
        effective,
        max_steps=None if args.smoke_test else args.max_steps,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run(
    args: argparse.Namespace,
    progress: ProgressReporter,
    console_emit: Callable[[dict[str, object]], None],
) -> None:
    if args.max_gpu_temperature <= 0:
        raise ValueError("--max-gpu-temperature must be positive")
    if not 0 < args.thermal_resume_temperature < args.max_gpu_temperature:
        raise ValueError(
            "thermal resume temperature must be positive and below the limit"
        )
    if args.max_cooldown_seconds <= 0:
        raise ValueError("--max-cooldown-seconds must be positive")
    if args.smoke_test and args.output_directory == DEFAULT_OUTPUT_DIRECTORY:
        args.output_directory = DEFAULT_OUTPUT_DIRECTORY.with_name(
            "sonogpt_16m_m3_smoke"
        )
    progress.update("environment_setup", detail="resolving device")
    device = _resolve_device("cpu" if args.smoke_test else args.device)

    progress.update("freeze_verification", detail=str(args.freeze_record))
    freeze_record = verify_freeze_record(
        args.freeze_record,
        project_root=PROJECT_ROOT,
    )
    frozen_artifacts = freeze_record["artifacts"]
    expected_paths = {
        "data directory": PROJECT_ROOT / frozen_artifacts["data_directory"],
        "data manifest": (
            PROJECT_ROOT / frozen_artifacts["data_manifest"]["path"]
        ),
        "tokenizer model": (
            PROJECT_ROOT / frozen_artifacts["tokenizer_model"]["path"]
        ),
    }
    requested_paths = {
        "data directory": args.data_directory,
        "data manifest": args.data_manifest,
        "tokenizer model": args.tokenizer_model,
    }
    for artifact_name, expected_path in expected_paths.items():
        if requested_paths[artifact_name].resolve() != expected_path.resolve():
            raise ValueError(
                f"{artifact_name} does not match the frozen release"
            )

    progress.update("manifest_verification", detail=str(args.data_manifest))
    verify_manifest(args.data_manifest, args.data_directory)
    manifest_payload = json.loads(
        args.data_manifest.read_text(encoding="utf-8")
    )

    progress.update("configuration", detail="loading model and training config")
    model_config = SonoGPTConfig.load(args.model_config)
    training_config = _effective_training_config(
        TrainingConfig.load(args.training_config), args
    )
    progress.update("tokenizer_loading", detail=str(args.tokenizer_model))
    tokenizer = SentencePieceBPETokenizer(args.tokenizer_model)
    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError("tokenizer and model vocab_size differ")

    progress.update("dataset_loading", detail="train and validation JSONL")
    train_samples = _load_samples(args.data_directory / "train.jsonl")
    validation_samples = _load_samples(
        args.data_directory / "validation.jsonl"
    )
    dataset_view = "full"
    if args.smoke_test:
        train_samples = train_samples[:32]
        validation_samples = validation_samples[:16]
        dataset_view = "smoke:train32:validation16"

    encoded_train = _encode_samples(
        train_samples,
        tokenizer,
        max_seq_len=model_config.max_seq_len,
        phase="encoding_train",
        progress=progress,
    )
    encoded_validation = _encode_samples(
        validation_samples,
        tokenizer,
        max_seq_len=model_config.max_seq_len,
        phase="encoding_validation",
        progress=progress,
    )

    run_identity = {
        "data_manifest_sha256": sha256_file(args.data_manifest),
        "dataset_view": dataset_view,
        "freeze_id": freeze_record["freeze_id"],
        "freeze_record_sha256": sha256_file(args.freeze_record),
        "freeze_payload_sha256": freeze_record["freeze_record_sha256"],
        "tokenizer_model_sha256": tokenizer.model_sha256,
        "torch_version": torch.__version__,
        "torch_cuda_version": str(torch.version.cuda),
        "train_file_sha256": manifest_payload["files"]["train"]["sha256"],
        "validation_file_sha256": manifest_payload["files"]["validation"][
            "sha256"
        ],
    }
    if device.type == "cuda":
        run_identity["cuda_device_name"] = torch.cuda.get_device_name(device)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    if (
        args.resume is None
        and (args.output_directory / "latest.json").exists()
    ):
        raise FileExistsError(
            "output directory already has a checkpoint; use --resume latest"
        )

    progress.update("model_initialization", detail=str(device))
    set_reproducible_seed(
        training_config.seed,
        deterministic=training_config.deterministic,
    )
    model = SonoGPT(model_config)
    trainer = Trainer(
        model,
        encoded_train,
        encoded_validation,
        pad_id=tokenizer.pad_id,
        config=training_config,
        device=device,
        run_identity=run_identity,
    )

    resume_path: Path | None = None
    if args.resume is not None:
        progress.update("checkpoint_restore", detail=str(args.resume))
        resume_path = (
            resolve_latest_pointer(args.output_directory)
            if args.resume == "latest"
            else Path(args.resume)
        )
        trainer.load_training_checkpoint(resume_path)

    progress.update("run_setup", detail=str(args.output_directory))
    run_manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "device": str(device),
        "effective_batch_size": training_config.effective_batch_size,
        "freeze_record": {
            "freeze_id": freeze_record["freeze_id"],
            "path": str(args.freeze_record),
            "sha256": sha256_file(args.freeze_record),
        },
        "model_config": model_config.to_dict(),
        "parameter_count": model.count_parameters(),
        "resume_checkpoint": str(resume_path) if resume_path else None,
        "run_identity": run_identity,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "thermal_control": {
            "action": args.thermal_action,
            "limit_c": args.max_gpu_temperature,
            "resume_c": args.thermal_resume_temperature,
            "max_cooldown_seconds": args.max_cooldown_seconds,
        },
        "train_sample_count": len(encoded_train),
        "training_config": training_config.to_dict(),
        "validation_sample_count": len(encoded_validation),
    }
    _write_json(args.output_directory / "run_manifest.json", run_manifest)

    metrics_path = args.output_directory / "metrics.jsonl"
    if resume_path is None:
        metrics_path.write_text("", encoding="utf-8")

    thermal_cooldown_count = 0
    thermal_cooldown_seconds = 0.0

    def record_event(event: dict[str, object]) -> None:
        serialized = json.dumps(
            event, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(serialized + "\n")
        console_emit(event)

    telemetry_monitor: NvidiaSmiMonitor | None = None

    def enforce_thermal_control(completed_step: int) -> None:
        nonlocal thermal_cooldown_count, thermal_cooldown_seconds
        if telemetry_monitor is None or not telemetry_monitor.samples:
            return
        current_temperature = telemetry_monitor.samples[-1].temperature_c
        if current_temperature < args.max_gpu_temperature:
            return
        if args.thermal_action == "abort":
            raise RuntimeError(
                "GPU temperature reached the configured safety limit"
            )

        thermal_cooldown_count += 1
        cooldown_started = time.perf_counter()
        while current_temperature > args.thermal_resume_temperature:
            cooldown_elapsed = time.perf_counter() - cooldown_started
            if cooldown_elapsed >= args.max_cooldown_seconds:
                raise RuntimeError("GPU thermal cooldown timed out")
            progress.update(
                "thermal_cooldown",
                completed_steps=completed_step,
                total_steps=training_config.max_steps,
                detail=(
                    f"{current_temperature:.0f}C; resume at "
                    f"{args.thermal_resume_temperature:.0f}C"
                ),
            )
            time.sleep(1.0)
            current_temperature = telemetry_monitor.samples[
                -1
            ].temperature_c
        thermal_cooldown_seconds += time.perf_counter() - cooldown_started
        progress.update(
            "training",
            completed_steps=completed_step,
            total_steps=training_config.max_steps,
            detail="thermal cooldown complete",
        )

    def update_training_progress(
        phase: str,
        completed: int | None,
        total: int | None,
        detail: str | None,
    ) -> None:
        progress.update(
            phase,
            completed_steps=completed,
            total_steps=total,
            detail=detail,
        )
        if phase == "training" and completed is not None:
            enforce_thermal_control(completed)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        telemetry_monitor = NvidiaSmiMonitor(
            device_index=device.index or torch.cuda.current_device()
        )
        telemetry_monitor.start()

    started = time.perf_counter()
    try:
        result = trainer.fit(
            checkpoint_directory=args.output_directory,
            event_callback=record_event,
            progress_callback=update_training_progress,
        )
    finally:
        if telemetry_monitor is not None:
            telemetry_monitor.stop()

    progress.update(
        "finalizing",
        completed_steps=result.state.global_step,
        total_steps=training_config.max_steps,
        detail="writing summary",
    )
    summary = {
        "device": str(device),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "final_validation": result.final_validation.to_dict(),
        "parameter_count": model.count_parameters(),
        "run_identity": run_identity,
        "runtime": result.runtime.to_dict(),
        "state": result.state.to_dict(),
        "thermal_control": {
            "cooldown_count": thermal_cooldown_count,
            "cooldown_seconds": thermal_cooldown_seconds,
            "limit_c": args.max_gpu_temperature,
            "resume_c": args.thermal_resume_temperature,
        },
    }
    if device.type == "cuda":
        summary["cuda_memory"] = {
            "max_allocated_mib": (
                torch.cuda.max_memory_allocated(device) / 1024**2
            ),
            "max_reserved_mib": (
                torch.cuda.max_memory_reserved(device) / 1024**2
            ),
        }
        summary["gpu_telemetry"] = (
            telemetry_monitor.summary() if telemetry_monitor else None
        )
    _write_json(args.output_directory / "summary.json", summary)
    console_emit({"event": "summary", **summary})


def main() -> None:
    args = parse_args()
    console_lock = threading.Lock()

    def console_emit(event: dict[str, object]) -> None:
        serialized = json.dumps(
            event,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        with console_lock:
            print(serialized, flush=True)

    progress = ProgressReporter(
        console_emit,
        heartbeat_interval_seconds=args.progress_interval_seconds,
        stall_warning_seconds=args.stall_warning_seconds,
    )
    progress.start()
    try:
        _run(args, progress, console_emit)
    except KeyboardInterrupt:
        progress.close(status="cancelled")
        raise
    except BaseException:
        progress.close(status="failed")
        raise
    else:
        progress.close(status="completed")


if __name__ == "__main__":
    main()
