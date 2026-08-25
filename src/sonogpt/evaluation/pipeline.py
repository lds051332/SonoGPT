"""Frozen generate-task evaluation over template splits and the challenge set."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import torch

from sonogpt.data.freeze import verify_freeze_record
from sonogpt.data.manifest import sha256_file
from sonogpt.data.renderers import GeneratedSample
from sonogpt.evaluation.challenge import ChallengeSample, verify_challenge_freeze
from sonogpt.evaluation.generation import (
    encode_generate_pair,
    generate_report,
    teacher_forced_metrics,
)
from sonogpt.evaluation.metrics import (
    DimensionComparison,
    FieldComparison,
    ScoredExample,
    aggregate_scores,
    compare_split_metrics,
    score_generated_report,
)
from sonogpt.model.config import MODEL_CONFIG_VERSION, SonoGPTConfig
from sonogpt.model.gpt import SonoGPT
from sonogpt.schemas.domain import ThyroidExam
from sonogpt.tokenizer.sentencepiece_bpe import SentencePieceBPETokenizer
from sonogpt.training.checkpoint import load_checkpoint, sha256_path

SEEN_SPLIT = "test_seen_templates"
HELDOUT_SPLIT = "test_heldout_templates"
CHALLENGE_SPLIT = "simulated_human_challenge"
ALL_SPLITS = (SEEN_SPLIT, HELDOUT_SPLIT, CHALLENGE_SPLIT)
FAILURE_EXAMPLE_LIMIT = 8


@dataclass(frozen=True)
class EvaluationExample:
    example_id: str
    split: str
    semantic_case_id: str
    input_text: str
    reference_text: str
    exam: ThyroidExam
    template_family: str | None


def _relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def model_config_from_checkpoint(payload: Mapping[str, object]) -> SonoGPTConfig:
    raw = dict(payload["model_config"])
    if raw.pop("config_version", None) != MODEL_CONFIG_VERSION:
        raise ValueError("unsupported model config version in checkpoint")
    return SonoGPTConfig(**raw)


def load_examples_from_generate_jsonl(
    path: Path, *, split: str, limit: int | None = None
) -> tuple[EvaluationExample, ...]:
    examples: list[EvaluationExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        sample = GeneratedSample(**json.loads(line))
        examples.append(
            EvaluationExample(
                example_id=sample.sample_id,
                split=split,
                semantic_case_id=sample.semantic_case_id,
                input_text=sample.input,
                reference_text=sample.target,
                exam=ThyroidExam.model_validate_json(sample.input),
                template_family=sample.template_family,
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    if not examples:
        raise ValueError(f"no generate samples found in {path}")
    return tuple(examples)


def load_examples_from_challenge_jsonl(
    path: Path, *, limit: int | None = None
) -> tuple[EvaluationExample, ...]:
    examples: list[EvaluationExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        sample = ChallengeSample.from_dict(json.loads(line))
        examples.append(
            EvaluationExample(
                example_id=sample.challenge_id,
                split=CHALLENGE_SPLIT,
                semantic_case_id=sample.semantic_case_id,
                input_text=sample.input,
                reference_text=sample.reference_report,
                exam=ThyroidExam.model_validate_json(sample.input),
                template_family=None,
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    if not examples:
        raise ValueError(f"no challenge samples found in {path}")
    return tuple(examples)


def load_model_for_evaluation(
    checkpoint_path: Path,
    tokenizer: SentencePieceBPETokenizer,
    *,
    device: torch.device,
    freeze_record: Mapping[str, object],
) -> tuple[SonoGPT, dict[str, object]]:
    payload = load_checkpoint(checkpoint_path)
    identity = payload.get("run_identity")
    if not isinstance(identity, dict):
        raise ValueError("checkpoint is missing run_identity")
    if identity.get("freeze_id") != freeze_record["freeze_id"]:
        raise ValueError("checkpoint freeze_id does not match the freeze record")
    if identity.get("freeze_payload_sha256") != freeze_record["freeze_record_sha256"]:
        raise ValueError("checkpoint freeze payload hash does not match")
    if identity.get("tokenizer_model_sha256") != tokenizer.model_sha256:
        raise ValueError("checkpoint tokenizer hash does not match")
    config = model_config_from_checkpoint(payload)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("tokenizer and checkpoint vocab_size differ")
    model = SonoGPT(config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def evaluate_examples(
    model: SonoGPT,
    tokenizer: SentencePieceBPETokenizer,
    examples: Sequence[EvaluationExample],
    *,
    device: torch.device,
    batch_size: int = 8,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, object]:
    if not examples:
        raise ValueError("evaluation split is empty")
    split = examples[0].split
    if any(example.split != split for example in examples):
        raise ValueError("cannot mix evaluation splits in one call")

    encoded = tuple(
        encode_generate_pair(
            example_id=example.example_id,
            input_text=example.input_text,
            target_text=example.reference_text,
            tokenizer=tokenizer,
            max_seq_len=model.config.max_seq_len,
        )
        for example in examples
    )
    forced = teacher_forced_metrics(
        model,
        encoded,
        pad_id=tokenizer.pad_id,
        device=device,
        batch_size=batch_size,
    )
    scored: list[ScoredExample] = []
    field_rows: list[FieldComparison] = []
    dimension_rows: list[DimensionComparison] = []
    for index, example in enumerate(examples, start=1):
        generated = generate_report(
            model, tokenizer, example.input_text, device=device
        )
        row, _parsed, fields, dimensions = score_generated_report(
            example_id=example.example_id,
            split=example.split,
            semantic_case_id=example.semantic_case_id,
            generated_text=generated.text,
            reference_text=example.reference_text,
            gold_exam=example.exam,
            eos_finished=generated.eos_finished,
            generation_error=generated.error,
        )
        scored.append(row)
        field_rows.append(fields)
        dimension_rows.append(dimensions)
        if progress_callback is not None:
            progress_callback(index, len(examples), split)

    metrics = aggregate_scores(
        tuple(scored),
        tuple(field_rows),
        tuple(dimension_rows),
        teacher_forced_loss=forced.loss,
        teacher_forced_token_accuracy=forced.token_accuracy,
        teacher_forced_token_count=forced.target_token_count,
    )
    failures = [
        {
            "example_id": row.example_id,
            "semantic_case_id": row.semantic_case_id,
            "exact_match": row.exact_match,
            "parseable": row.parseable,
            "generation_error": row.generation_error,
            "matched_template_family": row.matched_template_family,
            "reference_text": row.reference_text,
            "generated_text": row.generated_text,
        }
        for row in scored
        if not row.exact_match or row.generation_error is not None or not row.parseable
    ]
    metrics["split"] = split
    metrics["failure_examples"] = failures[:FAILURE_EXAMPLE_LIMIT]
    metrics["failure_example_count"] = len(failures)
    return metrics


def _require_existing(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} is missing: {path}. "
            "Place local artifacts under artifacts/ and data/processed/ (not stored in Git)."
        )
    return path


def _format_rate(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown_report(payload: Mapping[str, object]) -> str:
    lines = [
        "# SonoGPT generate-task evaluation",
        "",
        f"- Created (UTC): `{payload['created_at_utc']}`",
        f"- Freeze ID: `{payload['freeze_id']}`",
        f"- Challenge freeze ID: `{payload['challenge_freeze_id']}`",
        f"- Device: `{payload['device']}`",
        "",
        "Challenge metrics are reported separately and are **not** averaged with the template splits.",
        "This is a learning-demo evaluation, not a clinical validation.",
        "",
    ]
    for checkpoint in payload["checkpoints"]:
        identity = checkpoint["checkpoint"]
        lines.extend(
            [
                f"## {identity['name']}",
                "",
                f"- Path: `{identity['path']}`",
                f"- Global step: `{identity['global_step']}`",
                f"- File SHA-256: `{identity['file_sha256']}`",
                f"- Elapsed seconds: `{checkpoint['elapsed_seconds']}`",
                "",
            ]
        )
        for split_name in ALL_SPLITS:
            split = checkpoint["splits"][split_name]
            fields = split["fields"]
            dimensions = split["dimensions"]
            teacher = split["teacher_forced"]
            template = split["template_exact_match"]
            lines.extend(
                [
                    f"### {split_name}",
                    "",
                    f"- Samples: `{split['sample_count']}`",
                    f"- Exact match: `{_format_rate(split['exact_match_rate'])}`",
                    f"- Parseable: `{_format_rate(split['parseable_rate'])}`",
                    f"- EOS finished: `{_format_rate(split['eos_finished_rate'])}`",
                    f"- Teacher-forced loss: `{_format_rate(teacher['loss'])}`",
                    f"- Teacher-forced token accuracy: `{_format_rate(teacher['token_accuracy'])}`",
                    f"- Mentioned-field accuracy: `{_format_rate(fields['mentioned_accuracy'])}`",
                    f"- Dimension exact match: `{_format_rate(dimensions['exact_match_rate'])}`",
                    f"- Dimension MAE (mm): `{_format_rate(dimensions['mean_abs_error_mm'])}`",
                    f"- Any training-template exact match: `{_format_rate(template['any_family_rate'])}`",
                    "",
                ]
            )
    comparison = payload.get("comparison")
    if isinstance(comparison, Mapping) and comparison.get("splits"):
        lines.extend(["## Checkpoint comparison", ""])
        for split_name, deltas in comparison["splits"].items():
            lines.append(f"### {split_name}")
            lines.append("")
            for key, value in deltas.items():
                lines.append(f"- {key}: `{_format_rate(value)}`")
            lines.append("")
    return "\n".join(lines) + "\n"


def run_generate_evaluation(
    *,
    project_root: Path,
    freeze_record_path: Path,
    challenge_freeze_path: Path,
    checkpoint_paths: Sequence[Path],
    output_json: Path,
    output_markdown: Path,
    device_name: str = "auto",
    limit: int | None = None,
    batch_size: int = 8,
    splits: Iterable[str] = ALL_SPLITS,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, object]:
    selected_splits = tuple(splits)
    unknown = [name for name in selected_splits if name not in ALL_SPLITS]
    if unknown:
        raise ValueError(f"unknown evaluation splits: {unknown}")
    if not checkpoint_paths:
        raise ValueError("at least one checkpoint is required")

    freeze_record = verify_freeze_record(
        freeze_record_path, project_root=project_root
    )
    challenge_record = verify_challenge_freeze(
        challenge_freeze_path, project_root=project_root
    )
    artifacts = freeze_record["artifacts"]
    data_directory = project_root / artifacts["data_directory"]
    tokenizer_path = _require_existing(
        project_root / artifacts["tokenizer_model"]["path"],
        "frozen tokenizer",
    )
    seen_path = _require_existing(
        data_directory / "test_seen_templates.jsonl",
        "seen-template test set",
    )
    heldout_path = _require_existing(
        data_directory / "test_heldout_templates.jsonl",
        "held-out-template test set",
    )
    challenge_path = _require_existing(
        project_root / challenge_record["challenge_file"]["path"],
        "simulated challenge set",
    )

    examples_by_split = {
        SEEN_SPLIT: load_examples_from_generate_jsonl(
            seen_path, split=SEEN_SPLIT, limit=limit
        ),
        HELDOUT_SPLIT: load_examples_from_generate_jsonl(
            heldout_path, split=HELDOUT_SPLIT, limit=limit
        ),
        CHALLENGE_SPLIT: load_examples_from_challenge_jsonl(
            challenge_path, limit=limit
        ),
    }
    tokenizer = SentencePieceBPETokenizer(tokenizer_path)
    device = resolve_device(device_name)
    checkpoint_reports: list[dict[str, object]] = []

    for checkpoint_path in checkpoint_paths:
        checkpoint_path = _require_existing(checkpoint_path, "checkpoint")
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model, payload = load_model_for_evaluation(
            checkpoint_path,
            tokenizer,
            device=device,
            freeze_record=freeze_record,
        )
        split_metrics: dict[str, object] = {}
        for split_name in selected_splits:
            split_metrics[split_name] = evaluate_examples(
                model,
                tokenizer,
                examples_by_split[split_name],
                device=device,
                batch_size=batch_size,
                progress_callback=progress_callback,
            )
        elapsed = time.perf_counter() - started
        peak_memory = None
        if device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        training_state = payload.get("training_state", {})
        checkpoint_reports.append(
            {
                "checkpoint": {
                    "name": checkpoint_path.name,
                    "path": _relative(checkpoint_path, project_root),
                    "file_sha256": sha256_path(checkpoint_path),
                    "global_step": training_state.get("global_step"),
                    "best_validation_loss": training_state.get(
                        "best_validation_loss"
                    ),
                },
                "elapsed_seconds": elapsed,
                "peak_cuda_memory_allocated_mib": peak_memory,
                "splits": split_metrics,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    comparison = None
    if len(checkpoint_reports) >= 2:
        primary = checkpoint_reports[0]["splits"]
        reference = checkpoint_reports[1]["splits"]
        comparison = {
            "primary_checkpoint": checkpoint_reports[0]["checkpoint"]["name"],
            "reference_checkpoint": checkpoint_reports[1]["checkpoint"]["name"],
            "note": (
                "Deltas are primary minus reference. Challenge rows stay "
                "separate from template splits."
            ),
            "splits": {
                split_name: compare_split_metrics(
                    primary[split_name], reference[split_name]
                )
                for split_name in selected_splits
            },
        }

    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "evaluation_task": "generate",
        "evaluation_version": "1.0.0",
        "freeze_id": freeze_record["freeze_id"],
        "freeze_record_sha256": sha256_file(freeze_record_path),
        "challenge_freeze_id": challenge_record["freeze_id"],
        "challenge_freeze_sha256": sha256_file(challenge_freeze_path),
        "device": str(device),
        "tokenizer_model_sha256": tokenizer.model_sha256,
        "limit_per_split": limit,
        "notes": [
            "Challenge metrics are reported separately and are not averaged with template splits.",
            "Field accuracy uses an independent rule parser, not the model under test.",
            "This is a learning-demo evaluation, not a clinical validation.",
        ],
        "checkpoints": checkpoint_reports,
        "comparison": comparison,
    }
    _write_json(output_json, report)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(render_markdown_report(report), encoding="utf-8")
    return report
