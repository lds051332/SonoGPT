"""Evaluate the independent rule extractor on frozen generate reports."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Mapping

from sonogpt.baselines.rule_extract import RULE_EXTRACT_VERSION, extract_exam
from sonogpt.data.freeze import verify_freeze_record
from sonogpt.data.manifest import sha256_file
from sonogpt.data.semantic_generator import canonical_exam_json
from sonogpt.evaluation.challenge import verify_challenge_freeze
from sonogpt.evaluation.metrics import (
    DimensionComparison,
    FieldComparison,
    aggregate_scores,
    compare_dimensions,
    compare_fields,
)
from sonogpt.evaluation.pipeline import (
    ALL_SPLITS,
    CHALLENGE_SPLIT,
    HELDOUT_SPLIT,
    SEEN_SPLIT,
    EvaluationExample,
    load_examples_from_challenge_jsonl,
    load_examples_from_generate_jsonl,
)
from sonogpt.evaluation.metrics import ScoredExample
from sonogpt.schemas.domain import ThyroidExam

FIXTURE_SPLIT = "fixtures"
FAILURE_EXAMPLE_LIMIT = 8


def load_fixture_examples(path: Path) -> tuple[EvaluationExample, ...]:
    examples: list[EvaluationExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        exam = ThyroidExam.model_validate(sample["structure"])
        examples.append(
            EvaluationExample(
                example_id=str(sample["sample_id"]),
                split=FIXTURE_SPLIT,
                semantic_case_id=str(sample["sample_id"]),
                input_text=canonical_exam_json(exam),
                reference_text=str(sample["report"]),
                exam=exam,
                template_family="fixture_report",
            )
        )
    if not examples:
        raise ValueError(f"no fixtures found in {path}")
    return tuple(examples)


def score_extract_examples(
    examples: tuple[EvaluationExample, ...],
) -> dict[str, object]:
    scored: list[ScoredExample] = []
    field_rows: list[FieldComparison] = []
    dimension_rows: list[DimensionComparison] = []
    exact_structure = 0
    for example in examples:
        extracted = extract_exam(example.reference_text)
        fields = compare_fields(extracted.exam, example.exam)
        dimensions = compare_dimensions(extracted.exam, example.exam)
        structure_match = extracted.canonical_json == canonical_exam_json(example.exam)
        exact_structure += int(structure_match)
        scored.append(
            ScoredExample(
                example_id=example.example_id,
                split=example.split,
                semantic_case_id=example.semantic_case_id,
                generated_text=extracted.canonical_json,
                reference_text=example.reference_text,
                exact_match=structure_match,
                eos_finished=True,
                parseable=extracted.parseable,
                matched_template_family=None,
                generation_error=None,
                field_hits={
                    name: bool(stats["correct"])
                    for name, stats in fields.per_field.items()
                    if stats["mentioned"]
                },
                dimension_exact=None
                if dimensions.comparable_count == 0
                else bool(dimensions.exact_match),
                dimension_tolerant=None
                if dimensions.comparable_count == 0
                else bool(dimensions.tolerant_match),
            )
        )
        field_rows.append(fields)
        dimension_rows.append(dimensions)

    metrics = aggregate_scores(
        tuple(scored),
        tuple(field_rows),
        tuple(dimension_rows),
        teacher_forced_loss=None,
        teacher_forced_token_accuracy=None,
        teacher_forced_token_count=0,
    )
    metrics["split"] = examples[0].split if examples else None
    metrics["exact_structure_match_rate"] = (
        None if not examples else exact_structure / len(examples)
    )
    failures = [
        {
            "example_id": row.example_id,
            "semantic_case_id": row.semantic_case_id,
            "parseable": row.parseable,
            "exact_structure_match": row.exact_match,
            "reference_text": row.reference_text,
            "extracted_json": row.generated_text,
        }
        for row in scored
        if not row.exact_match or not row.parseable
    ]
    metrics["failure_examples"] = failures[:FAILURE_EXAMPLE_LIMIT]
    metrics["failure_example_count"] = len(failures)
    return metrics


def _format_rate(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_extract_markdown(payload: Mapping[str, object]) -> str:
    lines = [
        "# SonoGPT 规则抽取基线评估",
        "",
        f"- Created (UTC): `{payload['created_at_utc']}`",
        f"- Extractor: `{payload['extractor']}` `{payload['extractor_version']}`",
        f"- Freeze ID: `{payload['freeze_id']}`",
        f"- Challenge freeze ID: `{payload['challenge_freeze_id']}`",
        "",
        "这是 **extract** 方向（报告 → 结构）的规则基线，不使用神经网络。",
        "已见模板、留出模板、挑战集和夹具分开报告，没有混合平均。",
        "本结果用于学习 Demo 对照，不是临床验证。",
        "",
    ]
    for split_name, split in payload["splits"].items():
        fields = split["fields"]
        dimensions = split["dimensions"]
        lines.extend(
            [
                f"## {split_name}",
                "",
                f"- Samples: `{split['sample_count']}`",
                f"- Parseable: `{_format_rate(split['parseable_rate'])}`",
                f"- Exact structure match: `{_format_rate(split['exact_structure_match_rate'])}`",
                f"- Mentioned-field accuracy: `{_format_rate(fields['mentioned_accuracy'])}`",
                f"- Hallucination rate: `{_format_rate(fields['hallucination_rate'])}`",
                f"- Dimension exact match: `{_format_rate(dimensions['exact_match_rate'])}`",
                f"- Dimension MAE (mm): `{_format_rate(dimensions['mean_abs_error_mm'])}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def run_extract_baseline_evaluation(
    *,
    project_root: Path,
    freeze_record_path: Path,
    challenge_freeze_path: Path,
    fixture_path: Path,
    output_json: Path,
    output_markdown: Path,
    limit: int | None = None,
    splits: Iterable[str] = (*ALL_SPLITS, FIXTURE_SPLIT),
) -> dict[str, object]:
    selected = tuple(splits)
    freeze_record = verify_freeze_record(
        freeze_record_path, project_root=project_root
    )
    challenge_record = verify_challenge_freeze(
        challenge_freeze_path, project_root=project_root
    )
    artifacts = freeze_record["artifacts"]
    data_directory = project_root / artifacts["data_directory"]
    examples = {
        FIXTURE_SPLIT: load_fixture_examples(fixture_path),
        SEEN_SPLIT: load_examples_from_generate_jsonl(
            data_directory / "test_seen_templates.jsonl",
            split=SEEN_SPLIT,
            limit=limit,
        ),
        HELDOUT_SPLIT: load_examples_from_generate_jsonl(
            data_directory / "test_heldout_templates.jsonl",
            split=HELDOUT_SPLIT,
            limit=limit,
        ),
        CHALLENGE_SPLIT: load_examples_from_challenge_jsonl(
            project_root / challenge_record["challenge_file"]["path"],
            limit=limit,
        ),
    }
    split_metrics = {
        name: score_extract_examples(examples[name]) for name in selected
    }
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "evaluation_task": "extract",
        "evaluation_version": "1.0.0",
        "extractor": "rule_extract",
        "extractor_version": RULE_EXTRACT_VERSION,
        "freeze_id": freeze_record["freeze_id"],
        "freeze_record_sha256": sha256_file(freeze_record_path),
        "challenge_freeze_id": challenge_record["freeze_id"],
        "challenge_freeze_sha256": sha256_file(challenge_freeze_path),
        "limit_per_split": limit,
        "notes": [
            "Rule extraction is independent of SonoGPT weights.",
            "Exact structure match treats unknown and not_mentioned as different.",
            "Challenge metrics are reported separately.",
            "This is a learning-demo evaluation, not a clinical validation.",
        ],
        "splits": split_metrics,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_markdown.write_text(render_extract_markdown(report), encoding="utf-8")
    return report
