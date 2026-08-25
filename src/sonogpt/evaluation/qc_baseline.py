"""Evaluate versioned QC rules on clean pairs and injected defects."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from sonogpt.baselines.qc import QC_RULES_VERSION, QcResult, run_qc
from sonogpt.baselines.template_report import render_report
from sonogpt.data.freeze import verify_freeze_record
from sonogpt.data.manifest import sha256_file
from sonogpt.evaluation.challenge import verify_challenge_freeze
from sonogpt.evaluation.extract_baseline import load_fixture_examples
from sonogpt.evaluation.pipeline import (
    CHALLENGE_SPLIT,
    HELDOUT_SPLIT,
    SEEN_SPLIT,
    EvaluationExample,
    load_examples_from_challenge_jsonl,
    load_examples_from_generate_jsonl,
)
from sonogpt.schemas.domain import ThyroidExam, Vascularity

EXPECTED_RULES = {
    "missing_composition": "QC.TEXT_OMITS_STATED_COMPOSITION",
    "missing_echogenicity": "QC.TEXT_OMITS_STATED_ECHOGENICITY",
    "missing_shape": "QC.TEXT_OMITS_STATED_SHAPE",
    "missing_margin": "QC.TEXT_OMITS_STATED_MARGIN",
    "missing_foci": "QC.TEXT_OMITS_STATED_FOCI",
    "dimension_mismatch": "QC.TEXT_DIMENSION_MISMATCH",
    "laterality_mismatch": "QC.TEXT_LOCATION_MISMATCH",
    "unit_mixed": "QC.UNIT_MIXED_MM_CM",
    "text_contradiction": "QC.TEXT_FIELD_CONTRADICTION",
    "negation_mislabeled": "QC.STRUCTURE_NEGATION_MISLABEL",
    "diagnostic_language": "QC.DIAGNOSTIC_LANGUAGE",
}


@dataclass(frozen=True)
class DefectCase:
    defect_id: str
    example_id: str
    report: str
    exam: ThyroidExam
    expected_rule: str


def _strip_first(report: str, phrases: tuple[str, ...]) -> str | None:
    for phrase in phrases:
        if phrase in report:
            return report.replace(phrase, "", 1)
    return None


def _inject_missing_composition(example: EvaluationExample) -> DefectCase | None:
    mutated = _strip_first(
        example.reference_text,
        ("囊实性", "海绵状", "海绵样", "囊性", "液性", "实性"),
    )
    if mutated is None:
        return None
    return DefectCase(
        "missing_composition",
        example.example_id,
        mutated,
        example.exam,
        EXPECTED_RULES["missing_composition"],
    )


def _inject_missing_echogenicity(example: EvaluationExample) -> DefectCase | None:
    mutated = _strip_first(
        example.reference_text,
        ("极低回声", "低回声", "高回声", "等回声", "无回声"),
    )
    if mutated is None:
        return None
    return DefectCase(
        "missing_echogenicity",
        example.example_id,
        mutated,
        example.exam,
        EXPECTED_RULES["missing_echogenicity"],
    )


def _inject_missing_shape(example: EvaluationExample) -> DefectCase | None:
    mutated = _strip_first(
        example.reference_text,
        ("呈非高于宽形", "呈高于宽形"),
    )
    if mutated is None:
        return None
    return DefectCase(
        "missing_shape",
        example.example_id,
        mutated,
        example.exam,
        EXPECTED_RULES["missing_shape"],
    )


def _inject_missing_margin(example: EvaluationExample) -> DefectCase | None:
    mutated = _strip_first(
        example.reference_text,
        ("边缘光滑", "边界欠清", "边缘分叶或不规则", "可见甲状腺外侵犯征象"),
    )
    if mutated is None:
        return None
    return DefectCase(
        "missing_margin",
        example.example_id,
        mutated,
        example.exam,
        EXPECTED_RULES["missing_margin"],
    )


def _inject_missing_foci(example: EvaluationExample) -> DefectCase | None:
    mutated = _strip_first(
        example.reference_text,
        (
            "内未见明显强回声",
            "内见点状强回声",
            "内见彗尾征",
            "内见粗大钙化",
            "周边见环形钙化",
        ),
    )
    if mutated is None:
        return None
    return DefectCase(
        "missing_foci",
        example.example_id,
        mutated,
        example.exam,
        EXPECTED_RULES["missing_foci"],
    )


def _inject_dimension_mismatch(example: EvaluationExample) -> DefectCase | None:
    if not isinstance(example.exam.nodules[0].dimensions_mm, list):
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", example.reference_text)
    if match is None:
        return None
    original = match.group(1)
    replacement = str(float(original) + 5)
    mutated = example.reference_text.replace(original, replacement, 1)
    if mutated == example.reference_text:
        return None
    return DefectCase(
        "dimension_mismatch",
        example.example_id,
        mutated,
        example.exam,
        EXPECTED_RULES["dimension_mismatch"],
    )


def _inject_laterality_mismatch(example: EvaluationExample) -> DefectCase | None:
    report = example.reference_text
    if "左叶" in report and "右叶" not in report:
        mutated = report.replace("左叶", "右叶")
    elif "右叶" in report and "左叶" not in report:
        mutated = report.replace("右叶", "左叶")
    else:
        return None
    return DefectCase(
        "laterality_mismatch",
        example.example_id,
        mutated,
        example.exam,
        EXPECTED_RULES["laterality_mismatch"],
    )


def _inject_unit_mixed(example: EvaluationExample) -> DefectCase | None:
    if "mm" not in example.reference_text:
        return None
    mutated = example.reference_text.replace("mm", "mm（约1.0cm）", 1)
    return DefectCase(
        "unit_mixed",
        example.example_id,
        mutated,
        example.exam,
        EXPECTED_RULES["unit_mixed"],
    )


def _inject_text_contradiction(example: EvaluationExample) -> DefectCase | None:
    if "左叶" in example.reference_text:
        mutated = example.reference_text + "同时可见右叶结节。"
    elif "右叶" in example.reference_text:
        mutated = example.reference_text + "同时可见左叶结节。"
    else:
        return None
    return DefectCase(
        "text_contradiction",
        example.example_id,
        mutated,
        example.exam,
        EXPECTED_RULES["text_contradiction"],
    )


def _inject_negation_mislabeled(example: EvaluationExample) -> DefectCase | None:
    if example.exam.nodules[0].vascularity != Vascularity.NONE:
        return None
    if "未见明显血流" not in example.reference_text:
        return None
    nodule = example.exam.nodules[0]
    mutated_exam = example.exam.model_copy(
        update={
            "nodules": [
                nodule.model_copy(update={"vascularity": Vascularity.NOT_MENTIONED})
            ]
        }
    )
    return DefectCase(
        "negation_mislabeled",
        example.example_id,
        example.reference_text,
        mutated_exam,
        EXPECTED_RULES["negation_mislabeled"],
    )


def _inject_diagnostic_language(example: EvaluationExample) -> DefectCase | None:
    return DefectCase(
        "diagnostic_language",
        example.example_id,
        example.reference_text + "考虑恶性可能。",
        example.exam,
        EXPECTED_RULES["diagnostic_language"],
    )


INJECTORS: dict[str, Callable[[EvaluationExample], DefectCase | None]] = {
    "missing_composition": _inject_missing_composition,
    "missing_echogenicity": _inject_missing_echogenicity,
    "missing_shape": _inject_missing_shape,
    "missing_margin": _inject_missing_margin,
    "missing_foci": _inject_missing_foci,
    "dimension_mismatch": _inject_dimension_mismatch,
    "laterality_mismatch": _inject_laterality_mismatch,
    "unit_mixed": _inject_unit_mixed,
    "text_contradiction": _inject_text_contradiction,
    "negation_mislabeled": _inject_negation_mislabeled,
    "diagnostic_language": _inject_diagnostic_language,
}


def build_defect_cases(
    examples: tuple[EvaluationExample, ...],
    *,
    per_type_limit: int = 20,
) -> tuple[DefectCase, ...]:
    selected: list[DefectCase] = []
    for defect_id, injector in INJECTORS.items():
        count = 0
        for example in examples:
            case = injector(example)
            if case is None:
                continue
            selected.append(case)
            count += 1
            if count >= per_type_limit:
                break
        if count == 0:
            raise ValueError(f"could not inject defect type {defect_id}")
    return tuple(selected)


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def evaluate_clean_pairs(
    examples: tuple[EvaluationExample, ...],
) -> dict[str, object]:
    passed = 0
    error_count = 0
    warning_count = 0
    false_error_examples: list[dict[str, object]] = []
    for example in examples:
        result = run_qc(example.reference_text, example.exam)
        passed += int(result.passed)
        error_count += result.error_count
        warning_count += result.warning_count
        if not result.passed:
            false_error_examples.append(
                {
                    "example_id": example.example_id,
                    "split": example.split,
                    "findings": result.to_dict()["findings"],
                }
            )
    return {
        "sample_count": len(examples),
        "pass_rate": _rate(passed, len(examples)),
        "mean_errors": error_count / len(examples) if examples else None,
        "mean_warnings": warning_count / len(examples) if examples else None,
        "false_error_count": len(false_error_examples),
        "false_error_examples": false_error_examples[:8],
    }


def evaluate_injected_defects(cases: tuple[DefectCase, ...]) -> dict[str, object]:
    per_type: dict[str, dict[str, int]] = {}
    for case in cases:
        stats = per_type.setdefault(
            case.defect_id,
            {"support": 0, "detected": 0, "expected_rule_hits": 0},
        )
        result = run_qc(case.report, case.exam)
        stats["support"] += 1
        stats["detected"] += int(result.has(case.expected_rule))
        stats["expected_rule_hits"] += int(result.has(case.expected_rule))
    recall = {
        defect_id: {
            "support": stats["support"],
            "recall": _rate(stats["detected"], stats["support"]),
            "expected_rule": EXPECTED_RULES[defect_id],
        }
        for defect_id, stats in per_type.items()
    }
    detected = sum(stats["detected"] for stats in per_type.values())
    return {
        "case_count": len(cases),
        "macro_recall": sum(
            item["recall"] or 0.0 for item in recall.values()
        )
        / len(recall),
        "micro_recall": _rate(detected, len(cases)),
        "per_defect": recall,
    }


def render_qc_markdown(payload: MappingPayload) -> str:
    lines = [
        "# SonoGPT 质控规则基线评估",
        "",
        f"- Created (UTC): `{payload['created_at_utc']}`",
        f"- Rules version: `{payload['rules_version']}`",
        f"- Freeze ID: `{payload['freeze_id']}`",
        "",
        "质控规则检查报告与可选结构的一致性，不输出诊断结论。",
        "干净样本上的 error 视为假阳性；注入缺陷上的 recall 按缺陷类型分开报告。",
        "",
        "## 干净样本（报告 + 金标准结构）",
        "",
    ]
    for split_name, split in payload["clean_splits"].items():
        lines.extend(
            [
                f"### {split_name}",
                "",
                f"- Samples: `{split['sample_count']}`",
                f"- Pass rate (no errors): `{_format(split['pass_rate'])}`",
                f"- Mean errors: `{_format(split['mean_errors'])}`",
                f"- Mean warnings: `{_format(split['mean_warnings'])}`",
                f"- False-error count: `{split['false_error_count']}`",
                "",
            ]
        )
    injected = payload["injected_defects"]
    lines.extend(
        [
            "## 注入缺陷检测",
            "",
            f"- Cases: `{injected['case_count']}`",
            f"- Micro recall: `{_format(injected['micro_recall'])}`",
            f"- Macro recall: `{_format(injected['macro_recall'])}`",
            "",
        ]
    )
    for defect_id, stats in injected["per_defect"].items():
        lines.append(
            f"- `{defect_id}` ({stats['expected_rule']}): "
            f"recall `{_format(stats['recall'])}` / support `{stats['support']}`"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _format(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


# Avoid importing Mapping only for the type hint in render; use dict.
MappingPayload = dict[str, object]


def run_qc_baseline_evaluation(
    *,
    project_root: Path,
    freeze_record_path: Path,
    challenge_freeze_path: Path,
    fixture_path: Path,
    output_json: Path,
    output_markdown: Path,
    limit: int | None = None,
    defect_limit: int = 20,
) -> dict[str, object]:
    freeze_record = verify_freeze_record(
        freeze_record_path, project_root=project_root
    )
    challenge_record = verify_challenge_freeze(
        challenge_freeze_path, project_root=project_root
    )
    artifacts = freeze_record["artifacts"]
    data_directory = project_root / artifacts["data_directory"]
    fixtures = load_fixture_examples(fixture_path)
    seen = load_examples_from_generate_jsonl(
        data_directory / "test_seen_templates.jsonl",
        split=SEEN_SPLIT,
        limit=limit,
    )
    heldout = load_examples_from_generate_jsonl(
        data_directory / "test_heldout_templates.jsonl",
        split=HELDOUT_SPLIT,
        limit=limit,
    )
    challenge = load_examples_from_challenge_jsonl(
        project_root / challenge_record["challenge_file"]["path"],
        limit=limit,
    )
    clean_splits = {
        "fixtures": evaluate_clean_pairs(fixtures),
        SEEN_SPLIT: evaluate_clean_pairs(seen),
        HELDOUT_SPLIT: evaluate_clean_pairs(heldout),
        CHALLENGE_SPLIT: evaluate_clean_pairs(challenge),
    }
    defect_source = fixtures + tuple(
        example
        for example in seen
        if example.template_family == "location_first_v2"
    )[:50]
    injected = evaluate_injected_defects(
        build_defect_cases(defect_source, per_type_limit=defect_limit)
    )
    report: dict[str, object] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "evaluation_task": "qc",
        "evaluation_version": "1.0.0",
        "rules_version": QC_RULES_VERSION,
        "freeze_id": freeze_record["freeze_id"],
        "freeze_record_sha256": sha256_file(freeze_record_path),
        "challenge_freeze_id": challenge_record["freeze_id"],
        "limit_per_split": limit,
        "notes": [
            "QC rules are engineering checks, not diagnostic criteria.",
            "Clean-split errors are false positives against gold structure.",
            "Injected defects measure recall of the expected rule ID.",
            "Challenge metrics are reported separately.",
        ],
        "clean_splits": clean_splits,
        "injected_defects": injected,
        "smoke_template_roundtrip_passed": run_qc(
            render_report(fixtures[0].exam), fixtures[0].exam
        ).passed,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_markdown.write_text(render_qc_markdown(report), encoding="utf-8")
    return report
