from __future__ import annotations

import json
from pathlib import Path

from sonogpt.baselines.qc import run_qc
from sonogpt.baselines.rule_extract import extract_exam
from sonogpt.baselines.template_report import render_report
from sonogpt.data.renderers import TEMPLATE_FAMILIES, render_with_family
from sonogpt.data.semantic_generator import canonical_exam_json, sample_semantic_cases
from sonogpt.evaluation.extract_baseline import score_extract_examples
from sonogpt.evaluation.metrics import compare_fields, compare_dimensions
from sonogpt.evaluation.pipeline import EvaluationExample
from sonogpt.evaluation.qc_baseline import (
    EXPECTED_RULES,
    _inject_diagnostic_language,
    _inject_laterality_mismatch,
    _inject_negation_mislabeled,
    _inject_unit_mixed,
)
from sonogpt.schemas.domain import (
    EchogenicFocus,
    ThyroidExam,
    Vascularity,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "fixtures"
    / "single_nodule_v1.jsonl"
)


def _fixture_example() -> EvaluationExample:
    sample = json.loads(FIXTURE_PATH.read_text(encoding="utf-8").splitlines()[0])
    exam = ThyroidExam.model_validate(sample["structure"])
    return EvaluationExample(
        example_id=str(sample["sample_id"]),
        split="fixtures",
        semantic_case_id=str(sample["sample_id"]),
        input_text=canonical_exam_json(exam),
        reference_text=str(sample["report"]),
        exam=exam,
        template_family="fixture_report",
    )


def test_rule_extract_recovers_fixture_and_template_reports() -> None:
    for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        exam = ThyroidExam.model_validate(sample["structure"])
        extracted = extract_exam(str(sample["report"]))
        fields = compare_fields(extracted.exam, exam)
        assert extracted.parseable
        if fields.mentioned_count:
            assert fields.mentioned_accuracy == 1.0, sample["sample_id"]
        assert fields.hallucinated_count == 0, sample["sample_id"]
        if isinstance(exam.nodules[0].dimensions_mm, list):
            assert compare_dimensions(extracted.exam, exam).exact_match == 1

    for semantic_case in sample_semantic_cases(12, seed=903):
        for family in TEMPLATE_FAMILIES:
            extracted = extract_exam(render_with_family(semantic_case.exam, family))
            assert compare_fields(extracted.exam, semantic_case.exam).mentioned_accuracy == 1.0


def test_rule_extract_converts_cm_measurements_to_mm() -> None:
    exam = ThyroidExam.model_validate(
        {
            "nodules": [
                {
                    "location": {"side": "right", "segment": "middle"},
                    "dimensions_mm": [12, 8],
                    "composition": "solid",
                    "echogenicity": "hypoechoic",
                }
            ]
        }
    )
    extracted = extract_exam("甲状腺右叶中部见一枚实性低回声结节，大小约1.2×0.8cm。")
    assert extracted.exam.nodules[0].dimensions_mm == [12.0, 8.0]
    assert compare_dimensions(extracted.exam, exam).exact_match == 1


def test_extract_baseline_scores_fixtures_without_pooling() -> None:
    examples = tuple(
        EvaluationExample(
            example_id=str(sample["sample_id"]),
            split="fixtures",
            semantic_case_id=str(sample["sample_id"]),
            input_text="",
            reference_text=str(sample["report"]),
            exam=ThyroidExam.model_validate(sample["structure"]),
            template_family=None,
        )
        for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for sample in [json.loads(line)]
    )
    metrics = score_extract_examples(examples)
    assert metrics["split"] == "fixtures"
    assert metrics["sample_count"] == 50
    assert metrics["parseable_rate"] == 1.0
    assert metrics["fields"]["mentioned_accuracy"] == 1.0


def test_qc_passes_matching_template_and_structure() -> None:
    exam = ThyroidExam.model_validate(
        {
            "nodules": [
                {
                    "location": {"side": "left", "segment": "upper"},
                    "dimensions_mm": [10, 6],
                    "composition": "solid",
                    "echogenicity": "hypoechoic",
                    "shape": "wider_than_tall",
                    "margin": "smooth",
                    "echogenic_foci": "none",
                    "vascularity": "none",
                }
            ],
            "lymph_nodes": "no_suspicious",
        }
    )
    result = run_qc(render_report(exam), exam)
    assert result.passed
    assert result.error_count == 0


def test_qc_detects_injected_defects() -> None:
    example = _fixture_example()
    laterality = _inject_laterality_mismatch(example)
    diagnostic = _inject_diagnostic_language(example)
    units = _inject_unit_mixed(example)
    assert laterality is not None
    assert run_qc(laterality.report, laterality.exam).has(
        EXPECTED_RULES["laterality_mismatch"]
    )
    assert diagnostic is not None
    assert run_qc(diagnostic.report, diagnostic.exam).has(
        EXPECTED_RULES["diagnostic_language"]
    )
    assert units is not None
    assert run_qc(units.report, units.exam).has(EXPECTED_RULES["unit_mixed"])


def test_qc_flags_explicit_negative_labeled_as_not_mentioned() -> None:
    exam = ThyroidExam.model_validate(
        {
            "nodules": [
                {
                    "location": {"side": "right", "segment": "middle"},
                    "dimensions_mm": [8, 6],
                    "composition": "solid",
                    "echogenicity": "hypoechoic",
                    "shape": "wider_than_tall",
                    "margin": "smooth",
                    "echogenic_foci": "none",
                    "vascularity": "none",
                }
            ]
        }
    )
    example = EvaluationExample(
        example_id="neg",
        split="fixtures",
        semantic_case_id="neg",
        input_text=canonical_exam_json(exam),
        reference_text=render_report(exam),
        exam=exam,
        template_family=None,
    )
    defect = _inject_negation_mislabeled(example)
    assert defect is not None
    assert exam.nodules[0].vascularity == Vascularity.NONE
    assert defect.exam.nodules[0].vascularity == Vascularity.NOT_MENTIONED
    result = run_qc(defect.report, defect.exam)
    assert result.has(EXPECTED_RULES["negation_mislabeled"])
    assert exam.nodules[0].echogenic_foci == EchogenicFocus.NONE
