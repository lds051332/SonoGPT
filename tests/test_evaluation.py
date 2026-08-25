from __future__ import annotations

import json
from pathlib import Path

from sonogpt.baselines.template_report import render_report
from sonogpt.data.renderers import TEMPLATE_FAMILIES, render_with_family
from sonogpt.data.semantic_generator import sample_semantic_cases
from sonogpt.evaluation.challenge import ChallengeSample
from sonogpt.evaluation.metrics import (
    compare_fields,
    compare_split_metrics,
    is_stated_field,
    score_generated_report,
)
from sonogpt.evaluation.report_parser import parse_report
from sonogpt.schemas.domain import ThyroidExam

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "fixtures"
    / "single_nodule_v1.jsonl"
)
CHALLENGE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "challenges"
    / "simulated_human_challenge_v1.jsonl"
)


def _assert_stated_fields_recovered(exam: ThyroidExam, report: str) -> None:
    parsed = parse_report(report)
    assert parsed.parseable, report
    comparison = compare_fields(parsed.exam, exam)
    missed = [
        name
        for name, stats in comparison.per_field.items()
        if stats["mentioned"] and not stats["correct"]
    ]
    gold_dimensions = exam.nodules[0].dimensions_mm
    if isinstance(gold_dimensions, list):
        predicted = parsed.exam.nodules[0].dimensions_mm
        if predicted != gold_dimensions:
            missed.append("dimensions_mm")
    assert not missed, (missed, report, parsed.exam.model_dump(mode="json"))


def test_parser_recovers_stated_fields_from_fixture_templates() -> None:
    for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        exam = ThyroidExam.model_validate(sample["structure"])
        report = render_report(exam)
        assert report == sample["report"]
        _assert_stated_fields_recovered(exam, report)


def test_parser_recovers_stated_fields_from_all_template_families() -> None:
    for semantic_case in sample_semantic_cases(24, seed=812):
        for family in TEMPLATE_FAMILIES:
            report = render_with_family(semantic_case.exam, family)
            _assert_stated_fields_recovered(semantic_case.exam, report)


def test_parser_recovers_stated_fields_from_challenge_references() -> None:
    for line in CHALLENGE_PATH.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        sample = ChallengeSample.from_dict(json.loads(line))
        exam = ThyroidExam.model_validate_json(sample.input)
        _assert_stated_fields_recovered(exam, sample.reference_report)


def test_parser_does_not_require_unstated_challenge_fields() -> None:
    line = CHALLENGE_PATH.read_text(encoding="utf-8").splitlines()[9]
    sample = ChallengeSample.from_dict(json.loads(line))
    exam = ThyroidExam.model_validate_json(sample.input)
    parsed = parse_report(sample.reference_report)
    stated = [
        name
        for name in (
            "location_side",
            "composition",
            "echogenicity",
            "shape",
            "margin",
            "echogenic_foci",
            "vascularity",
            "lymph_nodes",
        )
        if is_stated_field(name, exam)
    ]
    assert stated == []
    assert parsed.parseable
    assert not is_stated_field("location_side", parsed.exam)
    assert not isinstance(parsed.exam.nodules[0].dimensions_mm, list)


def test_score_generated_report_rewards_exact_template_copy() -> None:
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
            ],
            "lymph_nodes": "no_suspicious",
        }
    )
    report = render_report(exam)
    scored, _parsed, fields, dimensions = score_generated_report(
        example_id="demo",
        split="test_seen_templates",
        semantic_case_id="sc_demo",
        generated_text=report,
        reference_text=report,
        gold_exam=exam,
        eos_finished=True,
        generation_error=None,
    )
    assert scored.exact_match
    assert scored.parseable
    assert scored.matched_template_family == "location_first_v2"
    assert fields.mentioned_accuracy == 1.0
    assert dimensions.exact_match == 1


def test_checkpoint_deltas_do_not_pool_challenge_with_templates() -> None:
    template_metrics = {
        "exact_match_rate": 0.9,
        "parseable_rate": 1.0,
        "fields": {"mentioned_accuracy": 0.8},
        "dimensions": {"exact_match_rate": 0.7},
        "template_exact_match": {"any_family_rate": 0.85},
        "teacher_forced": {"token_accuracy": 0.95, "loss": 0.1},
    }
    challenge_metrics = {
        "exact_match_rate": 0.1,
        "parseable_rate": 1.0,
        "fields": {"mentioned_accuracy": 0.6},
        "dimensions": {"exact_match_rate": 0.5},
        "template_exact_match": {"any_family_rate": 0.4},
        "teacher_forced": {"token_accuracy": 0.7, "loss": 0.4},
    }
    template_delta = compare_split_metrics(template_metrics, template_metrics)
    challenge_delta = compare_split_metrics(challenge_metrics, template_metrics)
    assert template_delta["exact_match_rate"] == 0.0
    assert challenge_delta["exact_match_rate"] == -0.8
    assert "pooled" not in template_delta
    assert "overall_average" not in challenge_delta
