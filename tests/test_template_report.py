from __future__ import annotations

import json
from pathlib import Path

from sonogpt.baselines.template_report import render_report
from sonogpt.schemas.domain import ThyroidExam


FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "fixtures"
    / "single_nodule_v1.jsonl"
)


def _load_fixtures() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_fixture_set_contains_50_unique_curated_samples() -> None:
    fixtures = _load_fixtures()
    sample_ids = [sample["sample_id"] for sample in fixtures]

    assert len(fixtures) == 50
    assert len(sample_ids) == len(set(sample_ids))


def test_all_fixture_structures_pass_schema_validation() -> None:
    for sample in _load_fixtures():
        ThyroidExam.model_validate(sample["structure"])


def test_template_renderer_matches_all_expected_reports() -> None:
    for sample in _load_fixtures():
        exam = ThyroidExam.model_validate(sample["structure"])

        assert render_report(exam) == sample["report"], sample["sample_id"]


def test_baseline_does_not_emit_diagnostic_conclusions() -> None:
    forbidden_terms = {"恶性", "癌", "良性", "确诊", "治疗"}

    for sample in _load_fixtures():
        report = str(sample["report"])
        assert not any(term in report for term in forbidden_terms), sample["sample_id"]
