from __future__ import annotations

import json

import pytest

from sonogpt.baselines.template_report import render_report
from sonogpt.data.renderers import (
    BASELINE_TEMPLATE_FAMILY,
    TEMPLATE_FAMILIES,
    render_case,
    render_with_family,
)
from sonogpt.data.semantic_generator import SemanticCase
from sonogpt.schemas.domain import ThyroidExam


@pytest.fixture
def rich_exam() -> ThyroidExam:
    return ThyroidExam.model_validate(
        {
            "nodules": [
                {
                    "location": {"side": "right", "segment": "middle"},
                    "dimensions_mm": [12, 8, 6],
                    "composition": "solid",
                    "echogenicity": "hypoechoic",
                    "shape": "wider_than_tall",
                    "margin": "smooth",
                    "echogenic_foci": "punctate_echogenic_foci",
                    "vascularity": "internal",
                }
            ],
            "lymph_nodes": "no_suspicious",
        }
    )


def test_four_template_families_use_distinct_word_orders(
    rich_exam: ThyroidExam,
) -> None:
    reports = {
        family: render_with_family(rich_exam, family)
        for family in TEMPLATE_FAMILIES
    }

    assert len(TEMPLATE_FAMILIES) >= 4
    assert len(set(reports.values())) == len(TEMPLATE_FAMILIES)
    assert reports[BASELINE_TEMPLATE_FAMILY] == render_report(rich_exam)


def test_every_family_preserves_all_explicit_observations(
    rich_exam: ThyroidExam,
) -> None:
    required_fragments = {
        "甲状腺右叶中部",
        "12×8×6mm",
        "实性低回声结节",
        "非高于宽形",
        "边缘光滑",
        "点状强回声",
        "内部可见血流信号",
        "颈部未见明显可疑淋巴结",
    }
    forbidden_terms = {"恶性", "癌", "良性", "确诊", "治疗"}

    for family in TEMPLATE_FAMILIES:
        report = render_with_family(rich_exam, family)
        assert all(fragment in report for fragment in required_fragments), family
        assert not any(term in report for term in forbidden_terms), family


def test_rendered_sample_is_stable_generate_only_record(
    rich_exam: ThyroidExam,
) -> None:
    semantic_case = SemanticCase.from_exam(rich_exam)
    first = render_case(semantic_case, "descriptor_first_v2", seed=42)
    second = render_case(semantic_case, "descriptor_first_v2", seed=42)

    assert first == second
    assert first.task == "generate"
    assert first.semantic_case_id == semantic_case.semantic_case_id
    assert json.loads(first.input) == rich_exam.model_dump(mode="json")


def test_unknown_template_family_is_rejected(rich_exam: ThyroidExam) -> None:
    with pytest.raises(ValueError, match="unknown template family"):
        render_with_family(rich_exam, "not-a-family")


def test_v2_templates_preserve_smooth_margin_and_absent_blood_flow() -> None:
    exam = ThyroidExam.model_validate(
        {
            "nodules": [
                {
                    "location": {"side": "left", "segment": "upper"},
                    "dimensions_mm": [8, 6],
                    "composition": "solid",
                    "echogenicity": "hypoechoic",
                    "margin": "smooth",
                    "vascularity": "none",
                }
            ]
        }
    )

    for family in TEMPLATE_FAMILIES:
        report = render_with_family(exam, family)
        assert "边缘光滑" in report
        assert "结节内及周边未见明显血流信号" in report
