from __future__ import annotations

import pytest

from sonogpt.data.constraints import synthetic_constraint_violations
from sonogpt.data.semantic_generator import sample_semantic_cases
from sonogpt.schemas.domain import ThyroidExam


def _exam(**nodule_overrides: object) -> ThyroidExam:
    nodule: dict[str, object] = {
        "location": {"side": "right", "segment": "middle"},
        "dimensions_mm": [8, 6],
        "composition": "solid",
        "echogenicity": "hypoechoic",
        "shape": "wider_than_tall",
        "margin": "smooth",
        "echogenic_foci": "none",
        "vascularity": "none",
    }
    nodule.update(nodule_overrides)
    return ThyroidExam.model_validate({"nodules": [nodule]})


def test_all_sampled_cases_satisfy_synthetic_constraints() -> None:
    for semantic_case in sample_semantic_cases(5000, seed=20260824):
        assert synthetic_constraint_violations(semantic_case.exam) == ()


@pytest.mark.parametrize(
    ("overrides", "expected_rule"),
    (
        (
            {"composition": "solid", "echogenicity": "anechoic"},
            "composition.non_cystic_anechoic",
        ),
        (
            {"composition": "cystic", "echogenicity": "hypoechoic"},
            "composition.cystic_echogenicity",
        ),
        (
            {
                "composition": "not_mentioned",
                "echogenicity": "anechoic",
                "vascularity": "internal",
            },
            "echogenicity.anechoic_vascularity",
        ),
        (
            {
                "composition": "spongiform",
                "echogenicity": "anechoic",
                "vascularity": "internal",
            },
            "composition.spongiform_echogenicity",
        ),
        (
            {"dimensions_mm": "not_applicable"},
            "dimensions.not_applicable",
        ),
        (
            {"dimensions_mm": [5, 2.9]},
            "dimensions.below_synthetic_minimum",
        ),
        (
            {"echogenic_foci": "not_applicable"},
            "echogenic_foci.not_applicable",
        ),
        (
            {"dimensions_mm": [39, 2]},
            "dimensions.extreme_ratio",
        ),
        (
            {"dimensions_mm": [8, 9], "shape": "wider_than_tall"},
            "shape.wider_than_tall_mismatch",
        ),
        (
            {"dimensions_mm": [9, 8], "shape": "taller_than_wide"},
            "shape.taller_than_wide_mismatch",
        ),
    ),
)
def test_schema_valid_but_low_quality_combinations_are_rejected_for_synthesis(
    overrides: dict[str, object], expected_rule: str
) -> None:
    exam = _exam(**overrides)

    assert expected_rule in synthetic_constraint_violations(exam)


def test_plausible_cystic_case_passes_synthetic_constraints() -> None:
    exam = _exam(
        composition="cystic",
        echogenicity="anechoic",
        shape="wider_than_tall",
        margin="smooth",
        echogenic_foci="comet_tail",
        vascularity="peripheral",
    )

    assert synthetic_constraint_violations(exam) == ()
