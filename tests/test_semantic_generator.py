from __future__ import annotations

import json

import pytest

from sonogpt.data.semantic_generator import (
    canonical_exam_json,
    sample_semantic_cases,
    semantic_case_id_for,
)
from sonogpt.schemas.domain import ThyroidExam


def test_same_seed_produces_same_unique_semantic_cases() -> None:
    first = sample_semantic_cases(40, seed=1234)
    second = sample_semantic_cases(40, seed=1234)

    assert first == second
    assert len({case.semantic_case_id for case in first}) == 40


def test_sampled_cases_are_schema_valid_and_content_addressed() -> None:
    for case in sample_semantic_cases(50, seed=20260824):
        validated = ThyroidExam.model_validate(case.exam.model_dump(mode="json"))

        assert validated == case.exam
        assert semantic_case_id_for(validated) == case.semantic_case_id
        assert json.loads(canonical_exam_json(validated)) == validated.model_dump(
            mode="json"
        )


def test_negative_case_count_is_rejected() -> None:
    with pytest.raises(ValueError):
        sample_semantic_cases(-1)


def test_non_integer_case_count_is_rejected() -> None:
    with pytest.raises(TypeError):
        sample_semantic_cases(True)
