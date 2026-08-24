from __future__ import annotations

import pytest
from pydantic import ValidationError

from sonogpt.schemas.domain import ThyroidExam


def _minimal_exam(**nodule_overrides: object) -> dict[str, object]:
    nodule: dict[str, object] = {
        "location": {"side": "right", "segment": "middle"},
        "dimensions_mm": [8, 6],
    }
    nodule.update(nodule_overrides)
    return {"nodules": [nodule]}


def test_minimal_exam_uses_versioned_defaults() -> None:
    exam = ThyroidExam.model_validate(_minimal_exam())

    assert exam.schema_version == "1.0.0"
    assert exam.organ == "thyroid"
    assert exam.nodules[0].composition.value == "not_mentioned"


@pytest.mark.parametrize(
    "dimensions",
    ([8], [8, 6, 4, 2], [8, 0], [-8, 6], [8, float("inf")], [True, 6]),
)
def test_invalid_dimensions_are_rejected(dimensions: list[object]) -> None:
    with pytest.raises(ValidationError):
        ThyroidExam.model_validate(_minimal_exam(dimensions_mm=dimensions))


def test_extra_fields_are_rejected() -> None:
    payload = _minimal_exam(unexpected="value")

    with pytest.raises(ValidationError):
        ThyroidExam.model_validate(payload)


def test_v1_requires_exactly_one_nodule() -> None:
    nodule = _minimal_exam()["nodules"][0]  # type: ignore[index]

    with pytest.raises(ValidationError):
        ThyroidExam.model_validate({"nodules": []})
    with pytest.raises(ValidationError):
        ThyroidExam.model_validate({"nodules": [nodule, nodule]})


def test_isthmus_requires_not_applicable_segment() -> None:
    with pytest.raises(ValidationError):
        ThyroidExam.model_validate(
            _minimal_exam(location={"side": "isthmus", "segment": "middle"})
        )

    exam = ThyroidExam.model_validate(
        _minimal_exam(location={"side": "isthmus", "segment": "not_applicable"})
    )
    assert exam.nodules[0].location.side.value == "isthmus"


def test_unmentioned_and_explicit_none_are_distinct() -> None:
    unmentioned = ThyroidExam.model_validate(_minimal_exam())
    explicit_none = ThyroidExam.model_validate(
        _minimal_exam(echogenic_foci="none", vascularity="none")
    )

    assert unmentioned.nodules[0].echogenic_foci.value == "not_mentioned"
    assert explicit_none.nodules[0].echogenic_foci.value == "none"
    assert unmentioned.nodules[0].vascularity.value == "not_mentioned"
    assert explicit_none.nodules[0].vascularity.value == "none"
