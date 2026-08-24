"""Seeded semantic-case sampling for the V1 single-nodule task."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

from sonogpt.data.constraints import assert_synthetic_case_valid
from sonogpt.schemas.domain import (
    Composition,
    EchogenicFocus,
    Echogenicity,
    Location,
    LocationSegment,
    LocationSide,
    LymphNodeFinding,
    Margin,
    Nodule,
    ObservationState,
    Shape,
    ThyroidExam,
    Vascularity,
)

GENERATOR_VERSION = "1.1.0"
DEFAULT_SEED = 20260824

EnumT = TypeVar("EnumT", bound=Enum)


def canonical_exam_json(exam: ThyroidExam) -> str:
    """Serialize an exam with stable key ordering and no insignificant spaces."""

    return json.dumps(
        exam.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def semantic_case_id_for(exam: ThyroidExam) -> str:
    """Return a content-derived ID that is stable across sampling runs."""

    digest = hashlib.sha256(canonical_exam_json(exam).encode("utf-8")).hexdigest()
    return f"sc_{digest}"


@dataclass(frozen=True)
class SemanticCase:
    """A validated domain object paired with its content-derived identity."""

    semantic_case_id: str
    exam: ThyroidExam

    def __post_init__(self) -> None:
        expected = semantic_case_id_for(self.exam)
        if self.semantic_case_id != expected:
            raise ValueError("semantic_case_id does not match the canonical exam")

    @classmethod
    def from_exam(cls, exam: ThyroidExam) -> "SemanticCase":
        return cls(semantic_case_id=semantic_case_id_for(exam), exam=exam)


def _choose(
    rng: random.Random, weighted_values: tuple[tuple[EnumT, int], ...]
) -> EnumT:
    total = sum(weight for _, weight in weighted_values)
    position = rng.randrange(total)
    for value, weight in weighted_values:
        if position < weight:
            return value
        position -= weight
    raise RuntimeError("weighted choice did not select a value")


def _sample_location(rng: random.Random) -> Location:
    side = _choose(
        rng,
        (
            (LocationSide.LEFT, 44),
            (LocationSide.RIGHT, 44),
            (LocationSide.ISTHMUS, 8),
            (LocationSide.NOT_MENTIONED, 2),
            (LocationSide.UNKNOWN, 2),
        ),
    )
    if side == LocationSide.ISTHMUS:
        segment = LocationSegment.NOT_APPLICABLE
    elif side == LocationSide.NOT_MENTIONED:
        segment = LocationSegment.NOT_MENTIONED
    elif side == LocationSide.UNKNOWN:
        segment = LocationSegment.UNKNOWN
    else:
        segment = _choose(
            rng,
            (
                (LocationSegment.UPPER, 30),
                (LocationSegment.MIDDLE, 30),
                (LocationSegment.LOWER, 30),
                (LocationSegment.NOT_MENTIONED, 6),
                (LocationSegment.UNKNOWN, 4),
            ),
        )
    return Location(side=side, segment=segment)


def _sample_dimensions(
    rng: random.Random,
    shape: Shape,
) -> list[float] | ObservationState:
    state_roll = rng.randrange(100)
    if state_roll >= 94:
        if state_roll < 98:
            return ObservationState.NOT_MENTIONED
        return ObservationState.UNKNOWN

    dimension_count = 2 if rng.randrange(100) < 70 else 3
    if shape == Shape.TALLER_THAN_WIDE:
        transverse_mm = rng.randint(50, 295) / 10
        ap_ratio_percent = rng.randint(105, 135)
    elif shape == Shape.WIDER_THAN_TALL:
        transverse_mm = rng.randint(50, 400) / 10
        ap_ratio_percent = rng.randint(45, 95)
    else:
        transverse_mm = rng.randint(50, 320) / 10
        ap_ratio_percent = rng.randint(55, 125)

    anteroposterior_mm = min(
        40.0,
        max(3.0, round(transverse_mm * ap_ratio_percent / 100, 1)),
    )
    if (
        shape == Shape.TALLER_THAN_WIDE
        and anteroposterior_mm <= transverse_mm
    ):
        anteroposterior_mm = round(transverse_mm + 0.1, 1)
    dimensions = [transverse_mm, anteroposterior_mm]
    if dimension_count == 3:
        reference_mm = max(dimensions)
        longitudinal_mm = min(
            40.0,
            max(
                3.0,
                round(reference_mm * rng.randint(75, 125) / 100, 1),
            ),
        )
        dimensions.append(longitudinal_mm)
    return dimensions


def _sample_composition(rng: random.Random) -> Composition:
    return _choose(
        rng,
        (
            (Composition.SOLID, 48),
            (Composition.CYSTIC, 14),
            (Composition.MIXED_CYSTIC_SOLID, 20),
            (Composition.SPONGIFORM, 8),
            (Composition.NOT_MENTIONED, 6),
            (Composition.UNKNOWN, 4),
        ),
    )


def _sample_shape(
    rng: random.Random,
    composition: Composition,
    echogenicity: Echogenicity,
) -> Shape:
    if (
        composition in {Composition.CYSTIC, Composition.SPONGIFORM}
        or echogenicity == Echogenicity.ANECHOIC
    ):
        return _choose(
            rng,
            (
                (Shape.WIDER_THAN_TALL, 80),
                (Shape.NOT_MENTIONED, 12),
                (Shape.UNKNOWN, 8),
            ),
        )
    return _choose(
        rng,
        (
            (Shape.WIDER_THAN_TALL, 67),
            (Shape.TALLER_THAN_WIDE, 21),
            (Shape.NOT_MENTIONED, 8),
            (Shape.UNKNOWN, 4),
        ),
    )


def _sample_echogenicity(
    rng: random.Random, composition: Composition
) -> Echogenicity:
    if composition == Composition.CYSTIC:
        return _choose(
            rng,
            (
                (Echogenicity.ANECHOIC, 85),
                (Echogenicity.NOT_MENTIONED, 10),
                (Echogenicity.UNKNOWN, 5),
            ),
        )
    if composition == Composition.SPONGIFORM:
        return _choose(
            rng,
            (
                (Echogenicity.HYPERECHOIC, 10),
                (Echogenicity.ISOECHOIC, 55),
                (Echogenicity.HYPOECHOIC, 15),
                (Echogenicity.NOT_MENTIONED, 10),
                (Echogenicity.UNKNOWN, 10),
            ),
        )
    if composition in {Composition.SOLID, Composition.MIXED_CYSTIC_SOLID}:
        return _choose(
            rng,
            (
                (Echogenicity.HYPERECHOIC, 12),
                (Echogenicity.ISOECHOIC, 27),
                (Echogenicity.HYPOECHOIC, 40),
                (Echogenicity.VERY_HYPOECHOIC, 10),
                (Echogenicity.NOT_MENTIONED, 6),
                (Echogenicity.UNKNOWN, 5),
            ),
        )
    return _choose(
        rng,
        (
            (Echogenicity.ANECHOIC, 10),
            (Echogenicity.HYPERECHOIC, 10),
            (Echogenicity.ISOECHOIC, 25),
            (Echogenicity.HYPOECHOIC, 35),
            (Echogenicity.VERY_HYPOECHOIC, 8),
            (Echogenicity.NOT_MENTIONED, 7),
            (Echogenicity.UNKNOWN, 5),
        ),
    )


def _sample_margin(
    rng: random.Random,
    composition: Composition,
    echogenicity: Echogenicity,
) -> Margin:
    if (
        composition in {Composition.CYSTIC, Composition.SPONGIFORM}
        or echogenicity == Echogenicity.ANECHOIC
    ):
        return _choose(
            rng,
            (
                (Margin.SMOOTH, 75),
                (Margin.ILL_DEFINED, 12),
                (Margin.NOT_MENTIONED, 8),
                (Margin.UNKNOWN, 5),
            ),
        )
    return _choose(
        rng,
        (
            (Margin.SMOOTH, 48),
            (Margin.ILL_DEFINED, 18),
            (Margin.LOBULATED_OR_IRREGULAR, 17),
            (Margin.EXTRATHYROIDAL_EXTENSION, 3),
            (Margin.NOT_MENTIONED, 9),
            (Margin.UNKNOWN, 5),
        ),
    )


def _sample_echogenic_foci(
    rng: random.Random,
    composition: Composition,
    echogenicity: Echogenicity,
) -> EchogenicFocus:
    if (
        composition in {Composition.CYSTIC, Composition.SPONGIFORM}
        or echogenicity == Echogenicity.ANECHOIC
    ):
        return _choose(
            rng,
            (
                (EchogenicFocus.NONE, 40),
                (EchogenicFocus.COMET_TAIL, 40),
                (EchogenicFocus.NOT_MENTIONED, 13),
                (EchogenicFocus.UNKNOWN, 7),
            ),
        )
    return _choose(
        rng,
        (
            (EchogenicFocus.NONE, 32),
            (EchogenicFocus.COMET_TAIL, 10),
            (EchogenicFocus.MACROCALCIFICATION, 13),
            (EchogenicFocus.PERIPHERAL_CALCIFICATION, 10),
            (EchogenicFocus.PUNCTATE_ECHOGENIC_FOCI, 16),
            (EchogenicFocus.NOT_MENTIONED, 13),
            (EchogenicFocus.UNKNOWN, 6),
        ),
    )


def _sample_vascularity(
    rng: random.Random,
    composition: Composition,
    echogenicity: Echogenicity,
) -> Vascularity:
    if (
        composition in {Composition.CYSTIC, Composition.SPONGIFORM}
        or echogenicity == Echogenicity.ANECHOIC
    ):
        return _choose(
            rng,
            (
                (Vascularity.NONE, 45),
                (Vascularity.PERIPHERAL, 25),
                (Vascularity.NOT_MENTIONED, 20),
                (Vascularity.UNKNOWN, 10),
            ),
        )
    return _choose(
        rng,
        (
            (Vascularity.NONE, 26),
            (Vascularity.PERIPHERAL, 20),
            (Vascularity.INTERNAL, 21),
            (Vascularity.MIXED, 18),
            (Vascularity.NOT_MENTIONED, 10),
            (Vascularity.UNKNOWN, 5),
        ),
    )


def _sample_exam(rng: random.Random) -> ThyroidExam:
    composition = _sample_composition(rng)
    echogenicity = _sample_echogenicity(rng, composition)
    shape = _sample_shape(rng, composition, echogenicity)
    nodule = Nodule(
        location=_sample_location(rng),
        dimensions_mm=_sample_dimensions(rng, shape),
        composition=composition,
        echogenicity=echogenicity,
        shape=shape,
        margin=_sample_margin(rng, composition, echogenicity),
        echogenic_foci=_sample_echogenic_foci(
            rng, composition, echogenicity
        ),
        vascularity=_sample_vascularity(rng, composition, echogenicity),
    )
    lymph_nodes = _choose(
        rng,
        (
            (LymphNodeFinding.NO_SUSPICIOUS, 38),
            (LymphNodeFinding.SUSPICIOUS, 9),
            (LymphNodeFinding.NOT_MENTIONED, 46),
            (LymphNodeFinding.UNKNOWN, 7),
        ),
    )
    exam = ThyroidExam(nodules=[nodule], lymph_nodes=lymph_nodes)
    assert_synthetic_case_valid(exam)
    return exam


def sample_semantic_cases(
    count: int, seed: int = DEFAULT_SEED
) -> tuple[SemanticCase, ...]:
    """Sample unique, Schema-valid cases reproducibly from a local RNG."""

    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer")
    if count < 0:
        raise ValueError("count must be non-negative")

    rng = random.Random(seed)
    cases: list[SemanticCase] = []
    seen_ids: set[str] = set()
    max_attempts = max(100, count * 100)
    attempts = 0

    while len(cases) < count:
        if attempts >= max_attempts:
            raise RuntimeError("could not sample the requested number of unique cases")
        attempts += 1
        semantic_case = SemanticCase.from_exam(_sample_exam(rng))
        if semantic_case.semantic_case_id in seen_ids:
            continue
        seen_ids.add(semantic_case.semantic_case_id)
        cases.append(semantic_case)

    return tuple(cases)
