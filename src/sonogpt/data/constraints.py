"""Versioned quality constraints for synthetic V1 semantic cases.

These rules constrain generated training data only. They are not diagnostic
rules and do not narrow the canonical Schema accepted from external sources.
"""

from __future__ import annotations

from sonogpt.schemas.domain import (
    Composition,
    EchogenicFocus,
    Echogenicity,
    Margin,
    Nodule,
    ObservationState,
    Shape,
    ThyroidExam,
    Vascularity,
)

SYNTHETIC_CONSTRAINTS_VERSION = "1.1.0"
MAX_SYNTHETIC_DIMENSION_RATIO = 3.0
MIN_SYNTHETIC_REPORTED_DIMENSION_MM = 3.0

_CYSTIC_ALLOWED_FOCI = {
    EchogenicFocus.NONE,
    EchogenicFocus.COMET_TAIL,
    EchogenicFocus.NOT_MENTIONED,
    EchogenicFocus.UNKNOWN,
}
_CYSTIC_ALLOWED_VASCULARITY = {
    Vascularity.NONE,
    Vascularity.PERIPHERAL,
    Vascularity.NOT_MENTIONED,
    Vascularity.UNKNOWN,
}
_BENIGN_PATTERN_ALLOWED_MARGINS = {
    Margin.SMOOTH,
    Margin.ILL_DEFINED,
    Margin.NOT_MENTIONED,
    Margin.UNKNOWN,
}
_BENIGN_PATTERN_ALLOWED_SHAPES = {
    Shape.WIDER_THAN_TALL,
    Shape.NOT_MENTIONED,
    Shape.UNKNOWN,
}


def synthetic_constraint_violations(exam: ThyroidExam) -> tuple[str, ...]:
    """Return stable rule IDs for combinations excluded from synthetic data."""

    nodule = exam.nodules[0]
    violations: list[str] = []
    _validate_observation_states(nodule, violations)
    _validate_composition_features(nodule, violations)
    _validate_dimensions_and_shape(nodule, violations)
    return tuple(violations)


def assert_synthetic_case_valid(exam: ThyroidExam) -> None:
    violations = synthetic_constraint_violations(exam)
    if violations:
        raise ValueError(
            "synthetic case violates constraints: " + ", ".join(violations)
        )


def _validate_observation_states(
    nodule: Nodule, violations: list[str]
) -> None:
    if nodule.dimensions_mm == ObservationState.NOT_APPLICABLE:
        violations.append("dimensions.not_applicable")
    if nodule.echogenicity == Echogenicity.NOT_APPLICABLE:
        violations.append("echogenicity.not_applicable")
    if nodule.echogenic_foci == EchogenicFocus.NOT_APPLICABLE:
        violations.append("echogenic_foci.not_applicable")
    if nodule.vascularity == Vascularity.NOT_APPLICABLE:
        violations.append("vascularity.not_applicable")


def _validate_composition_features(
    nodule: Nodule, violations: list[str]
) -> None:
    if nodule.composition in {
        Composition.SOLID,
        Composition.MIXED_CYSTIC_SOLID,
    } and nodule.echogenicity in {
        Echogenicity.ANECHOIC,
        Echogenicity.NOT_APPLICABLE,
    }:
        violations.append("composition.non_cystic_anechoic")

    if nodule.composition == Composition.CYSTIC:
        if nodule.echogenicity not in {
            Echogenicity.ANECHOIC,
            Echogenicity.NOT_MENTIONED,
            Echogenicity.UNKNOWN,
        }:
            violations.append("composition.cystic_echogenicity")
        if nodule.echogenic_foci not in _CYSTIC_ALLOWED_FOCI:
            violations.append("composition.cystic_echogenic_foci")
        if nodule.vascularity not in _CYSTIC_ALLOWED_VASCULARITY:
            violations.append("composition.cystic_vascularity")
        if nodule.shape not in _BENIGN_PATTERN_ALLOWED_SHAPES:
            violations.append("composition.cystic_shape")
        if nodule.margin not in _BENIGN_PATTERN_ALLOWED_MARGINS:
            violations.append("composition.cystic_margin")

    if nodule.composition == Composition.SPONGIFORM:
        if nodule.echogenicity in {
            Echogenicity.ANECHOIC,
            Echogenicity.VERY_HYPOECHOIC,
            Echogenicity.NOT_APPLICABLE,
        }:
            violations.append("composition.spongiform_echogenicity")
        if nodule.echogenic_foci not in _CYSTIC_ALLOWED_FOCI:
            violations.append("composition.spongiform_echogenic_foci")
        if nodule.vascularity not in _CYSTIC_ALLOWED_VASCULARITY:
            violations.append("composition.spongiform_vascularity")
        if nodule.shape not in _BENIGN_PATTERN_ALLOWED_SHAPES:
            violations.append("composition.spongiform_shape")
        if nodule.margin not in _BENIGN_PATTERN_ALLOWED_MARGINS:
            violations.append("composition.spongiform_margin")

    if nodule.echogenicity == Echogenicity.ANECHOIC:
        if nodule.echogenic_foci not in _CYSTIC_ALLOWED_FOCI:
            violations.append("echogenicity.anechoic_echogenic_foci")
        if nodule.vascularity not in _CYSTIC_ALLOWED_VASCULARITY:
            violations.append("echogenicity.anechoic_vascularity")
        if nodule.shape not in _BENIGN_PATTERN_ALLOWED_SHAPES:
            violations.append("echogenicity.anechoic_shape")
        if nodule.margin not in _BENIGN_PATTERN_ALLOWED_MARGINS:
            violations.append("echogenicity.anechoic_margin")


def _validate_dimensions_and_shape(
    nodule: Nodule, violations: list[str]
) -> None:
    if not isinstance(nodule.dimensions_mm, list):
        return

    dimensions = nodule.dimensions_mm
    if min(dimensions) < MIN_SYNTHETIC_REPORTED_DIMENSION_MM:
        violations.append("dimensions.below_synthetic_minimum")
    if max(dimensions) / min(dimensions) > MAX_SYNTHETIC_DIMENSION_RATIO:
        violations.append("dimensions.extreme_ratio")

    transverse_mm, anteroposterior_mm = dimensions[:2]
    if (
        nodule.shape == Shape.WIDER_THAN_TALL
        and anteroposterior_mm > transverse_mm
    ):
        violations.append("shape.wider_than_tall_mismatch")
    if (
        nodule.shape == Shape.TALLER_THAN_WIDE
        and anteroposterior_mm <= transverse_mm
    ):
        violations.append("shape.taller_than_wide_mismatch")
