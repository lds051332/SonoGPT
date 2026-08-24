"""Canonical V1 schema for a single thyroid nodule.

The schema deliberately separates an unmentioned observation from an explicit
negative finding. It is a data contract for generation, training, evaluation,
and inference; it is not a diagnostic or TI-RADS implementation.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StringEnum(str, Enum):
    """Enum whose serialized representation is its string value."""


class ObservationState(StringEnum):
    NOT_MENTIONED = "not_mentioned"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class LocationSide(StringEnum):
    LEFT = "left"
    RIGHT = "right"
    ISTHMUS = "isthmus"
    NOT_MENTIONED = "not_mentioned"
    UNKNOWN = "unknown"


class LocationSegment(StringEnum):
    UPPER = "upper"
    MIDDLE = "middle"
    LOWER = "lower"
    NOT_MENTIONED = "not_mentioned"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class Composition(StringEnum):
    SOLID = "solid"
    CYSTIC = "cystic"
    MIXED_CYSTIC_SOLID = "mixed_cystic_solid"
    SPONGIFORM = "spongiform"
    NOT_MENTIONED = "not_mentioned"
    UNKNOWN = "unknown"


class Echogenicity(StringEnum):
    ANECHOIC = "anechoic"
    HYPERECHOIC = "hyperechoic"
    ISOECHOIC = "isoechoic"
    HYPOECHOIC = "hypoechoic"
    VERY_HYPOECHOIC = "very_hypoechoic"
    NOT_MENTIONED = "not_mentioned"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class Shape(StringEnum):
    WIDER_THAN_TALL = "wider_than_tall"
    TALLER_THAN_WIDE = "taller_than_wide"
    NOT_MENTIONED = "not_mentioned"
    UNKNOWN = "unknown"


class Margin(StringEnum):
    SMOOTH = "smooth"
    ILL_DEFINED = "ill_defined"
    LOBULATED_OR_IRREGULAR = "lobulated_or_irregular"
    EXTRATHYROIDAL_EXTENSION = "extrathyroidal_extension"
    NOT_MENTIONED = "not_mentioned"
    UNKNOWN = "unknown"


class EchogenicFocus(StringEnum):
    NONE = "none"
    COMET_TAIL = "comet_tail"
    MACROCALCIFICATION = "macrocalcification"
    PERIPHERAL_CALCIFICATION = "peripheral_calcification"
    PUNCTATE_ECHOGENIC_FOCI = "punctate_echogenic_foci"
    NOT_MENTIONED = "not_mentioned"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class Vascularity(StringEnum):
    NONE = "none"
    PERIPHERAL = "peripheral"
    INTERNAL = "internal"
    MIXED = "mixed"
    NOT_MENTIONED = "not_mentioned"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class LymphNodeFinding(StringEnum):
    NO_SUSPICIOUS = "no_suspicious"
    SUSPICIOUS = "suspicious"
    NOT_MENTIONED = "not_mentioned"
    UNKNOWN = "unknown"


class SonoBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Location(SonoBaseModel):
    side: LocationSide
    segment: LocationSegment = LocationSegment.NOT_MENTIONED

    @model_validator(mode="after")
    def validate_side_segment(self) -> "Location":
        if (
            self.side == LocationSide.ISTHMUS
            and self.segment != LocationSegment.NOT_APPLICABLE
        ):
            raise ValueError("isthmus location requires segment='not_applicable'")
        if (
            self.side in {LocationSide.LEFT, LocationSide.RIGHT}
            and self.segment == LocationSegment.NOT_APPLICABLE
        ):
            raise ValueError("left/right location cannot use segment='not_applicable'")
        return self


class Nodule(SonoBaseModel):
    nodule_id: str = Field(default="n1", min_length=1, max_length=32)
    location: Location
    dimensions_mm: list[float] | ObservationState = Field(
        default=ObservationState.NOT_MENTIONED,
        description=(
            "Measurements in transverse, anteroposterior, and optional "
            "longitudinal order, all in millimetres"
        ),
    )
    composition: Composition = Composition.NOT_MENTIONED
    echogenicity: Echogenicity = Echogenicity.NOT_MENTIONED
    shape: Shape = Shape.NOT_MENTIONED
    margin: Margin = Margin.NOT_MENTIONED
    echogenic_foci: EchogenicFocus = EchogenicFocus.NOT_MENTIONED
    vascularity: Vascularity = Vascularity.NOT_MENTIONED

    @field_validator("dimensions_mm", mode="before")
    @classmethod
    def validate_dimensions(
        cls, value: list[float] | ObservationState | str
    ) -> list[float] | ObservationState | str:
        if not isinstance(value, list):
            return value
        if len(value) not in {2, 3}:
            raise ValueError("dimensions_mm must contain two or three values")
        for dimension in value:
            if isinstance(dimension, bool) or not isinstance(dimension, (int, float)):
                raise ValueError("each dimension must be a number")
            if not math.isfinite(float(dimension)) or float(dimension) <= 0:
                raise ValueError("each dimension must be finite and greater than zero")
        return value


class ThyroidExam(SonoBaseModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    organ: Literal["thyroid"] = "thyroid"
    nodules: list[Nodule] = Field(min_length=1, max_length=1)
    lymph_nodes: LymphNodeFinding = LymphNodeFinding.NOT_MENTIONED
