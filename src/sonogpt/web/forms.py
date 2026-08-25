"""Convert the website payload into a validated ThyroidExam."""

from __future__ import annotations

import copy
from typing import Any, Mapping

from sonogpt.schemas.domain import LocationSide, ThyroidExam


def coerce_exam(payload: Mapping[str, Any]) -> ThyroidExam:
    """Validate Schema v1 and force isthmus segment to not_applicable."""

    data = copy.deepcopy(dict(payload))
    nodules = data.get("nodules")
    if isinstance(nodules, list) and nodules and isinstance(nodules[0], dict):
        location = dict(nodules[0].get("location") or {})
        if location.get("side") == LocationSide.ISTHMUS.value:
            location["segment"] = "not_applicable"
            nodules[0]["location"] = location
    return ThyroidExam.model_validate(data)
