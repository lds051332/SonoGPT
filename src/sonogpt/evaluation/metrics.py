"""Generate-task metrics computed against gold exams and independent parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from sonogpt.data.renderers import TEMPLATE_FAMILIES, render_with_family
from sonogpt.evaluation.report_parser import (
    ParsedReport,
    is_stated_value,
    parse_report,
)
from sonogpt.schemas.domain import (
    LocationSegment,
    LocationSide,
    ObservationState,
    ThyroidExam,
)

DIMENSION_TOLERANCE_MM = 0.05
FIELD_NAMES = (
    "location_side",
    "location_segment",
    "composition",
    "echogenicity",
    "shape",
    "margin",
    "echogenic_foci",
    "vascularity",
    "lymph_nodes",
)


@dataclass(frozen=True)
class FieldComparison:
    mentioned_count: int
    mentioned_correct: int
    hallucinated_count: int
    unstated_count: int
    per_field: dict[str, dict[str, int]]

    @property
    def mentioned_accuracy(self) -> float | None:
        if self.mentioned_count == 0:
            return None
        return self.mentioned_correct / self.mentioned_count

    @property
    def hallucination_rate(self) -> float | None:
        if self.unstated_count == 0:
            return None
        return self.hallucinated_count / self.unstated_count

    def to_dict(self) -> dict[str, object]:
        return {
            "mentioned_count": self.mentioned_count,
            "mentioned_correct": self.mentioned_correct,
            "mentioned_accuracy": self.mentioned_accuracy,
            "hallucinated_count": self.hallucinated_count,
            "unstated_count": self.unstated_count,
            "hallucination_rate": self.hallucination_rate,
            "per_field": self.per_field,
        }


@dataclass(frozen=True)
class DimensionComparison:
    comparable_count: int
    count_match: int
    exact_match: int
    tolerant_match: int
    abs_error_sum: float
    abs_error_count: int

    @property
    def mean_abs_error_mm(self) -> float | None:
        if self.abs_error_count == 0:
            return None
        return self.abs_error_sum / self.abs_error_count

    def to_dict(self) -> dict[str, object]:
        total = self.comparable_count
        return {
            "comparable_count": total,
            "count_match_rate": None if total == 0 else self.count_match / total,
            "exact_match_rate": None if total == 0 else self.exact_match / total,
            "tolerant_match_rate": None if total == 0 else self.tolerant_match / total,
            "mean_abs_error_mm": self.mean_abs_error_mm,
            "tolerance_mm": DIMENSION_TOLERANCE_MM,
        }


@dataclass(frozen=True)
class ScoredExample:
    example_id: str
    split: str
    semantic_case_id: str
    generated_text: str
    reference_text: str
    exact_match: bool
    eos_finished: bool
    parseable: bool
    matched_template_family: str | None
    generation_error: str | None
    field_hits: dict[str, bool]
    dimension_exact: bool | None
    dimension_tolerant: bool | None


def gold_field_values(exam: ThyroidExam) -> dict[str, object]:
    nodule = exam.nodules[0]
    return {
        "location_side": nodule.location.side,
        "location_segment": nodule.location.segment,
        "composition": nodule.composition,
        "echogenicity": nodule.echogenicity,
        "shape": nodule.shape,
        "margin": nodule.margin,
        "echogenic_foci": nodule.echogenic_foci,
        "vascularity": nodule.vascularity,
        "lymph_nodes": exam.lymph_nodes,
        "dimensions_mm": nodule.dimensions_mm,
    }


def is_stated_field(name: str, exam: ThyroidExam) -> bool:
    values = gold_field_values(exam)
    if name == "location_segment":
        if values["location_side"] == LocationSide.ISTHMUS:
            return values["location_segment"] == LocationSegment.NOT_APPLICABLE
        return is_stated_value(values[name])
    if name == "location_side":
        return is_stated_value(values[name], allow_not_applicable=False)
    return is_stated_value(values[name])


def _field_equal(left: object, right: object) -> bool:
    if isinstance(left, list) or isinstance(right, list):
        return False
    left_raw = left.value if hasattr(left, "value") else left
    right_raw = right.value if hasattr(right, "value") else right
    return left_raw == right_raw


def _dimension_errors(
    predicted: list[float] | ObservationState | object,
    gold: list[float] | ObservationState | object,
) -> tuple[bool, bool, float | None]:
    if not isinstance(predicted, list) or not isinstance(gold, list):
        return False, False, None
    count_match = len(predicted) == len(gold)
    if not count_match:
        return False, False, None
    abs_errors = [abs(pred - true) for pred, true in zip(predicted, gold, strict=True)]
    exact = all(error <= 1e-6 for error in abs_errors)
    tolerant = all(error <= DIMENSION_TOLERANCE_MM for error in abs_errors)
    mean_error = sum(abs_errors) / len(abs_errors)
    return exact, tolerant, mean_error


def compare_dimensions(parsed: ThyroidExam, gold: ThyroidExam) -> DimensionComparison:
    gold_dimensions = gold.nodules[0].dimensions_mm
    predicted_dimensions = parsed.nodules[0].dimensions_mm
    exact, tolerant, mean_error = _dimension_errors(
        predicted_dimensions, gold_dimensions
    )
    comparable = isinstance(gold_dimensions, list)
    return DimensionComparison(
        comparable_count=int(comparable),
        count_match=int(
            comparable
            and isinstance(predicted_dimensions, list)
            and len(predicted_dimensions) == len(gold_dimensions)
        ),
        exact_match=int(comparable and exact),
        tolerant_match=int(comparable and tolerant),
        abs_error_sum=0.0 if mean_error is None or not comparable else mean_error,
        abs_error_count=int(comparable and mean_error is not None),
    )


def compare_fields(parsed: ThyroidExam, gold: ThyroidExam) -> FieldComparison:
    predicted = gold_field_values(parsed)
    actual = gold_field_values(gold)
    mentioned_count = 0
    mentioned_correct = 0
    hallucinated_count = 0
    unstated_count = 0
    per_field: dict[str, dict[str, int]] = {}
    for name in FIELD_NAMES:
        stated = is_stated_field(name, gold)
        equal = _field_equal(predicted[name], actual[name])
        hallucinated = (not stated) and is_stated_field(name, parsed)
        per_field[name] = {
            "mentioned": int(stated),
            "correct": int(stated and equal),
            "hallucinated": int(hallucinated),
        }
        if stated:
            mentioned_count += 1
            mentioned_correct += int(equal)
        else:
            unstated_count += 1
            hallucinated_count += int(hallucinated)
    return FieldComparison(
        mentioned_count=mentioned_count,
        mentioned_correct=mentioned_correct,
        hallucinated_count=hallucinated_count,
        unstated_count=unstated_count,
        per_field=per_field,
    )


def matched_template_family(text: str, exam: ThyroidExam) -> str | None:
    for family in TEMPLATE_FAMILIES:
        if text == render_with_family(exam, family):
            return family
    return None


def score_generated_report(
    *,
    example_id: str,
    split: str,
    semantic_case_id: str,
    generated_text: str,
    reference_text: str,
    gold_exam: ThyroidExam,
    eos_finished: bool,
    generation_error: str | None,
) -> tuple[ScoredExample, ParsedReport, FieldComparison, DimensionComparison]:
    parsed = parse_report(generated_text)
    fields = compare_fields(parsed.exam, gold_exam)
    dimension = compare_dimensions(parsed.exam, gold_exam)
    comparable = isinstance(gold_exam.nodules[0].dimensions_mm, list)
    exact = bool(dimension.exact_match)
    tolerant = bool(dimension.tolerant_match)
    example = ScoredExample(
        example_id=example_id,
        split=split,
        semantic_case_id=semantic_case_id,
        generated_text=generated_text,
        reference_text=reference_text,
        exact_match=generated_text == reference_text,
        eos_finished=eos_finished,
        parseable=parsed.parseable and generation_error is None,
        matched_template_family=matched_template_family(generated_text, gold_exam),
        generation_error=generation_error,
        field_hits={
            name: bool(stats["correct"])
            for name, stats in fields.per_field.items()
            if stats["mentioned"]
        },
        dimension_exact=None if not comparable else exact,
        dimension_tolerant=None if not comparable else tolerant,
    )
    return example, parsed, fields, dimension


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def aggregate_scores(
    rows: tuple[ScoredExample, ...],
    field_rows: tuple[FieldComparison, ...],
    dimension_rows: tuple[DimensionComparison, ...],
    *,
    teacher_forced_loss: float | None,
    teacher_forced_token_accuracy: float | None,
    teacher_forced_token_count: int,
) -> dict[str, object]:
    if len(rows) != len(field_rows) or len(rows) != len(dimension_rows):
        raise ValueError("score row counts must match")
    sample_count = len(rows)
    template_counts = {family: 0 for family in TEMPLATE_FAMILIES}
    template_counts["none"] = 0
    mentioned_count = 0
    mentioned_correct = 0
    hallucinated_count = 0
    unstated_count = 0
    per_field: dict[str, dict[str, int]] = {
        name: {"mentioned": 0, "correct": 0, "hallucinated": 0} for name in FIELD_NAMES
    }
    dim_comparable = 0
    dim_count_match = 0
    dim_exact = 0
    dim_tolerant = 0
    abs_error_sum = 0.0
    abs_error_count = 0
    for scored, fields, dimensions in zip(rows, field_rows, dimension_rows, strict=True):
        family = scored.matched_template_family or "none"
        template_counts[family] += 1
        mentioned_count += fields.mentioned_count
        mentioned_correct += fields.mentioned_correct
        hallucinated_count += fields.hallucinated_count
        unstated_count += fields.unstated_count
        for name, stats in fields.per_field.items():
            for key in ("mentioned", "correct", "hallucinated"):
                per_field[name][key] += stats[key]
        dim_comparable += dimensions.comparable_count
        dim_count_match += dimensions.count_match
        dim_exact += dimensions.exact_match
        dim_tolerant += dimensions.tolerant_match
        abs_error_sum += dimensions.abs_error_sum
        abs_error_count += dimensions.abs_error_count

    field_accuracy = {
        name: {
            "mentioned_count": stats["mentioned"],
            "accuracy": _rate(stats["correct"], stats["mentioned"]),
            "hallucinated_count": stats["hallucinated"],
        }
        for name, stats in per_field.items()
    }
    return {
        "sample_count": sample_count,
        "eos_finished_rate": _rate(sum(row.eos_finished for row in rows), sample_count),
        "parseable_rate": _rate(sum(row.parseable for row in rows), sample_count),
        "exact_match_rate": _rate(sum(row.exact_match for row in rows), sample_count),
        "generation_error_rate": _rate(
            sum(row.generation_error is not None for row in rows), sample_count
        ),
        "teacher_forced": {
            "loss": teacher_forced_loss,
            "token_accuracy": teacher_forced_token_accuracy,
            "target_token_count": teacher_forced_token_count,
        },
        "fields": {
            "mentioned_count": mentioned_count,
            "mentioned_accuracy": _rate(mentioned_correct, mentioned_count),
            "hallucinated_count": hallucinated_count,
            "unstated_count": unstated_count,
            "hallucination_rate": _rate(hallucinated_count, unstated_count),
            "per_field": field_accuracy,
        },
        "dimensions": {
            "comparable_count": dim_comparable,
            "count_match_rate": _rate(dim_count_match, dim_comparable),
            "exact_match_rate": _rate(dim_exact, dim_comparable),
            "tolerant_match_rate": _rate(dim_tolerant, dim_comparable),
            "mean_abs_error_mm": (
                None if abs_error_count == 0 else abs_error_sum / abs_error_count
            ),
            "tolerance_mm": DIMENSION_TOLERANCE_MM,
        },
        "template_exact_match": {
            "any_family_rate": _rate(
                sum(row.matched_template_family is not None for row in rows),
                sample_count,
            ),
            "family_counts": template_counts,
        },
    }


def compare_split_metrics(
    primary: Mapping[str, object],
    reference: Mapping[str, object],
) -> dict[str, object]:
    def nested_get(payload: Mapping[str, object], *keys: str) -> object:
        current: object = payload
        for key in keys:
            if not isinstance(current, Mapping):
                return None
            current = current.get(key)
        return current

    keys = {
        "exact_match_rate": ("exact_match_rate",),
        "parseable_rate": ("parseable_rate",),
        "mentioned_field_accuracy": ("fields", "mentioned_accuracy"),
        "dimension_exact_match_rate": ("dimensions", "exact_match_rate"),
        "any_template_family_rate": ("template_exact_match", "any_family_rate"),
        "teacher_forced_token_accuracy": ("teacher_forced", "token_accuracy"),
        "teacher_forced_loss": ("teacher_forced", "loss"),
    }
    deltas: dict[str, object] = {}
    for name, path in keys.items():
        left = nested_get(primary, *path)
        right = nested_get(reference, *path)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            deltas[name] = left - right
        else:
            deltas[name] = None
    return deltas
