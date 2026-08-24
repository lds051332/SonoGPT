"""Deterministic template families for the JSON-to-report task."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Literal

from sonogpt.baselines.template_report import build_report_parts, render_report
from sonogpt.data.semantic_generator import (
    DEFAULT_SEED,
    GENERATOR_VERSION,
    SemanticCase,
    canonical_exam_json,
)
from sonogpt.schemas.domain import ThyroidExam

RENDERER_VERSION = "1.1.0"
BASELINE_TEMPLATE_FAMILY = "location_first_v2"
TEMPLATE_FAMILIES = (
    BASELINE_TEMPLATE_FAMILY,
    "descriptor_first_v2",
    "dimensions_first_v2",
    "flow_first_v2",
)


def _finish(clauses: list[str], lymph_nodes: str | None) -> str:
    report = "，".join(clauses) + "。"
    return report + (lymph_nodes or "")


def _render_descriptor_first(exam: ThyroidExam) -> str:
    parts = build_report_parts(exam)
    clauses = [f"可见一枚{parts.descriptor}，位于{parts.location}"]
    clauses.extend(
        clause
        for clause in (
            parts.margin,
            parts.shape,
            parts.dimensions,
            parts.vascularity,
            parts.echogenic_foci,
        )
        if clause
    )
    return _finish(clauses, parts.lymph_nodes)


def _render_dimensions_first(exam: ThyroidExam) -> str:
    parts = build_report_parts(exam)
    if parts.dimensions:
        measurement = parts.dimensions.removeprefix("大小约")
        lead = f"一枚约{measurement}的{parts.descriptor}位于{parts.location}"
    else:
        lead = f"一枚{parts.descriptor}位于{parts.location}"
    clauses = [lead]
    clauses.extend(
        clause
        for clause in (
            parts.echogenic_foci,
            parts.margin,
            parts.vascularity,
            parts.shape,
        )
        if clause
    )
    return _finish(clauses, parts.lymph_nodes)


def _render_flow_first(exam: ThyroidExam) -> str:
    parts = build_report_parts(exam)
    clauses = [f"{parts.location}探及一枚{parts.descriptor}"]
    clauses.extend(
        clause
        for clause in (
            parts.vascularity,
            parts.dimensions,
            parts.echogenic_foci,
            parts.shape,
            parts.margin,
        )
        if clause
    )
    return _finish(clauses, parts.lymph_nodes)


_RENDERERS = {
    BASELINE_TEMPLATE_FAMILY: render_report,
    "descriptor_first_v2": _render_descriptor_first,
    "dimensions_first_v2": _render_dimensions_first,
    "flow_first_v2": _render_flow_first,
}


def render_with_family(exam: ThyroidExam, template_family: str) -> str:
    """Render one exam with a named, versioned word-order family."""

    try:
        renderer = _RENDERERS[template_family]
    except KeyError as exc:
        raise ValueError(f"unknown template family: {template_family}") from exc
    return renderer(exam)


def sample_id_for(semantic_case_id: str, template_family: str) -> str:
    """Return the stable identity of one semantic-case/template pair."""

    if template_family not in TEMPLATE_FAMILIES:
        raise ValueError(f"unknown template family: {template_family}")
    identity = (
        f"generate:{semantic_case_id}:{template_family}:{RENDERER_VERSION}"
    ).encode("utf-8")
    return f"sample_{hashlib.sha256(identity).hexdigest()}"


@dataclass(frozen=True)
class GeneratedSample:
    """One canonical generate-task training row."""

    sample_id: str
    semantic_case_id: str
    task: Literal["generate"]
    input: str
    target: str
    template_family: str
    generator_version: str
    renderer_version: str
    schema_version: str
    seed: int
    source: Literal["synthetic"]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def render_case(
    semantic_case: SemanticCase,
    template_family: str,
    seed: int = DEFAULT_SEED,
) -> GeneratedSample:
    """Create a stable sample from one semantic case and one template family."""

    report = render_with_family(semantic_case.exam, template_family)
    return GeneratedSample(
        sample_id=sample_id_for(semantic_case.semantic_case_id, template_family),
        semantic_case_id=semantic_case.semantic_case_id,
        task="generate",
        input=canonical_exam_json(semantic_case.exam),
        target=report,
        template_family=template_family,
        generator_version=GENERATOR_VERSION,
        renderer_version=RENDERER_VERSION,
        schema_version=semantic_case.exam.schema_version,
        seed=seed,
        source="synthetic",
    )


def render_cases(
    semantic_cases: tuple[SemanticCase, ...],
    template_families: tuple[str, ...] = TEMPLATE_FAMILIES,
    seed: int = DEFAULT_SEED,
) -> tuple[GeneratedSample, ...]:
    """Render each semantic case once with every selected family."""

    if len(template_families) != len(set(template_families)):
        raise ValueError("template_families must not contain duplicates")
    unknown = set(template_families).difference(TEMPLATE_FAMILIES)
    if unknown:
        raise ValueError(f"unknown template families: {sorted(unknown)}")

    return tuple(
        render_case(semantic_case, template_family, seed)
        for semantic_case in semantic_cases
        for template_family in template_families
    )
