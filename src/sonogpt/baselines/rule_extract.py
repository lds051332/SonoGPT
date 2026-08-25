"""Versioned rule extractor: report text → Schema v1.

This is the extract-task baseline. It does not use the neural model and does
not infer diagnoses. Unmentioned observations stay `not_mentioned`.
"""

from __future__ import annotations

from dataclasses import dataclass

from sonogpt.data.semantic_generator import canonical_exam_json
from sonogpt.evaluation.report_parser import ParsedReport, parse_report
from sonogpt.schemas.domain import ThyroidExam

RULE_EXTRACT_VERSION = "1.0.0"


@dataclass(frozen=True)
class ExtractResult:
    exam: ThyroidExam
    canonical_json: str
    parseable: bool
    explicit_field_count: int
    extractor: str = "rule_extract"
    version: str = RULE_EXTRACT_VERSION
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "exam": self.exam.model_dump(mode="json"),
            "canonical_json": self.canonical_json,
            "parseable": self.parseable,
            "explicit_field_count": self.explicit_field_count,
            "extractor": self.extractor,
            "version": self.version,
            "notes": list(self.notes),
        }


def extract_exam(report: str) -> ExtractResult:
    """Parse a Chinese report with the independent rule extractor."""

    parsed: ParsedReport = parse_report(report)
    return ExtractResult(
        exam=parsed.exam,
        canonical_json=canonical_exam_json(parsed.exam),
        parseable=parsed.parseable,
        explicit_field_count=parsed.explicit_field_count,
        notes=parsed.notes,
    )
