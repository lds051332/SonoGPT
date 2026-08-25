"""Versioned quality-control rules for reports and optional structures.

These checks are engineering guards for a learning demo. They are not a
TI-RADS implementation and must not be treated as diagnostic advice.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Literal

from sonogpt.baselines.rule_extract import extract_exam
from sonogpt.data.constraints import synthetic_constraint_violations
from sonogpt.evaluation.metrics import (
    DIMENSION_TOLERANCE_MM,
    FIELD_NAMES,
    compare_fields,
    gold_field_values,
    is_stated_field,
)
from sonogpt.evaluation.report_parser import is_stated_value
from sonogpt.schemas.domain import (
    EchogenicFocus,
    ThyroidExam,
    Vascularity,
)

QC_RULES_VERSION = "1.0.0"
Severity = Literal["error", "warning", "info"]

FORBIDDEN_DIAGNOSTIC_TERMS = ("TI-RADS", "良性", "恶性", "癌", "确诊", "治疗")
_OMISSION_FIELDS = {
    "location_side": "QC.TEXT_OMITS_STATED_LOCATION",
    "composition": "QC.TEXT_OMITS_STATED_COMPOSITION",
    "echogenicity": "QC.TEXT_OMITS_STATED_ECHOGENICITY",
    "shape": "QC.TEXT_OMITS_STATED_SHAPE",
    "margin": "QC.TEXT_OMITS_STATED_MARGIN",
    "echogenic_foci": "QC.TEXT_OMITS_STATED_FOCI",
    "vascularity": "QC.TEXT_OMITS_STATED_VASCULARITY",
    "lymph_nodes": "QC.TEXT_OMITS_STATED_LYMPH_NODES",
}
_CORE_OMISSION_ERRORS = {
    "QC.TEXT_OMITS_STATED_LOCATION",
    "QC.TEXT_OMITS_STATED_COMPOSITION",
    "QC.TEXT_OMITS_STATED_ECHOGENICITY",
}
_CM_MEASUREMENT_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:cm)|\d+(?:\.\d+)?\s*[×xX*]\s*\d+(?:\.\d+)?(?:\s*[×xX*]\s*\d+(?:\.\d+)?)?\s*cm",
    re.IGNORECASE,
)
_MM_MEASUREMENT_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:mm)|\d+(?:\.\d+)?\s*[×xX*]\s*\d+(?:\.\d+)?(?:\s*[×xX*]\s*\d+(?:\.\d+)?)?\s*mm",
    re.IGNORECASE,
)
_EXPLICIT_NEGATIVE_VASCULARITY = (
    "未见明显血流",
    "未记录到明确血流",
    "无明确血流",
    "未显示明显血流",
    "结节内及周边未见明显血流",
)
_EXPLICIT_NEGATIVE_FOCI = (
    "未见明显强回声",
    "没有明确强回声",
    "未检出强回声",
    "未见明确强回声",
    "内部未见强回声",
)


@dataclass(frozen=True)
class QcFinding:
    rule_id: str
    severity: Severity
    message: str
    field: str | None = None
    evidence: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class QcResult:
    findings: tuple[QcFinding, ...]
    rules_version: str = QC_RULES_VERSION

    @property
    def error_count(self) -> int:
        return sum(finding.severity == "error" for finding in self.findings)

    @property
    def warning_count(self) -> int:
        return sum(finding.severity == "warning" for finding in self.findings)

    @property
    def passed(self) -> bool:
        return self.error_count == 0

    def has(self, rule_id: str) -> bool:
        return any(finding.rule_id == rule_id for finding in self.findings)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "rules_version": self.rules_version,
            "findings": [finding.to_dict() for finding in self.findings],
        }


def _clip(text: str, limit: int = 80) -> str:
    stripped = " ".join(text.split())
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 1] + "…"


def _finding(
    rule_id: str,
    message: str,
    *,
    severity: Severity = "warning",
    field: str | None = None,
    evidence: str | None = None,
) -> QcFinding:
    return QcFinding(
        rule_id=rule_id,
        severity=severity,
        message=message,
        field=field,
        evidence=evidence,
    )


def _text_contradictions(report: str) -> tuple[QcFinding, ...]:
    findings: list[QcFinding] = []
    if "左叶" in report and "右叶" in report:
        findings.append(
            _finding(
                "QC.TEXT_FIELD_CONTRADICTION",
                "报告同时出现左叶和右叶",
                severity="error",
                field="location_side",
                evidence=_clip(report),
            )
        )
    if "非高于宽" in report and ("呈高于宽" in report or "前后径超过横径" in report):
        findings.append(
            _finding(
                "QC.TEXT_FIELD_CONTRADICTION",
                "报告同时出现高于宽与非高于宽描述",
                severity="error",
                field="shape",
                evidence=_clip(report),
            )
        )
    if "实性" in report and "无回声" in report and "囊" not in report:
        findings.append(
            _finding(
                "QC.TEXT_FIELD_CONTRADICTION",
                "报告同时将结节写成实性和无回声",
                severity="error",
                field="echogenicity",
                evidence=_clip(report),
            )
        )
    return tuple(findings)


def run_qc(report: str, structure: ThyroidExam | None = None) -> QcResult:
    """Check a report, optionally against a provided Schema v1 structure."""

    findings: list[QcFinding] = []
    stripped = report.strip()
    if not stripped:
        findings.append(
            _finding(
                "QC.EMPTY_REPORT",
                "报告文本为空",
                severity="error",
            )
        )
        return QcResult(findings=tuple(findings))

    for term in FORBIDDEN_DIAGNOSTIC_TERMS:
        if term in stripped:
            findings.append(
                _finding(
                    "QC.DIAGNOSTIC_LANGUAGE",
                    f"报告包含诊断性用语：{term}",
                    severity="error",
                    evidence=term,
                )
            )

    findings.extend(_text_contradictions(stripped))
    has_mm = _MM_MEASUREMENT_RE.search(stripped) is not None
    has_cm = _CM_MEASUREMENT_RE.search(stripped) is not None
    if has_mm and has_cm:
        findings.append(
            _finding(
                "QC.UNIT_MIXED_MM_CM",
                "报告同时使用 mm 与 cm 描述尺寸，容易造成单位错误",
                severity="warning",
                field="dimensions_mm",
                evidence=_clip(stripped),
            )
        )

    extracted = extract_exam(stripped)
    if not extracted.parseable:
        findings.append(
            _finding(
                "QC.UNPARSEABLE_REPORT",
                "规则抽取无法从报告中得到可对齐结构",
                severity="error",
            )
        )

    if structure is None:
        return QcResult(findings=tuple(findings))

    parsed = extracted.exam
    fields = compare_fields(parsed, structure)
    if is_stated_field("location_side", structure) and not (
        fields.per_field["location_side"]["correct"]
    ):
        findings.append(
            _finding(
                "QC.TEXT_LOCATION_MISMATCH",
                "报告部位与结构中的侧别不一致",
                severity="error",
                field="location_side",
                evidence=_clip(stripped),
            )
        )
    gold_dimensions = structure.nodules[0].dimensions_mm
    parsed_dimensions = parsed.nodules[0].dimensions_mm
    if isinstance(gold_dimensions, list):
        mismatch = not isinstance(parsed_dimensions, list)
        if isinstance(parsed_dimensions, list):
            if len(parsed_dimensions) != len(gold_dimensions):
                mismatch = True
            elif any(
                abs(left - right) > DIMENSION_TOLERANCE_MM
                for left, right in zip(parsed_dimensions, gold_dimensions, strict=True)
            ):
                mismatch = True
        if mismatch:
            findings.append(
                _finding(
                    "QC.TEXT_DIMENSION_MISMATCH",
                    "报告尺寸与结构中的毫米值不一致",
                    severity="error",
                    field="dimensions_mm",
                    evidence=_clip(stripped),
                )
            )

    for name in FIELD_NAMES:
        if name in {"location_side", "location_segment"}:
            continue
        if not is_stated_field(name, structure):
            continue
        if fields.per_field[name]["correct"]:
            continue
        rule_id = _OMISSION_FIELDS[name]
        severity: Severity = (
            "error" if rule_id in _CORE_OMISSION_ERRORS else "warning"
        )
        if is_stated_value(gold_field_values(parsed)[name]):
            findings.append(
                _finding(
                    "QC.TEXT_FIELD_VALUE_MISMATCH",
                    f"报告中的 {name} 与结构不一致",
                    severity="error",
                    field=name,
                    evidence=_clip(stripped),
                )
            )
        else:
            findings.append(
                _finding(
                    rule_id,
                    f"结构中已给出 {name}，但报告未写出对应观察",
                    severity=severity,
                    field=name,
                    evidence=_clip(stripped),
                )
            )

    nodule = structure.nodules[0]
    if nodule.vascularity == Vascularity.NOT_MENTIONED and any(
        phrase in stripped for phrase in _EXPLICIT_NEGATIVE_VASCULARITY
    ):
        findings.append(
            _finding(
                "QC.STRUCTURE_NEGATION_MISLABEL",
                "报告已明示未见血流，但结构写成 not_mentioned 而不是 none",
                severity="warning",
                field="vascularity",
            )
        )
    if nodule.echogenic_foci == EchogenicFocus.NOT_MENTIONED and any(
        phrase in stripped for phrase in _EXPLICIT_NEGATIVE_FOCI
    ):
        findings.append(
            _finding(
                "QC.STRUCTURE_NEGATION_MISLABEL",
                "报告已明示未见强回声，但结构写成 not_mentioned 而不是 none",
                severity="warning",
                field="echogenic_foci",
            )
        )

    for violation in synthetic_constraint_violations(structure):
        findings.append(
            _finding(
                "QC.STRUCTURE_INTERNAL_CONSISTENCY",
                "结构内部组合与工程一致性规则冲突：" + violation,
                severity="warning",
                evidence=violation,
            )
        )

    return QcResult(findings=tuple(findings))
