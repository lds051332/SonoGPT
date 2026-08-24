"""Deterministic report renderer for the V1 thyroid schema."""

from __future__ import annotations

from dataclasses import dataclass

from sonogpt.schemas.domain import (
    Composition,
    EchogenicFocus,
    Echogenicity,
    LymphNodeFinding,
    Location,
    LocationSegment,
    LocationSide,
    Margin,
    Nodule,
    ObservationState,
    Shape,
    ThyroidExam,
    Vascularity,
)


_SIDE_TEXT = {
    LocationSide.LEFT: "左叶",
    LocationSide.RIGHT: "右叶",
    LocationSide.ISTHMUS: "峡部",
}

_SEGMENT_TEXT = {
    LocationSegment.UPPER: "上部",
    LocationSegment.MIDDLE: "中部",
    LocationSegment.LOWER: "下部",
}

_COMPOSITION_TEXT = {
    Composition.SOLID: "实性",
    Composition.CYSTIC: "囊性",
    Composition.MIXED_CYSTIC_SOLID: "囊实性",
    Composition.SPONGIFORM: "海绵状",
}

_ECHOGENICITY_TEXT = {
    Echogenicity.ANECHOIC: "无回声",
    Echogenicity.HYPERECHOIC: "高回声",
    Echogenicity.ISOECHOIC: "等回声",
    Echogenicity.HYPOECHOIC: "低回声",
    Echogenicity.VERY_HYPOECHOIC: "极低回声",
}

_SHAPE_TEXT = {
    Shape.WIDER_THAN_TALL: "呈非高于宽形",
    Shape.TALLER_THAN_WIDE: "呈高于宽形",
}

_MARGIN_TEXT = {
    Margin.SMOOTH: "边缘光滑",
    Margin.ILL_DEFINED: "边界欠清",
    Margin.LOBULATED_OR_IRREGULAR: "边缘分叶或不规则",
    Margin.EXTRATHYROIDAL_EXTENSION: "可见甲状腺外侵犯征象",
}

_FOCUS_TEXT = {
    EchogenicFocus.NONE: "内未见明显强回声",
    EchogenicFocus.COMET_TAIL: "内见彗尾征",
    EchogenicFocus.MACROCALCIFICATION: "内见粗大钙化",
    EchogenicFocus.PERIPHERAL_CALCIFICATION: "周边见环形钙化",
    EchogenicFocus.PUNCTATE_ECHOGENIC_FOCI: "内见点状强回声",
}

_VASCULARITY_TEXT = {
    Vascularity.NONE: "CDFI示结节内及周边未见明显血流信号",
    Vascularity.PERIPHERAL: "CDFI示周边可见血流信号",
    Vascularity.INTERNAL: "CDFI示内部可见血流信号",
    Vascularity.MIXED: "CDFI示内部及周边可见血流信号",
}


@dataclass(frozen=True)
class ReportParts:
    """Canonical text fragments shared by deterministic renderers."""

    location: str
    descriptor: str
    dimensions: str | None
    shape: str | None
    margin: str | None
    echogenic_foci: str | None
    vascularity: str | None
    lymph_nodes: str | None

    @property
    def optional_nodule_clauses(self) -> tuple[str, ...]:
        return tuple(
            clause
            for clause in (
                self.dimensions,
                self.shape,
                self.margin,
                self.echogenic_foci,
                self.vascularity,
            )
            if clause
        )


def _render_location(location: Location) -> str:
    side = _SIDE_TEXT.get(location.side)
    if side is None:
        return "甲状腺内"
    if location.side == LocationSide.ISTHMUS:
        return f"甲状腺{side}"
    segment = _SEGMENT_TEXT.get(location.segment, "")
    return f"甲状腺{side}{segment}"


def _render_dimensions(dimensions: list[float] | ObservationState) -> str | None:
    if not isinstance(dimensions, list):
        return None
    text = "×".join(format(value, "g") for value in dimensions)
    return f"大小约{text}mm"


def _render_descriptor(nodule: Nodule) -> str:
    composition = _COMPOSITION_TEXT.get(nodule.composition, "")
    echogenicity = _ECHOGENICITY_TEXT.get(nodule.echogenicity, "")
    return f"{composition}{echogenicity}结节"


def build_report_parts(exam: ThyroidExam) -> ReportParts:
    """Build canonical fragments without adding or inferring observations."""

    nodule = exam.nodules[0]
    lymph_nodes = None
    if exam.lymph_nodes == LymphNodeFinding.NO_SUSPICIOUS:
        lymph_nodes = "颈部未见明显可疑淋巴结。"
    elif exam.lymph_nodes == LymphNodeFinding.SUSPICIOUS:
        lymph_nodes = "颈部可见可疑淋巴结。"

    return ReportParts(
        location=_render_location(nodule.location),
        descriptor=_render_descriptor(nodule),
        dimensions=_render_dimensions(nodule.dimensions_mm),
        shape=_SHAPE_TEXT.get(nodule.shape),
        margin=_MARGIN_TEXT.get(nodule.margin),
        echogenic_foci=_FOCUS_TEXT.get(nodule.echogenic_foci),
        vascularity=_VASCULARITY_TEXT.get(nodule.vascularity),
        lymph_nodes=lymph_nodes,
    )


def render_report(exam: ThyroidExam) -> str:
    """Render a stable report without inference or diagnostic conclusions."""

    parts = build_report_parts(exam)
    clauses = [f"{parts.location}见一枚{parts.descriptor}"]
    clauses.extend(parts.optional_nodule_clauses)
    report = "，".join(clauses) + "。"
    return report + (parts.lymph_nodes or "")
