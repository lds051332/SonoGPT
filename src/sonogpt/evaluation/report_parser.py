"""Independent rule parser for generated thyroid reports.

This parser is the evaluation extractor. It is not the model under test, and it
does not infer diagnoses. Unmentioned observations stay `not_mentioned`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TypeVar

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

EnumT = TypeVar("EnumT")

_DIMENSION_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*[×xX*]\s*(\d+(?:\.\d+)?)"
    r"(?:\s*(?:mm)?)(?:\s*[×xX*]\s*(\d+(?:\.\d+)?))?\s*(mm|cm)?"
)


@dataclass(frozen=True)
class ParsedReport:
    exam: ThyroidExam
    explicit_field_count: int
    parseable: bool
    notes: tuple[str, ...] = ()


def _first_enum(
    text: str,
    rules: tuple[tuple[EnumT, tuple[str, ...]], ...],
    default: EnumT,
) -> EnumT:
    for value, phrases in rules:
        if any(phrase in text for phrase in phrases):
            return value
    return default


def _extract_dimensions(text: str) -> list[float] | ObservationState:
    if any(
        marker in text
        for marker in (
            "未给出结节径线",
            "具体大小无法",
            "大小无法获得",
            "径线及纵横形态",
        )
    ):
        return ObservationState.NOT_MENTIONED
    match = _DIMENSION_RE.search(text)
    if match is None:
        return ObservationState.NOT_MENTIONED
    values = [float(match.group(1)), float(match.group(2))]
    if match.group(3):
        values.append(float(match.group(3)))
    unit = match.group(4)
    trailing = text[match.end() : match.end() + 4]
    if unit == "cm" or (unit is None and trailing.startswith("cm")):
        values = [value * 10.0 for value in values]
    return values


def _parse_location(text: str) -> Location:
    if "峡部" in text:
        return Location(side=LocationSide.ISTHMUS, segment=LocationSegment.NOT_APPLICABLE)
    side = _first_enum(
        text,
        (
            (
                LocationSide.LEFT,
                ("左叶", "左侧叶", "左侧甲状腺"),
            ),
            (
                LocationSide.RIGHT,
                ("右叶", "右侧叶", "右侧甲状腺"),
            ),
        ),
        LocationSide.NOT_MENTIONED,
    )
    segment = _first_enum(
        text,
        (
            (LocationSegment.UPPER, ("上部", "上段")),
            (LocationSegment.MIDDLE, ("中部", "中段")),
            (LocationSegment.LOWER, ("下部", "下段")),
        ),
        LocationSegment.NOT_MENTIONED,
    )
    return Location(side=side, segment=segment)


def parse_report(text: str) -> ParsedReport:
    """Map a Chinese report onto Schema v1 without using the neural model."""

    notes: list[str] = []
    stripped = text.strip()
    if not stripped:
        notes.append("empty_report")
        exam = ThyroidExam(
            nodules=[
                Nodule(location=Location(side=LocationSide.NOT_MENTIONED))
            ]
        )
        return ParsedReport(exam=exam, explicit_field_count=0, parseable=False, notes=tuple(notes))

    location = _parse_location(stripped)
    composition = _first_enum(
        stripped,
        (
            (
                Composition.MIXED_CYSTIC_SOLID,
                (
                    "囊实性",
                    "囊实混合",
                    "囊实相间",
                    "囊、实",
                    "囊性与实性",
                    "囊性与实性部分并存",
                ),
            ),
            (Composition.SPONGIFORM, ("海绵状", "海绵样")),
            (Composition.CYSTIC, ("囊性", "液性")),
            (Composition.SOLID, ("实性",)),
        ),
        Composition.NOT_MENTIONED,
    )
    echogenicity = _first_enum(
        stripped,
        (
            (Echogenicity.VERY_HYPOECHOIC, ("极低回声",)),
            (Echogenicity.HYPOECHOIC, ("低回声", "回声偏低")),
            (Echogenicity.HYPERECHOIC, ("高回声",)),
            (Echogenicity.ISOECHOIC, ("等回声", "回声与周围腺体接近")),
            (Echogenicity.ANECHOIC, ("无回声",)),
        ),
        Echogenicity.NOT_MENTIONED,
    )
    shape = _first_enum(
        stripped,
        (
            (
                Shape.WIDER_THAN_TALL,
                (
                    "非高于宽",
                    "横径大于前后径",
                    "横径较前后径大",
                    "横径略大于",
                    "横径超过前后径",
                    "横向径大于前后径",
                    "横向生长",
                    "横向排列",
                    "横向径占优",
                    "横向径较大",
                    "横径较大",
                    "前后径小于横径",
                    "前后径未超过横径",
                ),
            ),
            (
                Shape.TALLER_THAN_WIDE,
                (
                    "高于宽",
                    "前后径超过横径",
                    "前后径大于横径",
                ),
            ),
        ),
        Shape.NOT_MENTIONED,
    )
    margin = _first_enum(
        stripped,
        (
            (
                Margin.EXTRATHYROIDAL_EXTENSION,
                (
                    "甲状腺外侵犯",
                    "越出甲状腺轮廓",
                    "延伸至腺体轮廓之外",
                ),
            ),
            (
                Margin.LOBULATED_OR_IRREGULAR,
                (
                    "分叶或不规则",
                    "分叶且不规则",
                    "分叶样不规则",
                    "分叶、不规则",
                    "分叶不规则",
                    "外形欠规则",
                    "轮廓分叶",
                    "外缘呈分叶",
                ),
            ),
            (
                Margin.ILL_DEFINED,
                (
                    "边界欠清",
                    "边界显示欠清",
                    "边缘分辨欠佳",
                    "边缘显示不够清楚",
                ),
            ),
            (
                Margin.SMOOTH,
                (
                    "边缘光滑",
                    "轮廓光整",
                    "壁缘规整",
                    "边缘连续光滑",
                    "边缘尚光整",
                    "边缘平整",
                    "轮廓平滑",
                    "外缘平滑",
                    "轮廓规整",
                    "边缘规则",
                    "边缘光整",
                    "边缘平滑",
                    "边缘仍光滑",
                ),
            ),
        ),
        Margin.NOT_MENTIONED,
    )
    echogenic_foci = _first_enum(
        stripped,
        (
            (
                EchogenicFocus.NONE,
                (
                    "未见明显强回声",
                    "没有明确强回声",
                    "未检出强回声",
                    "未发现明确强回声",
                    "未见内部强回声",
                    "未见明确强回声",
                    "内部未发现强回声",
                    "内部没有明确强回声",
                    "内部未见强回声",
                    "未见强回声灶",
                    "未发现强回声灶",
                ),
            ),
            (
                EchogenicFocus.PUNCTATE_ECHOGENIC_FOCI,
                ("点状强回声", "细小点状", "细点状"),
            ),
            (EchogenicFocus.COMET_TAIL, ("彗尾",)),
            (
                EchogenicFocus.MACROCALCIFICATION,
                ("粗大钙化", "较大钙化"),
            ),
            (
                EchogenicFocus.PERIPHERAL_CALCIFICATION,
                (
                    "环形钙化",
                    "周边见环形",
                    "周缘伴钙化",
                    "周边伴钙化",
                ),
            ),
        ),
        EchogenicFocus.NOT_MENTIONED,
    )
    vascularity = _first_enum(
        stripped,
        (
            (
                Vascularity.NONE,
                (
                    "结节内及周边未见明显血流",
                    "未见明显血流信号",
                    "未记录到明确血流",
                    "结节内外未见明确信号",
                    "均无明确显示",
                    "结节内外无明确血流",
                    "内外均无明显血流",
                    "彩色血流亦未显示",
                    "未显示明显血流",
                    "无明确血流",
                ),
            ),
            (
                Vascularity.MIXED,
                (
                    "内部及周边可见血流",
                    "结节内和周围均可见血流",
                    "内部与周边均有",
                    "内部和外围都有血流",
                    "结节内外均显示血流",
                ),
            ),
            (
                Vascularity.INTERNAL,
                (
                    "内部可见血流信号",
                    "进入结节内部",
                    "可记录到内部血流",
                    "点状强回声及血流",
                    "点状强回声与血流",
                    "强回声和血流信号",
                    "强回声与血流信号",
                    "内部有血流",
                    "内部血流",
                    "并可记录到血流",
                ),
            ),
            (
                Vascularity.PERIPHERAL,
                (
                    "周边可见血流信号",
                    "环绕结节",
                    "周缘可记录到血流",
                    "血流仅在外围",
                    "血流环绕",
                    "环周血流",
                    "周边有少量血流",
                    "血流主要位于结节周围",
                    "周边可见血流",
                ),
            ),
        ),
        Vascularity.NOT_MENTIONED,
    )
    lymph_nodes = _first_enum(
        stripped,
        (
            (
                LymphNodeFinding.NO_SUSPICIOUS,
                (
                    "未见明显可疑淋巴结",
                    "未扫及明显异常淋巴结",
                    "未见可疑改变",
                    "未发现形态可疑",
                    "未扫及可疑淋巴结",
                    "未发现明显可疑淋巴结",
                    "未发现可疑淋巴结",
                    "未见可疑淋巴结",
                ),
            ),
            (
                LymphNodeFinding.SUSPICIOUS,
                (
                    "可见可疑淋巴结",
                    "形态可疑淋巴结",
                    "另见形态可疑",
                    "同时发现可疑淋巴结",
                    "另有可疑淋巴结",
                    "颈部见形态可疑",
                ),
            ),
        ),
        LymphNodeFinding.NOT_MENTIONED,
    )
    dimensions = _extract_dimensions(stripped)
    exam = ThyroidExam(
        nodules=[
            Nodule(
                location=location,
                dimensions_mm=dimensions,
                composition=composition,
                echogenicity=echogenicity,
                shape=shape,
                margin=margin,
                echogenic_foci=echogenic_foci,
                vascularity=vascularity,
            )
        ],
        lymph_nodes=lymph_nodes,
    )
    explicit_field_count = count_explicit_fields(exam)
    parseable = bool(
        explicit_field_count
        or "甲状腺" in stripped
        or "结节" in stripped
        or "峡部" in stripped
    )
    return ParsedReport(
        exam=exam,
        explicit_field_count=explicit_field_count,
        parseable=parseable,
        notes=tuple(notes),
    )


def count_explicit_fields(exam: ThyroidExam) -> int:
    """Count schema fields that name an actual observation rather than absence."""

    nodule = exam.nodules[0]
    values: list[object] = [
        nodule.location.side,
        nodule.composition,
        nodule.echogenicity,
        nodule.shape,
        nodule.margin,
        nodule.echogenic_foci,
        nodule.vascularity,
        exam.lymph_nodes,
    ]
    count = 0
    if nodule.location.side == LocationSide.ISTHMUS:
        count += 1
    elif is_stated_value(nodule.location.side):
        count += 1
        if is_stated_value(nodule.location.segment, allow_not_applicable=False):
            count += 1
    elif is_stated_value(nodule.location.segment, allow_not_applicable=False):
        count += 1
    if isinstance(nodule.dimensions_mm, list):
        count += 1
    count += sum(1 for value in values[1:] if is_stated_value(value))
    return count


def is_stated_value(value: object, *, allow_not_applicable: bool = False) -> bool:
    if isinstance(value, list):
        return True
    raw = value.value if hasattr(value, "value") else value
    if allow_not_applicable and raw == "not_applicable":
        return True
    return raw not in {"not_mentioned", "unknown", "not_applicable"}
