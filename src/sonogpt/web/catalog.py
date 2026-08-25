"""Form labels and demo presets for the local website.

Chinese labels are for display only. Serialized values stay on Schema v1.
This catalog is not a clinical lexicon or TI-RADS mapping.
"""

from __future__ import annotations

from typing import Any

from sonogpt.schemas.domain import ThyroidExam

WEB_DEMO_VERSION = "1.0.0"


def _opt(value: str, label: str) -> dict[str, str]:
    return {"value": value, "label": label}


_NOT_MENTIONED = _opt("not_mentioned", "未提及")
_UNKNOWN = _opt("unknown", "未知")
_NOT_APPLICABLE = _opt("not_applicable", "不适用")


def _with_states(
    *options: dict[str, str], include_na: bool = False
) -> list[dict[str, str]]:
    extra = [_NOT_APPLICABLE] if include_na else []
    return [*options, _NOT_MENTIONED, _UNKNOWN, *extra]


SIDE_OPTIONS = [
    _opt("right", "右叶"),
    _opt("left", "左叶"),
    _opt("isthmus", "峡部"),
    _NOT_MENTIONED,
    _UNKNOWN,
]

SEGMENT_OPTIONS = [
    _opt("upper", "上部"),
    _opt("middle", "中部"),
    _opt("lower", "下部"),
    _NOT_MENTIONED,
    _UNKNOWN,
    _NOT_APPLICABLE,
]

DIMENSION_MODE_OPTIONS = [
    _opt("two", "两个径线（mm）"),
    _opt("three", "三个径线（mm）"),
    _NOT_MENTIONED,
    _UNKNOWN,
]

COMPOSITION_OPTIONS = _with_states(
    _opt("solid", "实性"),
    _opt("cystic", "囊性"),
    _opt("mixed_cystic_solid", "囊实性"),
    _opt("spongiform", "海绵状"),
)

ECHOGENICITY_OPTIONS = _with_states(
    _opt("anechoic", "无回声"),
    _opt("hyperechoic", "高回声"),
    _opt("isoechoic", "等回声"),
    _opt("hypoechoic", "低回声"),
    _opt("very_hypoechoic", "极低回声"),
    include_na=True,
)

SHAPE_OPTIONS = _with_states(
    _opt("wider_than_tall", "非高于宽"),
    _opt("taller_than_wide", "高于宽"),
)

MARGIN_OPTIONS = _with_states(
    _opt("smooth", "光滑"),
    _opt("ill_defined", "欠清"),
    _opt("lobulated_or_irregular", "分叶或不规则"),
    _opt("extrathyroidal_extension", "甲状腺外侵犯征象"),
)

FOCI_OPTIONS = _with_states(
    _opt("none", "明示未见"),
    _opt("comet_tail", "彗尾征"),
    _opt("macrocalcification", "粗大钙化"),
    _opt("peripheral_calcification", "环形钙化"),
    _opt("punctate_echogenic_foci", "点状强回声"),
    include_na=True,
)

VASCULARITY_OPTIONS = _with_states(
    _opt("none", "明示未见血流"),
    _opt("peripheral", "周边血流"),
    _opt("internal", "内部血流"),
    _opt("mixed", "内部及周边血流"),
    include_na=True,
)

LYMPH_OPTIONS = _with_states(
    _opt("no_suspicious", "未见可疑淋巴结"),
    _opt("suspicious", "可见可疑淋巴结"),
)

FIELDS: list[dict[str, object]] = [
    {
        "id": "side",
        "label": "部位",
        "kind": "enum",
        "options": SIDE_OPTIONS,
    },
    {
        "id": "segment",
        "label": "上 / 中 / 下",
        "kind": "enum",
        "options": SEGMENT_OPTIONS,
        "hint": "峡部会自动设为「不适用」。",
    },
    {
        "id": "dimensions_mode",
        "label": "尺寸",
        "kind": "dimensions",
        "options": DIMENSION_MODE_OPTIONS,
        "hint": "单位固定为毫米，与 Schema 一致。",
    },
    {
        "id": "composition",
        "label": "成分",
        "kind": "enum",
        "options": COMPOSITION_OPTIONS,
    },
    {
        "id": "echogenicity",
        "label": "回声",
        "kind": "enum",
        "options": ECHOGENICITY_OPTIONS,
    },
    {
        "id": "shape",
        "label": "形状",
        "kind": "enum",
        "options": SHAPE_OPTIONS,
    },
    {
        "id": "margin",
        "label": "边缘",
        "kind": "enum",
        "options": MARGIN_OPTIONS,
    },
    {
        "id": "echogenic_foci",
        "label": "强回声",
        "kind": "enum",
        "options": FOCI_OPTIONS,
        "hint": "「明示未见」不是「未提及」。",
    },
    {
        "id": "vascularity",
        "label": "血流",
        "kind": "enum",
        "options": VASCULARITY_OPTIONS,
    },
    {
        "id": "lymph_nodes",
        "label": "淋巴结",
        "kind": "enum",
        "options": LYMPH_OPTIONS,
    },
]


def _preset(
    *,
    preset_id: str,
    title: str,
    summary: str,
    structure: dict[str, Any],
) -> dict[str, Any]:
    exam = ThyroidExam.model_validate(structure)
    return {
        "id": preset_id,
        "title": title,
        "summary": summary,
        "exam": exam.model_dump(mode="json"),
    }


PRESETS: tuple[dict[str, Any], ...] = (
    _preset(
        preset_id="smoke_right_solid",
        title="冒烟样本",
        summary="右叶中部 8×6 mm 实性低回声，字段写全。CLI 冒烟用的同一条。",
        structure={
            "lymph_nodes": "no_suspicious",
            "nodules": [
                {
                    "composition": "solid",
                    "dimensions_mm": [8, 6],
                    "echogenic_foci": "none",
                    "echogenicity": "hypoechoic",
                    "location": {"segment": "middle", "side": "right"},
                    "margin": "smooth",
                    "shape": "wider_than_tall",
                    "vascularity": "none",
                }
            ],
        },
    ),
    _preset(
        preset_id="cystic_right_lower",
        title= "囊性无回声",
        summary= "右叶下部囊性结节，带彗尾征。",
        structure={
            "nodules": [
                {
                    "location": {"side": "right", "segment": "lower"},
                    "dimensions_mm": [5, 4],
                    "composition": "cystic",
                    "echogenicity": "anechoic",
                    "shape": "wider_than_tall",
                    "margin": "smooth",
                    "echogenic_foci": "comet_tail",
                    "vascularity": "none",
                }
            ]
        },
    ),
    _preset(
        preset_id="isthmus_mixed",
        title= "峡部囊实性",
        summary= "峡部必须 segment=not_applicable。",
        structure={
            "nodules": [
                {
                    "location": {"side": "isthmus", "segment": "not_applicable"},
                    "dimensions_mm": [7, 5],
                    "composition": "mixed_cystic_solid",
                    "echogenicity": "isoechoic",
                    "shape": "wider_than_tall",
                    "margin": "ill_defined",
                    "echogenic_foci": "macrocalcification",
                    "vascularity": "peripheral",
                }
            ]
        },
    ),
    _preset(
        preset_id="taller_left_upper",
        title= "高于宽 + 可疑淋巴结",
        summary= "左叶上部实性极低回声，三径线。",
        structure={
            "lymph_nodes": "suspicious",
            "nodules": [
                {
                    "location": {"side": "left", "segment": "upper"},
                    "dimensions_mm": [12, 9, 8],
                    "composition": "solid",
                    "echogenicity": "very_hypoechoic",
                    "shape": "taller_than_wide",
                    "margin": "lobulated_or_irregular",
                    "echogenic_foci": "punctate_echogenic_foci",
                    "vascularity": "internal",
                }
            ],
        },
    ),
    _preset(
        preset_id="partial_omission",
        title= "部分未提及",
        summary= "若干字段保持未提及，用来看模型会不会凭空补全。",
        structure={
            "nodules": [
                {
                    "location": {"side": "left", "segment": "not_mentioned"},
                    "dimensions_mm": [13, 10],
                    "composition": "solid",
                    "echogenicity": "hypoechoic",
                    "margin": "ill_defined",
                }
            ]
        },
    ),
)


def catalog_payload() -> dict[str, Any]:
    return {
        "web_demo_version": WEB_DEMO_VERSION,
        "schema_version": "1.0.0",
        "organ": "thyroid",
        "clinical_use": False,
        "fields": FIELDS,
        "presets": list(PRESETS),
        "notes": [
            "V1 accepts one thyroid nodule only.",
            "Extract on this page uses rules, not the 15M model.",
            "Not for clinical diagnosis.",
        ],
    }
