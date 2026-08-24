"""Build a fixed AI-simulated challenge set without training renderers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sonogpt.evaluation.challenge import (
    ChallengeSample,
    verify_challenge_freeze,
    write_frozen_challenge_set,
)
from sonogpt.schemas.domain import ThyroidExam

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHALLENGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "challenges"
    / "simulated_human_challenge_v1.jsonl"
)
DEFAULT_FREEZE_PATH = (
    PROJECT_ROOT
    / "data"
    / "releases"
    / "simulated_human_challenge_v1.freeze.json"
)
DEFAULT_TRAINING_FREEZE = (
    PROJECT_ROOT
    / "data"
    / "releases"
    / "synthetic_v1_5k_frozen_v1.freeze.json"
)


def _sample(
    structure: dict[str, object],
    reference_report: str,
    *difficulty_tags: str,
) -> ChallengeSample:
    return ChallengeSample.from_exam(
        ThyroidExam.model_validate(structure),
        reference_report=reference_report,
        difficulty_tags=tuple(difficulty_tags),
    )


def challenge_samples() -> tuple[ChallengeSample, ...]:
    """Return manually curated, fixed cases with non-template references."""

    return (
        _sample(
            {
                "nodules": [
                    {
                        "location": {"side": "right", "segment": "upper"},
                        "dimensions_mm": [9, 6, 12],
                        "composition": "solid",
                        "echogenicity": "hypoechoic",
                        "shape": "wider_than_tall",
                        "margin": "smooth",
                        "echogenic_foci": "none",
                        "vascularity": "peripheral",
                    }
                ],
                "lymph_nodes": "no_suspicious",
            },
            "结节位于甲状腺右叶上段，测得约9×6×12mm；呈实性低回声，横径大于前后径，轮廓光整。其内未检出强回声点，彩色血流主要环绕结节分布。双侧颈区未扫及明显异常淋巴结。",
            "lexical_paraphrase",
            "three_dimensions",
            "unseen_word_order",
        ),
        _sample(
            {
                "nodules": [
                    {
                        "location": {"side": "left", "segment": "lower"},
                        "dimensions_mm": [6, 9],
                        "composition": "solid",
                        "echogenicity": "very_hypoechoic",
                        "shape": "taller_than_wide",
                        "margin": "lobulated_or_irregular",
                        "echogenic_foci": "punctate_echogenic_foci",
                        "vascularity": "internal",
                    }
                ],
                "lymph_nodes": "suspicious",
            },
            "甲状腺左叶下段探及6×9mm结节，前后径超过横径。结节为实性极低回声，外形欠规则，并散在细小点状强回声；血流信号可进入结节内部。颈部另见形态可疑淋巴结。",
            "suspicious_features",
            "lymph_node_statement",
            "unseen_word_order",
        ),
        _sample(
            {
                "nodules": [
                    {
                        "location": {
                            "side": "isthmus",
                            "segment": "not_applicable",
                        },
                        "dimensions_mm": [11, 5, 13],
                        "composition": "mixed_cystic_solid",
                        "echogenicity": "isoechoic",
                        "shape": "wider_than_tall",
                        "margin": "ill_defined",
                        "echogenic_foci": "macrocalcification",
                        "vascularity": "peripheral",
                    }
                ]
            },
            "于峡部发现囊、实成分混合的等回声灶，三径约11×5×13mm，横向生长，边缘显示不够清楚。灶内伴较大钙化回声，周缘可记录到血流。",
            "isthmus",
            "lexical_paraphrase",
            "three_dimensions",
        ),
        _sample(
            {
                "nodules": [
                    {
                        "location": {"side": "right", "segment": "middle"},
                        "dimensions_mm": [7, 4],
                        "composition": "cystic",
                        "echogenicity": "anechoic",
                        "shape": "wider_than_tall",
                        "margin": "smooth",
                        "echogenic_foci": "comet_tail",
                        "vascularity": "none",
                    }
                ],
                "lymph_nodes": "no_suspicious",
            },
            "右侧叶中段可扫及一处7mm×4mm液性无回声区，前后径小于横径，壁缘规整，局部伴彗尾样回声。彩色多普勒在其内及周边均未记录到明确血流；颈部淋巴结未见可疑改变。",
            "benign_pattern",
            "explicit_negative_flow",
            "lexical_paraphrase",
        ),
        _sample(
            {
                "nodules": [
                    {
                        "location": {"side": "left", "segment": "upper"},
                        "dimensions_mm": [13, 7],
                        "composition": "spongiform",
                        "echogenicity": "isoechoic",
                        "shape": "wider_than_tall",
                        "margin": "smooth",
                        "echogenic_foci": "none",
                        "vascularity": "peripheral",
                    }
                ]
            },
            "左叶上段有一枚约13×7mm的海绵样等回声结节，整体横向排列，边缘连续光滑，内部没有明确强回声灶，血流仅在外围显示。",
            "benign_pattern",
            "no_lymph_node_statement",
            "unseen_word_order",
        ),
        _sample(
            {
                "nodules": [
                    {
                        "location": {"side": "right", "segment": "lower"},
                        "dimensions_mm": [15, 8, 17],
                        "composition": "mixed_cystic_solid",
                        "echogenicity": "hypoechoic",
                        "shape": "wider_than_tall",
                        "margin": "smooth",
                        "echogenic_foci": "macrocalcification",
                        "vascularity": "mixed",
                    }
                ],
                "lymph_nodes": "no_suspicious",
            },
            "右叶下段结节大小为15×8×17mm，囊实相间且回声偏低，前后径未超过横径，边缘尚光整。内部可见粗大钙化，结节内和周围均可见血流。颈区未发现形态可疑淋巴结。",
            "three_dimensions",
            "lexical_paraphrase",
            "lymph_node_statement",
        ),
        _sample(
            {
                "nodules": [
                    {
                        "location": {"side": "left", "segment": "middle"},
                        "dimensions_mm": [8, 5],
                        "composition": "solid",
                        "echogenicity": "hyperechoic",
                        "shape": "wider_than_tall",
                        "margin": "ill_defined",
                        "echogenic_foci": "peripheral_calcification",
                        "vascularity": "none",
                    }
                ]
            },
            "左侧叶中段见8×5mm实性高回声结节，呈横向生长，边缘分辨欠佳，周边见环形钙化样强回声。彩色血流检查在结节内外未见明确信号。",
            "peripheral_calcification",
            "explicit_negative_flow",
            "lexical_paraphrase",
        ),
        _sample(
            {
                "nodules": [
                    {
                        "location": {"side": "right", "segment": "upper"},
                        "dimensions_mm": [10.5, 6.5, 14],
                        "composition": "solid",
                        "echogenicity": "isoechoic",
                        "shape": "wider_than_tall",
                        "margin": "smooth",
                        "echogenic_foci": "comet_tail",
                        "vascularity": "mixed",
                    }
                ]
            },
            "在右叶上段扫查到约10.5×6.5×14mm实性结节，回声与周围腺体接近，横径较前后径大，边缘平整。结节内有彗尾样强回声，内部与周边均有血流显示。",
            "decimal_dimensions",
            "three_dimensions",
            "unseen_word_order",
        ),
        _sample(
            {
                "nodules": [
                    {
                        "location": {"side": "left", "segment": "lower"},
                        "dimensions_mm": [7, 10],
                        "composition": "mixed_cystic_solid",
                        "echogenicity": "very_hypoechoic",
                        "shape": "taller_than_wide",
                        "margin": "extrathyroidal_extension",
                        "echogenic_foci": "punctate_echogenic_foci",
                        "vascularity": "internal",
                    }
                ],
                "lymph_nodes": "suspicious",
            },
            "左叶下段可见囊实性极低回声结节，约7×10mm，前后径大于横径；边缘局部越出甲状腺轮廓，内部见点状强回声及血流。颈侧区同时发现可疑淋巴结。",
            "suspicious_features",
            "extrathyroidal_extension",
            "lymph_node_statement",
        ),
        _sample(
            {
                "nodules": [
                    {
                        "location": {"side": "unknown", "segment": "unknown"},
                        "dimensions_mm": "unknown",
                        "composition": "unknown",
                        "echogenicity": "unknown",
                        "shape": "unknown",
                        "margin": "unknown",
                        "echogenic_foci": "unknown",
                        "vascularity": "unknown",
                    }
                ],
                "lymph_nodes": "unknown",
            },
            "受显示条件限制，结节的具体侧别、大小、内部成分、回声、形态、边缘、强回声灶及血流情况均无法可靠判定，颈部淋巴结情况亦不能明确。",
            "unknown_state",
            "missing_measurement",
            "adverse_display_condition",
        ),
        _sample(
            {
                "nodules": [
                    {
                        "location": {
                            "side": "right",
                            "segment": "not_mentioned",
                        },
                        "dimensions_mm": [6, 4],
                        "composition": "not_mentioned",
                        "echogenicity": "hypoechoic",
                        "shape": "not_mentioned",
                        "margin": "smooth",
                        "echogenic_foci": "none",
                        "vascularity": "not_mentioned",
                    }
                ]
            },
            "右侧甲状腺内见一枚6×4mm低回声结节，边缘平滑，内部未发现明确强回声；原始信息未描述其具体叶段、成分、纵横形态和血流。",
            "not_mentioned_state",
            "partial_description",
            "missing_fields",
        ),
        _sample(
            {
                "nodules": [
                    {
                        "location": {"side": "left", "segment": "upper"},
                        "dimensions_mm": "not_mentioned",
                        "composition": "solid",
                        "echogenicity": "hypoechoic",
                        "shape": "not_mentioned",
                        "margin": "lobulated_or_irregular",
                        "echogenic_foci": "punctate_echogenic_foci",
                        "vascularity": "internal",
                    }
                ],
                "lymph_nodes": "unknown",
            },
            "左叶上段实性低回声结节，边缘呈分叶样不规则，内有细点状强回声和血流信号。资料未给出结节径线及纵横形态，颈部淋巴结状态无法判断。",
            "missing_measurement",
            "partial_description",
            "unknown_state",
        ),
        _sample(
            {
                "nodules": [
                    {
                        "location": {
                            "side": "isthmus",
                            "segment": "not_applicable",
                        },
                        "dimensions_mm": [5.5, 3.5],
                        "composition": "cystic",
                        "echogenicity": "anechoic",
                        "shape": "wider_than_tall",
                        "margin": "smooth",
                        "echogenic_foci": "none",
                        "vascularity": "none",
                    }
                ]
            },
            "峡部扫及5.5×3.5mm小囊性无回声结节，横径较大，边缘规则，未见内部强回声；彩色血流在结节内及其周围均无明确显示。",
            "isthmus",
            "decimal_dimensions",
            "explicit_negative_flow",
        ),
        _sample(
            {
                "nodules": [
                    {
                        "location": {"side": "right", "segment": "middle"},
                        "dimensions_mm": [18, 9, 20],
                        "composition": "spongiform",
                        "echogenicity": "hyperechoic",
                        "shape": "wider_than_tall",
                        "margin": "ill_defined",
                        "echogenic_foci": "comet_tail",
                        "vascularity": "peripheral",
                    }
                ],
                "lymph_nodes": "no_suspicious",
            },
            "甲状腺右叶中段海绵样高回声结节约18×9×20mm，前后径小于横径，边界显示欠清，其内见彗尾征，血流环绕分布。颈部未扫及可疑淋巴结。",
            "benign_pattern",
            "three_dimensions",
            "lymph_node_statement",
        ),
        _sample(
            {
                "nodules": [
                    {
                        "location": {"side": "left", "segment": "upper"},
                        "dimensions_mm": [4, 6, 7],
                        "composition": "solid",
                        "echogenicity": "very_hypoechoic",
                        "shape": "taller_than_wide",
                        "margin": "smooth",
                        "echogenic_foci": "none",
                        "vascularity": "internal",
                    }
                ]
            },
            "一枚4×6×7mm实性极低回声结节位于左叶上段，前后径超过横径，但边缘仍光滑；结节内未见明确强回声，可记录到内部血流。",
            "taller_than_wide",
            "three_dimensions",
            "unseen_word_order",
        ),
        _sample(
            {
                "nodules": [
                    {
                        "location": {"side": "right", "segment": "lower"},
                        "dimensions_mm": [20, 11, 24],
                        "composition": "solid",
                        "echogenicity": "hypoechoic",
                        "shape": "wider_than_tall",
                        "margin": "extrathyroidal_extension",
                        "echogenic_foci": "macrocalcification",
                        "vascularity": "mixed",
                    }
                ],
                "lymph_nodes": "suspicious",
            },
            "右叶下段实性低回声结节三径约20×11×24mm，横向径占优，局部边缘延伸至腺体轮廓之外，内部见粗大钙化。结节内外均显示血流，颈区另有可疑淋巴结。",
            "suspicious_features",
            "extrathyroidal_extension",
            "three_dimensions",
        ),
        _sample(
            {
                "nodules": [
                    {
                        "location": {"side": "left", "segment": "middle"},
                        "dimensions_mm": [9, 5],
                        "composition": "mixed_cystic_solid",
                        "echogenicity": "isoechoic",
                        "shape": "wider_than_tall",
                        "margin": "smooth",
                        "echogenic_foci": "peripheral_calcification",
                        "vascularity": "peripheral",
                    }
                ]
            },
            "左叶中段有9×5mm囊实混合性等回声结节，呈横向生长，边缘光整，结节周缘伴钙化样强回声及环周血流。",
            "peripheral_calcification",
            "lexical_paraphrase",
            "unseen_word_order",
        ),
        _sample(
            {
                "nodules": [
                    {
                        "location": {"side": "right", "segment": "upper"},
                        "dimensions_mm": [6, 3],
                        "composition": "not_mentioned",
                        "echogenicity": "anechoic",
                        "shape": "wider_than_tall",
                        "margin": "smooth",
                        "echogenic_foci": "comet_tail",
                        "vascularity": "none",
                    }
                ]
            },
            "右叶上段见6×3mm无回声结节，横径大于前后径，轮廓平滑，伴彗尾样回声，内外均无明显血流。资料没有说明该结节的成分分类。",
            "not_mentioned_state",
            "explicit_negative_flow",
            "partial_description",
        ),
        _sample(
            {
                "nodules": [
                    {
                        "location": {"side": "left", "segment": "lower"},
                        "dimensions_mm": [12, 7],
                        "composition": "solid",
                        "echogenicity": "unknown",
                        "shape": "unknown",
                        "margin": "ill_defined",
                        "echogenic_foci": "unknown",
                        "vascularity": "unknown",
                    }
                ]
            },
            "左叶下段见12×7mm实性结节，边界欠清；其回声水平、纵横形态、强回声灶以及血流情况受限，均不能可靠判断。",
            "unknown_state",
            "partial_description",
            "adverse_display_condition",
        ),
        _sample(
            {
                "nodules": [
                    {
                        "location": {
                            "side": "isthmus",
                            "segment": "not_applicable",
                        },
                        "dimensions_mm": [8, 5],
                        "composition": "solid",
                        "echogenicity": "hypoechoic",
                        "shape": "wider_than_tall",
                        "margin": "lobulated_or_irregular",
                        "echogenic_foci": "punctate_echogenic_foci",
                        "vascularity": "internal",
                    }
                ],
                "lymph_nodes": "no_suspicious",
            },
            "峡部8×5mm实性低回声结节，横向径较大，外缘呈分叶、不规则改变，内部可见点状强回声与血流信号。颈部未发现明显可疑淋巴结。",
            "isthmus",
            "suspicious_features",
            "lymph_node_statement",
        ),
        _sample(
            {
                "nodules": [
                    {
                        "location": {"side": "right", "segment": "middle"},
                        "dimensions_mm": [14, 8],
                        "composition": "mixed_cystic_solid",
                        "echogenicity": "hyperechoic",
                        "shape": "wider_than_tall",
                        "margin": "smooth",
                        "echogenic_foci": "comet_tail",
                        "vascularity": "mixed",
                    }
                ]
            },
            "右叶中段结节约14mm×8mm，囊性与实性部分并存，整体呈高回声，前后径小于横径且边缘规则。可见彗尾样回声，结节内部和外围都有血流。",
            "lexical_paraphrase",
            "unseen_word_order",
            "no_lymph_node_statement",
        ),
        _sample(
            {
                "nodules": [
                    {
                        "location": {"side": "left", "segment": "upper"},
                        "dimensions_mm": [9, 4],
                        "composition": "cystic",
                        "echogenicity": "unknown",
                        "shape": "unknown",
                        "margin": "unknown",
                        "echogenic_foci": "unknown",
                        "vascularity": "unknown",
                    }
                ]
            },
            "左叶上段有一枚9×4mm囊性结节；受图像条件影响，其内部回声、纵横形态、边缘、强回声灶和血流特征均无法进一步确认。",
            "unknown_state",
            "benign_pattern",
            "adverse_display_condition",
        ),
        _sample(
            {
                "nodules": [
                    {
                        "location": {"side": "right", "segment": "lower"},
                        "dimensions_mm": "unknown",
                        "composition": "solid",
                        "echogenicity": "isoechoic",
                        "shape": "wider_than_tall",
                        "margin": "smooth",
                        "echogenic_foci": "none",
                        "vascularity": "none",
                    }
                ]
            },
            "右叶下段可见实性等回声结节，横径大于前后径，边缘平整，内部未见强回声，彩色血流亦未显示；结节具体大小无法获得。",
            "missing_measurement",
            "explicit_negative_flow",
            "partial_description",
        ),
        _sample(
            {
                "nodules": [
                    {
                        "location": {"side": "left", "segment": "middle"},
                        "dimensions_mm": [3.6, 3.2],
                        "composition": "solid",
                        "echogenicity": "hypoechoic",
                        "shape": "wider_than_tall",
                        "margin": "ill_defined",
                        "echogenic_foci": "none",
                        "vascularity": "peripheral",
                    }
                ]
            },
            "左叶中段见约3.6×3.2mm实性低回声小结节，横径略大于前后径，边界显示欠清，未见明确强回声，周边有少量血流显示。",
            "decimal_dimensions",
            "small_nodule",
            "lexical_paraphrase",
        ),
        _sample(
            {
                "nodules": [
                    {
                        "location": {"side": "right", "segment": "upper"},
                        "dimensions_mm": [25, 12, 30],
                        "composition": "mixed_cystic_solid",
                        "echogenicity": "hypoechoic",
                        "shape": "wider_than_tall",
                        "margin": "lobulated_or_irregular",
                        "echogenic_foci": "macrocalcification",
                        "vascularity": "internal",
                    }
                ]
            },
            "右叶上段较大囊实性低回声结节，约25×12×30mm，横向径大于前后径，轮廓分叶且不规则。内部伴粗大钙化，并可记录到血流。",
            "large_nodule",
            "three_dimensions",
            "suspicious_features",
        ),
        _sample(
            {
                "nodules": [
                    {
                        "location": {"side": "left", "segment": "lower"},
                        "dimensions_mm": [16, 8],
                        "composition": "spongiform",
                        "echogenicity": "hypoechoic",
                        "shape": "wider_than_tall",
                        "margin": "smooth",
                        "echogenic_foci": "comet_tail",
                        "vascularity": "none",
                    }
                ],
                "lymph_nodes": "no_suspicious",
            },
            "左叶下段16×8mm海绵样低回声结节，呈横向排列，外缘平滑，内见彗尾征，结节内外无明确血流。颈部未见可疑淋巴结表现。",
            "benign_pattern",
            "explicit_negative_flow",
            "lymph_node_statement",
        ),
        _sample(
            {
                "nodules": [
                    {
                        "location": {
                            "side": "isthmus",
                            "segment": "not_applicable",
                        },
                        "dimensions_mm": [10, 6],
                        "composition": "mixed_cystic_solid",
                        "echogenicity": "unknown",
                        "shape": "wider_than_tall",
                        "margin": "unknown",
                        "echogenic_foci": "macrocalcification",
                        "vascularity": "peripheral",
                    }
                ]
            },
            "峡部囊实性结节约10×6mm，横径较大，内部回声水平及边缘情况不能明确；可见粗大钙化，血流主要位于结节周围。",
            "isthmus",
            "unknown_state",
            "partial_description",
        ),
        _sample(
            {
                "nodules": [
                    {
                        "location": {"side": "right", "segment": "middle"},
                        "dimensions_mm": [5, 7],
                        "composition": "solid",
                        "echogenicity": "very_hypoechoic",
                        "shape": "taller_than_wide",
                        "margin": "lobulated_or_irregular",
                        "echogenic_foci": "peripheral_calcification",
                        "vascularity": "internal",
                    }
                ],
                "lymph_nodes": "suspicious",
            },
            "右叶中段5×7mm实性极低回声结节，前后径大于横径，边缘分叶不规则，周边伴钙化强回声，内部有血流。颈部见形态可疑淋巴结。",
            "taller_than_wide",
            "peripheral_calcification",
            "suspicious_features",
        ),
        _sample(
            {
                "nodules": [
                    {
                        "location": {"side": "left", "segment": "upper"},
                        "dimensions_mm": [11, 6],
                        "composition": "unknown",
                        "echogenicity": "hypoechoic",
                        "shape": "wider_than_tall",
                        "margin": "smooth",
                        "echogenic_foci": "punctate_echogenic_foci",
                        "vascularity": "none",
                    }
                ]
            },
            "左叶上段11×6mm低回声结节，具体成分无法确定；结节横径较大、边缘平滑，内部散在点状强回声，但未显示明显血流。",
            "unknown_state",
            "explicit_negative_flow",
            "partial_description",
        ),
        _sample(
            {
                "nodules": [
                    {
                        "location": {"side": "right", "segment": "lower"},
                        "dimensions_mm": [7, 4, 9],
                        "composition": "solid",
                        "echogenicity": "hyperechoic",
                        "shape": "wider_than_tall",
                        "margin": "smooth",
                        "echogenic_foci": "none",
                        "vascularity": "peripheral",
                    }
                ],
                "lymph_nodes": "no_suspicious",
            },
            "甲状腺右叶下段实性高回声结节，大小7×4×9mm，横径超过前后径，轮廓规整，内部未发现强回声灶，周边可见血流。颈部未发现可疑淋巴结。",
            "three_dimensions",
            "lexical_paraphrase",
            "lymph_node_statement",
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--challenge-path",
        type=Path,
        default=DEFAULT_CHALLENGE_PATH,
    )
    parser.add_argument(
        "--freeze-record",
        type=Path,
        default=DEFAULT_FREEZE_PATH,
    )
    parser.add_argument(
        "--training-freeze-record",
        type=Path,
        default=DEFAULT_TRAINING_FREEZE,
    )
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify:
        record = verify_challenge_freeze(
            args.freeze_record,
            project_root=PROJECT_ROOT,
        )
    else:
        record = write_frozen_challenge_set(
            challenge_samples(),
            challenge_path=args.challenge_path,
            freeze_record_path=args.freeze_record,
            training_freeze_record=args.training_freeze_record,
            project_root=PROJECT_ROOT,
            builder_path=Path(__file__),
        )
    print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
