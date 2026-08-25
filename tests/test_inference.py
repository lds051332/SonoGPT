from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from sonogpt.baselines.template_report import render_report
from sonogpt.inference.engine import InferenceEngine
from sonogpt.schemas.domain import ThyroidExam

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _engine() -> InferenceEngine:
    return InferenceEngine(
        project_root=PROJECT_ROOT,
        freeze_record_path=(
            PROJECT_ROOT
            / "data"
            / "releases"
            / "synthetic_v1_5k_frozen_v1.freeze.json"
        ),
        checkpoint_path=(
            PROJECT_ROOT
            / "artifacts"
            / "training"
            / "sonogpt_16m_m3"
            / "missing.pt"
        ),
        device_name="cpu",
    )


def test_infer_extract_and_qc_do_not_require_checkpoint() -> None:
    exam = ThyroidExam.model_validate(
        {
            "nodules": [
                {
                    "location": {"side": "right", "segment": "middle"},
                    "dimensions_mm": [8, 6],
                    "composition": "solid",
                    "echogenicity": "hypoechoic",
                    "shape": "wider_than_tall",
                    "margin": "smooth",
                    "echogenic_foci": "none",
                    "vascularity": "none",
                }
            ],
            "lymph_nodes": "no_suspicious",
        }
    )
    report = render_report(exam)
    engine = _engine()
    extracted = engine.extract(report)
    qc = engine.qc(report, exam)
    info = engine.info(load_model=False)

    assert extracted["model_used"] is False
    assert extracted["parseable"] is True
    assert extracted["exam"]["nodules"][0]["composition"] == "solid"
    assert qc["qc"]["passed"] is True
    assert info.extract_task_trained is False
    assert info.generate_task_trained is True
    assert info.intended_use == "learning_demo_only"
    assert engine._model is None


def test_infer_cli_extract_writes_output(tmp_path: Path) -> None:
    output = tmp_path / "extract.json"
    report = "甲状腺右叶中部见一枚实性低回声结节，大小约8×6mm。"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "infer.py"),
            "extract",
            "--device",
            "cpu",
            "--text",
            report,
            "--output",
            str(output),
            "--checkpoint",
            str(
                PROJECT_ROOT
                / "artifacts"
                / "training"
                / "sonogpt_16m_m3"
                / "missing.pt"
            ),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    stdout = json.loads(result.stdout)
    assert payload == stdout
    assert payload["task"] == "extract"
    assert payload["model_used"] is False
    assert payload["parseable"] is True

