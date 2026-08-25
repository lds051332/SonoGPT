from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from sonogpt.inference.engine import InferenceEngine
from sonogpt.schemas.domain import ThyroidExam
from sonogpt.web.app import create_app
from sonogpt.web.catalog import PRESETS, catalog_payload
from sonogpt.web.forms import coerce_exam

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_EXAM = {
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
}


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


def _client() -> TestClient:
    return TestClient(create_app(_engine(), preload=False))


def test_presets_are_valid_schema_exams() -> None:
    assert len(PRESETS) >= 5
    for preset in PRESETS:
        exam = ThyroidExam.model_validate(preset["exam"])
        assert exam.organ == "thyroid"
        assert len(exam.nodules) == 1


def test_isthmus_segment_is_coerced() -> None:
    exam = coerce_exam(
        {
            "nodules": [
                {
                    "location": {"side": "isthmus", "segment": "upper"},
                    "dimensions_mm": [7, 5],
                }
            ]
        }
    )
    assert exam.nodules[0].location.segment.value == "not_applicable"


def test_catalog_payload_is_not_clinical() -> None:
    payload = catalog_payload()
    assert payload["clinical_use"] is False
    assert payload["schema_version"] == "1.0.0"
    assert any(field["id"] == "side" for field in payload["fields"])


def test_index_page_has_disclaimer() -> None:
    response = _client().get("/")
    assert response.status_code == 200
    assert "不能用于临床诊断" in response.text
    assert "生成报告" in response.text


def test_meta_and_health_do_not_load_checkpoint() -> None:
    engine = _engine()
    client = TestClient(create_app(engine, preload=False))
    meta = client.get("/api/meta")
    health = client.get("/api/health")
    assert meta.status_code == 200
    assert health.status_code == 200
    assert meta.json()["clinical_use"] is False
    assert health.json()["checkpoint_ready"] is False
    assert engine._model is None


def test_template_preview_does_not_use_model() -> None:
    response = _client().post("/api/template", json={"exam": SMOKE_EXAM})
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_used"] is False
    assert payload["clinical_use"] is False
    assert "右叶中部" in payload["report"]
    assert "实性低回声" in payload["report"]
    assert "8×6mm" in payload["report"]


def test_invalid_exam_returns_422() -> None:
    response = _client().post(
        "/api/template",
        json={"exam": {"nodules": []}},
    )
    assert response.status_code == 422


def test_generate_without_checkpoint_returns_503() -> None:
    response = _client().post("/api/generate", json={"exam": SMOKE_EXAM})
    assert response.status_code == 503
    assert "checkpoint" in response.json()["detail"].lower()


def test_extract_endpoint_uses_rules() -> None:
    response = _client().post(
        "/api/extract",
        json={"report": "甲状腺右叶中部见一枚实性低回声结节，大小约8×6mm。"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_used"] is False
    assert payload["extractor"] == "rule_extract"
    assert payload["parseable"] is True
    assert payload["clinical_use"] is False
