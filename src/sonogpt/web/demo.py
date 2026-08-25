"""Assemble one demo response from the existing inference engine."""

from __future__ import annotations

from sonogpt.baselines.template_report import render_report
from sonogpt.inference.engine import InferenceEngine
from sonogpt.schemas.domain import ThyroidExam
from sonogpt.web.catalog import WEB_DEMO_VERSION


def run_demo_generate(
    engine: InferenceEngine,
    exam: ThyroidExam,
    *,
    fallback_template: bool = True,
) -> dict[str, object]:
    generated = engine.generate(exam, fallback_template=fallback_template)
    report = str(generated["report"])
    template = render_report(exam)
    extracted = engine.extract(report)
    return {
        "task": "web_demo",
        "web_demo_version": WEB_DEMO_VERSION,
        "clinical_use": False,
        "intended_use": "learning_demo_only",
        "report": report,
        "used_fallback": generated["used_fallback"],
        "fallback_reason": generated["fallback_reason"],
        "model_used": generated["model_used"],
        "eos_finished": generated["eos_finished"],
        "generation_error": generated["generation_error"],
        "qc": generated["qc"],
        "exam": generated["exam"],
        "canonical_json": generated["canonical_json"],
        "template_report": template,
        "matches_default_template": report == template,
        "extract": {
            "extractor": "rule_extract",
            "model_used": False,
            "parseable": extracted["parseable"],
            "exam": extracted["exam"],
            "qc": extracted["qc"],
        },
        "checkpoint_sha256": generated["checkpoint_sha256"],
        "freeze_record_sha256": generated["freeze_record_sha256"],
        "elapsed_seconds": generated["elapsed_seconds"],
        "info": generated["info"],
        "notes": [
            "V1 generate uses the 15M Transformer.",
            "Extract uses the rule baseline, not the model.",
            "Not for clinical diagnosis.",
        ],
    }


def run_demo_template(exam: ThyroidExam) -> dict[str, object]:
    template = render_report(exam)
    return {
        "task": "template_preview",
        "web_demo_version": WEB_DEMO_VERSION,
        "clinical_use": False,
        "report": template,
        "model_used": False,
        "exam": exam.model_dump(mode="json"),
        "notes": [
            "Deterministic template baseline, no neural network.",
            "Not for clinical diagnosis.",
        ],
    }
