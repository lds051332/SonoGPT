"""Local CLI inference for generate, rule extract, and QC.

V1 only trained the generate task. Extract uses the independent rule baseline,
not the Transformer. QC is a versioned rule engine, not a diagnostic system.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from sonogpt import __version__ as PACKAGE_VERSION
from sonogpt.baselines.qc import QC_RULES_VERSION, QcResult, run_qc
from sonogpt.baselines.rule_extract import RULE_EXTRACT_VERSION, extract_exam
from sonogpt.baselines.template_report import render_report
from sonogpt.data.freeze import verify_freeze_record
from sonogpt.data.manifest import sha256_file
from sonogpt.data.semantic_generator import canonical_exam_json
from sonogpt.evaluation.generation import generate_report
from sonogpt.evaluation.pipeline import load_model_for_evaluation, resolve_device
from sonogpt.model.gpt import SonoGPT
from sonogpt.schemas.domain import ThyroidExam
from sonogpt.tokenizer.sentencepiece_bpe import SentencePieceBPETokenizer
from sonogpt.training.checkpoint import sha256_path

INFERENCE_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class InferenceInfo:
    package_version: str
    inference_version: str
    schema_version: str
    extractor_version: str
    qc_rules_version: str
    freeze_id: str
    tokenizer_sha256: str
    checkpoint_name: str | None
    device: str
    parameter_count: int | None
    generate_task_trained: bool = True
    extract_task_trained: bool = False
    intended_use: str = "learning_demo_only"

    def to_dict(self) -> dict[str, object]:
        return {
            "package_version": self.package_version,
            "inference_version": self.inference_version,
            "schema_version": self.schema_version,
            "extractor_version": self.extractor_version,
            "qc_rules_version": self.qc_rules_version,
            "freeze_id": self.freeze_id,
            "tokenizer_sha256": self.tokenizer_sha256,
            "checkpoint_name": self.checkpoint_name,
            "device": self.device,
            "parameter_count": self.parameter_count,
            "generate_task_trained": self.generate_task_trained,
            "extract_task_trained": self.extract_task_trained,
            "intended_use": self.intended_use,
            "clinical_use": False,
        }


class InferenceEngine:
    """Lazy-loading local inference wrapper around frozen artifacts."""

    def __init__(
        self,
        *,
        project_root: Path,
        freeze_record_path: Path,
        checkpoint_path: Path,
        tokenizer_path: Path | None = None,
        device_name: str = "auto",
    ) -> None:
        self.project_root = project_root
        self.freeze_record_path = freeze_record_path
        self.checkpoint_path = checkpoint_path
        self.device = resolve_device(device_name)
        self.freeze_record = verify_freeze_record(
            freeze_record_path, project_root=project_root
        )
        artifacts = self.freeze_record["artifacts"]
        resolved_tokenizer = tokenizer_path or (
            project_root / artifacts["tokenizer_model"]["path"]
        )
        if not resolved_tokenizer.is_file():
            raise FileNotFoundError(
                f"frozen tokenizer is missing: {resolved_tokenizer}"
            )
        self.tokenizer = SentencePieceBPETokenizer(resolved_tokenizer)
        self._model: SonoGPT | None = None
        self._checkpoint_payload: dict[str, Any] | None = None

    def _load_model(self) -> SonoGPT:
        if self._model is None:
            if not self.checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"checkpoint is missing: {self.checkpoint_path}. "
                    "Copy it using COMPANY_HANDOFF.md."
                )
            self._model, self._checkpoint_payload = load_model_for_evaluation(
                self.checkpoint_path,
                self.tokenizer,
                device=self.device,
                freeze_record=self.freeze_record,
            )
        return self._model

    def info(self, *, load_model: bool = False) -> InferenceInfo:
        parameter_count = None
        checkpoint_name = (
            self.checkpoint_path.name if self.checkpoint_path.is_file() else None
        )
        if load_model and self.checkpoint_path.is_file():
            parameter_count = self._load_model().count_parameters()
        return InferenceInfo(
            package_version=PACKAGE_VERSION,
            inference_version=INFERENCE_VERSION,
            schema_version=SCHEMA_VERSION,
            extractor_version=RULE_EXTRACT_VERSION,
            qc_rules_version=QC_RULES_VERSION,
            freeze_id=str(self.freeze_record["freeze_id"]),
            tokenizer_sha256=self.tokenizer.model_sha256,
            checkpoint_name=checkpoint_name,
            device=str(self.device),
            parameter_count=parameter_count,
        )

    def extract(self, report: str) -> dict[str, object]:
        started = time.perf_counter()
        extracted = extract_exam(report)
        qc = run_qc(report, extracted.exam)
        return {
            "task": "extract",
            "extractor": "rule_extract",
            "model_used": False,
            "parseable": extracted.parseable,
            "exam": extracted.exam.model_dump(mode="json"),
            "canonical_json": extracted.canonical_json,
            "qc": qc.to_dict(),
            "elapsed_seconds": time.perf_counter() - started,
            "info": self.info().to_dict(),
            "notes": [
                "V1 did not train an extract-task model; this path uses rules only.",
                "Not for clinical diagnosis.",
            ],
        }

    def qc(
        self, report: str, structure: ThyroidExam | Mapping[str, object] | None = None
    ) -> dict[str, object]:
        started = time.perf_counter()
        exam = None
        if structure is not None:
            exam = (
                structure
                if isinstance(structure, ThyroidExam)
                else ThyroidExam.model_validate(structure)
            )
        result = run_qc(report, exam)
        return {
            "task": "qc",
            "qc": result.to_dict(),
            "elapsed_seconds": time.perf_counter() - started,
            "info": self.info().to_dict(),
            "notes": [
                "QC findings are engineering checks, not diagnostic advice.",
            ],
        }

    def generate(
        self,
        structure: ThyroidExam | Mapping[str, object],
        *,
        fallback_template: bool = True,
    ) -> dict[str, object]:
        started = time.perf_counter()
        exam = (
            structure
            if isinstance(structure, ThyroidExam)
            else ThyroidExam.model_validate(structure)
        )
        canonical = canonical_exam_json(exam)
        model = self._load_model()
        generated = generate_report(
            model, self.tokenizer, canonical, device=self.device
        )
        report_text = generated.text
        qc = run_qc(report_text, exam)
        used_fallback = False
        fallback_reason = None
        if generated.error is not None:
            used_fallback = True
            fallback_reason = generated.error
        elif fallback_template and not qc.passed:
            used_fallback = True
            fallback_reason = "qc_errors"
        if used_fallback and fallback_template:
            report_text = render_report(exam)
            qc = run_qc(report_text, exam)
        return {
            "task": "generate",
            "model_used": generated.error is None,
            "report": report_text,
            "used_fallback": used_fallback,
            "fallback_reason": fallback_reason,
            "eos_finished": generated.eos_finished,
            "generation_error": generated.error,
            "qc": qc.to_dict(),
            "exam": exam.model_dump(mode="json"),
            "canonical_json": canonical,
            "checkpoint_sha256": sha256_path(self.checkpoint_path),
            "freeze_record_sha256": sha256_file(self.freeze_record_path),
            "elapsed_seconds": time.perf_counter() - started,
            "info": self.info(load_model=True).to_dict(),
            "notes": [
                "Template fallback is used when greedy generation fails QC errors.",
                "Not for clinical diagnosis.",
            ],
        }
