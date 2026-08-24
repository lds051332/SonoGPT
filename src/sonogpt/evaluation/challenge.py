"""Leakage checks and freezing for an AI-simulated human challenge set."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from sonogpt.data.constraints import assert_synthetic_case_valid
from sonogpt.data.freeze import verify_freeze_record
from sonogpt.data.manifest import sha256_file
from sonogpt.data.renderers import TEMPLATE_FAMILIES, render_with_family
from sonogpt.data.semantic_generator import (
    canonical_exam_json,
    semantic_case_id_for,
)
from sonogpt.schemas.domain import ThyroidExam

CHALLENGE_SET_VERSION = "1.0.0"
CHALLENGE_FREEZE_VERSION = "1.0.0"
CHALLENGE_AUTHORING_METHOD = "ai_simulated_manual_independent_v1"
CHALLENGE_SOURCE = "simulated_nonclinical"
_FORBIDDEN_DIAGNOSTIC_TERMS = ("TI-RADS", "良性", "恶性", "癌")


def _canonical_sha256(payload: dict[str, object]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _safe_project_path(value: object, project_root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("challenge freeze contains an invalid path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("challenge freeze contains an unsafe path")
    resolved = (project_root / relative).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as error:
        raise ValueError("challenge path escapes the project") from error
    return resolved


def _relative_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("challenge artifacts must be inside the project") from error


def challenge_id_for(
    semantic_case_id: str,
    reference_report: str,
) -> str:
    identity = (
        f"{CHALLENGE_SET_VERSION}:{semantic_case_id}:{reference_report}"
    ).encode("utf-8")
    return f"challenge_{hashlib.sha256(identity).hexdigest()}"


@dataclass(frozen=True)
class ChallengeSample:
    challenge_id: str
    semantic_case_id: str
    input: str
    reference_report: str
    difficulty_tags: tuple[str, ...]
    schema_version: str = "1.0.0"
    challenge_version: str = CHALLENGE_SET_VERSION
    authoring_method: str = CHALLENGE_AUTHORING_METHOD
    source: str = CHALLENGE_SOURCE

    @classmethod
    def from_exam(
        cls,
        exam: ThyroidExam,
        *,
        reference_report: str,
        difficulty_tags: tuple[str, ...],
    ) -> "ChallengeSample":
        semantic_case_id = semantic_case_id_for(exam)
        return cls(
            challenge_id=challenge_id_for(
                semantic_case_id, reference_report
            ),
            semantic_case_id=semantic_case_id,
            input=canonical_exam_json(exam),
            reference_report=reference_report,
            difficulty_tags=difficulty_tags,
            schema_version=exam.schema_version,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ChallengeSample":
        normalized = dict(payload)
        normalized["difficulty_tags"] = tuple(payload["difficulty_tags"])
        return cls(**normalized)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["difficulty_tags"] = list(self.difficulty_tags)
        return payload


def _load_frozen_training_identities(
    training_freeze_record: Path,
    *,
    project_root: Path,
) -> tuple[set[str], set[str]]:
    record = verify_freeze_record(
        training_freeze_record,
        project_root=project_root,
    )
    artifacts = record["artifacts"]
    data_directory = _safe_project_path(
        artifacts["data_directory"], project_root
    )
    semantic_case_ids: set[str] = set()
    reports: set[str] = set()
    for entry in artifacts["dataset_files"].values():
        if not str(entry["path"]).endswith(".jsonl"):
            continue
        path = data_directory / entry["path"]
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            semantic_case_ids.add(row["semantic_case_id"])
            reports.add(row["target"])
    return semantic_case_ids, reports


def validate_challenge_samples(
    samples: tuple[ChallengeSample, ...],
    *,
    frozen_semantic_case_ids: set[str],
    frozen_reports: set[str],
) -> dict[str, object]:
    if not samples:
        raise ValueError("challenge set must not be empty")
    challenge_ids: set[str] = set()
    semantic_case_ids: set[str] = set()
    reference_reports: set[str] = set()
    tag_counts: Counter[str] = Counter()

    for sample in samples:
        if sample.challenge_version != CHALLENGE_SET_VERSION:
            raise ValueError("challenge sample has an unsupported version")
        if sample.authoring_method != CHALLENGE_AUTHORING_METHOD:
            raise ValueError("challenge authoring method is not independent")
        if sample.source != CHALLENGE_SOURCE:
            raise ValueError("challenge source must be simulated and nonclinical")
        exam = ThyroidExam.model_validate_json(sample.input)
        if canonical_exam_json(exam) != sample.input:
            raise ValueError("challenge input is not canonical JSON")
        assert_synthetic_case_valid(exam)
        expected_semantic_case_id = semantic_case_id_for(exam)
        if sample.semantic_case_id != expected_semantic_case_id:
            raise ValueError("challenge semantic_case_id mismatch")
        if sample.challenge_id != challenge_id_for(
            sample.semantic_case_id, sample.reference_report
        ):
            raise ValueError("challenge_id mismatch")
        if not sample.reference_report.strip():
            raise ValueError("challenge reference report must not be empty")
        if any(
            term in sample.reference_report
            for term in _FORBIDDEN_DIAGNOSTIC_TERMS
        ):
            raise ValueError("challenge report contains a diagnostic conclusion")
        template_reports = {
            render_with_family(exam, family) for family in TEMPLATE_FAMILIES
        }
        if sample.reference_report in template_reports:
            raise ValueError("challenge report duplicates a training template")
        if not sample.difficulty_tags or len(sample.difficulty_tags) != len(
            set(sample.difficulty_tags)
        ):
            raise ValueError("challenge difficulty tags must be unique and non-empty")
        if sample.challenge_id in challenge_ids:
            raise ValueError("duplicate challenge_id")
        if sample.semantic_case_id in semantic_case_ids:
            raise ValueError("duplicate challenge semantic case")
        if sample.reference_report in reference_reports:
            raise ValueError("duplicate challenge reference report")
        if sample.semantic_case_id in frozen_semantic_case_ids:
            raise ValueError("challenge semantic case leaks from frozen data")
        if sample.reference_report in frozen_reports:
            raise ValueError("challenge report leaks from frozen data")
        challenge_ids.add(sample.challenge_id)
        semantic_case_ids.add(sample.semantic_case_id)
        reference_reports.add(sample.reference_report)
        tag_counts.update(sample.difficulty_tags)

    return {
        "sample_count": len(samples),
        "unique_semantic_case_count": len(semantic_case_ids),
        "difficulty_tag_counts": {
            key: tag_counts[key] for key in sorted(tag_counts)
        },
        "training_semantic_overlap_count": 0,
        "training_report_overlap_count": 0,
        "template_exact_match_count": 0,
    }


def write_frozen_challenge_set(
    samples: tuple[ChallengeSample, ...],
    *,
    challenge_path: Path,
    freeze_record_path: Path,
    training_freeze_record: Path,
    project_root: Path,
    builder_path: Path,
) -> dict[str, object]:
    if challenge_path.exists() or freeze_record_path.exists():
        raise FileExistsError("challenge release already exists")
    frozen_ids, frozen_reports = _load_frozen_training_identities(
        training_freeze_record,
        project_root=project_root,
    )
    statistics = validate_challenge_samples(
        samples,
        frozen_semantic_case_ids=frozen_ids,
        frozen_reports=frozen_reports,
    )
    challenge_path.parent.mkdir(parents=True, exist_ok=True)
    challenge_path.write_text(
        "".join(
            json.dumps(
                sample.to_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
            for sample in samples
        ),
        encoding="utf-8",
    )
    training_freeze = verify_freeze_record(
        training_freeze_record,
        project_root=project_root,
    )
    record: dict[str, object] = {
        "challenge_freeze_version": CHALLENGE_FREEZE_VERSION,
        "challenge_version": CHALLENGE_SET_VERSION,
        "freeze_id": "simulated_human_challenge_v1",
        "status": "frozen",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "authoring": {
            "method": CHALLENGE_AUTHORING_METHOD,
            "source": CHALLENGE_SOURCE,
            "human_authored": False,
            "clinical_data": False,
            "intended_use": "learning_demo_evaluation_only",
        },
        "challenge_file": {
            "path": _relative_path(challenge_path, project_root),
            "sha256": sha256_file(challenge_path),
            "bytes": challenge_path.stat().st_size,
            "records": len(samples),
        },
        "linked_training_freeze": {
            "path": _relative_path(training_freeze_record, project_root),
            "freeze_id": training_freeze["freeze_id"],
            "freeze_record_sha256": sha256_file(training_freeze_record),
            "freeze_payload_sha256": training_freeze[
                "freeze_record_sha256"
            ],
        },
        "builder": {
            "path": _relative_path(builder_path, project_root),
            "sha256": sha256_file(builder_path),
        },
        "statistics": statistics,
        "training_use_prohibited": True,
        "tokenizer_training_use_prohibited": True,
    }
    record["freeze_payload_sha256"] = _canonical_sha256(record)
    freeze_record_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


def verify_challenge_freeze(
    freeze_record_path: Path,
    *,
    project_root: Path,
    verify_builder: bool = True,
) -> dict[str, object]:
    record = json.loads(freeze_record_path.read_text(encoding="utf-8"))
    if record.get("challenge_freeze_version") != CHALLENGE_FREEZE_VERSION:
        raise ValueError("unsupported challenge freeze version")
    if record.get("status") != "frozen":
        raise ValueError("challenge set is not frozen")
    expected_payload_sha256 = record.pop("freeze_payload_sha256", None)
    actual_payload_sha256 = _canonical_sha256(record)
    record["freeze_payload_sha256"] = expected_payload_sha256
    if actual_payload_sha256 != expected_payload_sha256:
        raise ValueError("challenge freeze payload SHA-256 mismatch")
    if (
        not record["training_use_prohibited"]
        or not record["tokenizer_training_use_prohibited"]
        or record["authoring"]["human_authored"]
        or record["authoring"]["clinical_data"]
    ):
        raise ValueError("challenge usage or provenance flags are unsafe")

    training_entry = record["linked_training_freeze"]
    training_freeze_path = _safe_project_path(
        training_entry["path"], project_root
    )
    if sha256_file(training_freeze_path) != training_entry[
        "freeze_record_sha256"
    ]:
        raise ValueError("linked training freeze SHA-256 mismatch")
    verify_freeze_record(training_freeze_path, project_root=project_root)

    challenge_entry = record["challenge_file"]
    challenge_path = _safe_project_path(challenge_entry["path"], project_root)
    if sha256_file(challenge_path) != challenge_entry["sha256"]:
        raise ValueError("frozen challenge file SHA-256 mismatch")
    samples = tuple(
        ChallengeSample.from_dict(json.loads(line))
        for line in challenge_path.read_text(encoding="utf-8").splitlines()
        if line
    )
    frozen_ids, frozen_reports = _load_frozen_training_identities(
        training_freeze_path,
        project_root=project_root,
    )
    statistics = validate_challenge_samples(
        samples,
        frozen_semantic_case_ids=frozen_ids,
        frozen_reports=frozen_reports,
    )
    if statistics != record["statistics"]:
        raise ValueError("challenge statistics mismatch")
    if verify_builder:
        builder_entry = record["builder"]
        builder_path = _safe_project_path(builder_entry["path"], project_root)
        if sha256_file(builder_path) != builder_entry["sha256"]:
            raise ValueError("challenge builder SHA-256 mismatch")
    return record
