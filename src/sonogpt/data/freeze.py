"""Versioned freeze records for reviewed datasets and tokenizers."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from sonogpt.data.manifest import sha256_file, verify_manifest

FREEZE_RECORD_VERSION = "1.0.0"
APPROVED_REVIEW_OUTCOMES = {
    "approved_no_changes",
    "approved_after_changes",
}
DEFAULT_PROVENANCE_PATHS = (
    "scripts/generate_data.py",
    "scripts/train_tokenizer.py",
    "src/sonogpt/baselines/template_report.py",
    "src/sonogpt/data/constraints.py",
    "src/sonogpt/data/manifest.py",
    "src/sonogpt/data/renderers.py",
    "src/sonogpt/data/semantic_generator.py",
    "src/sonogpt/data/split.py",
    "src/sonogpt/schemas/domain.py",
    "src/sonogpt/tokenizer/sentencepiece_bpe.py",
)


def _canonical_payload_sha256(payload: dict[str, object]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _relative_project_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("frozen artifacts must be inside the project") from error


def _resolve_project_path(value: object, project_root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("freeze record contains an invalid project path")
    relative_path = Path(value)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("freeze record contains an unsafe project path")
    resolved = (project_root / relative_path).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as error:
        raise ValueError("freeze record path escapes the project") from error
    return resolved


def create_freeze_record(
    *,
    freeze_id: str,
    project_root: Path,
    data_directory: Path,
    data_manifest_path: Path,
    tokenizer_model_path: Path,
    tokenizer_manifest_path: Path,
    review_outcome: str,
    review_sample_count: int,
    reviewer_role: str,
    review_date: str,
    output_path: Path,
    provenance_paths: tuple[str, ...] = DEFAULT_PROVENANCE_PATHS,
) -> dict[str, object]:
    """Create a new immutable release record after integrity checks."""

    if not freeze_id or not freeze_id.replace("_", "").isalnum():
        raise ValueError("freeze_id must contain only letters, digits, and underscores")
    if review_outcome not in APPROVED_REVIEW_OUTCOMES:
        raise ValueError("the professional review has not approved this release")
    if review_sample_count <= 0:
        raise ValueError("review_sample_count must be positive")
    if not reviewer_role:
        raise ValueError("reviewer_role must be recorded")
    if output_path.exists():
        raise FileExistsError(f"freeze record already exists: {output_path}")

    verify_manifest(data_manifest_path, data_directory)
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    tokenizer_manifest = json.loads(
        tokenizer_manifest_path.read_text(encoding="utf-8")
    )
    data_manifest_sha256 = sha256_file(data_manifest_path)
    tokenizer_model_sha256 = sha256_file(tokenizer_model_path)
    if tokenizer_manifest["data_manifest_sha256"] != data_manifest_sha256:
        raise ValueError("tokenizer was not trained from this data manifest")
    if tokenizer_manifest["tokenizer_model_sha256"] != tokenizer_model_sha256:
        raise ValueError("tokenizer model does not match its manifest")

    provenance = {}
    for relative_path in provenance_paths:
        path = _resolve_project_path(relative_path, project_root)
        provenance[relative_path] = sha256_file(path)

    record: dict[str, object] = {
        "freeze_record_version": FREEZE_RECORD_VERSION,
        "freeze_id": freeze_id,
        "status": "frozen",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "artifacts": {
            "data_directory": _relative_project_path(
                data_directory, project_root
            ),
            "data_manifest": {
                "path": _relative_project_path(
                    data_manifest_path, project_root
                ),
                "sha256": data_manifest_sha256,
            },
            "dataset_files": deepcopy(data_manifest["files"]),
            "tokenizer_manifest": {
                "path": _relative_project_path(
                    tokenizer_manifest_path, project_root
                ),
                "sha256": sha256_file(tokenizer_manifest_path),
            },
            "tokenizer_model": {
                "path": _relative_project_path(
                    tokenizer_model_path, project_root
                ),
                "sha256": tokenizer_model_sha256,
            },
        },
        "dataset_identity": {
            "dataset_name": data_manifest["dataset_name"],
            "dataset_version": data_manifest["dataset_version"],
            "generator_version": data_manifest["generator_version"],
            "manifest_version": data_manifest["manifest_version"],
            "renderer_version": data_manifest["renderer_version"],
            "schema_versions": data_manifest["schema_versions"],
            "semantic_case_count": data_manifest["semantic_case_count"],
            "synthetic_constraints_version": data_manifest[
                "synthetic_constraints_version"
            ],
        },
        "professional_review": {
            "attestation_source": "project_owner_reported_external_review",
            "outcome": review_outcome,
            "review_date": review_date,
            "review_sample_count": review_sample_count,
            "reviewer_identity_recorded": False,
            "reviewer_role": reviewer_role,
            "unresolved_issue_count": 0,
        },
        "evaluation_scope": {
            "challenge_set_included": False,
            "challenge_set_status": "not_created",
            "included_splits": sorted(
                key
                for key in data_manifest["files"]
                if key != "statistics"
            ),
        },
        "provenance_file_sha256": provenance,
    }
    record["freeze_record_sha256"] = _canonical_payload_sha256(record)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


def verify_freeze_record(
    freeze_record_path: Path,
    *,
    project_root: Path,
    verify_provenance: bool = False,
) -> dict[str, object]:
    """Reject any mutation to a frozen dataset, tokenizer, or record."""

    record = json.loads(freeze_record_path.read_text(encoding="utf-8"))
    if record.get("freeze_record_version") != FREEZE_RECORD_VERSION:
        raise ValueError("unsupported freeze record version")
    if record.get("status") != "frozen":
        raise ValueError("release is not marked as frozen")
    expected_record_sha256 = record.pop("freeze_record_sha256", None)
    actual_record_sha256 = _canonical_payload_sha256(record)
    record["freeze_record_sha256"] = expected_record_sha256
    if actual_record_sha256 != expected_record_sha256:
        raise ValueError("freeze record SHA-256 mismatch")

    review = record["professional_review"]
    if (
        review["outcome"] not in APPROVED_REVIEW_OUTCOMES
        or review["review_sample_count"] <= 0
        or review["unresolved_issue_count"] != 0
    ):
        raise ValueError("freeze record does not contain an approved review")

    artifacts = record["artifacts"]
    data_directory = _resolve_project_path(
        artifacts["data_directory"], project_root
    )
    data_manifest_entry = artifacts["data_manifest"]
    data_manifest_path = _resolve_project_path(
        data_manifest_entry["path"], project_root
    )
    if sha256_file(data_manifest_path) != data_manifest_entry["sha256"]:
        raise ValueError("frozen data manifest SHA-256 mismatch")
    verify_manifest(data_manifest_path, data_directory)

    for artifact_name in ("tokenizer_manifest", "tokenizer_model"):
        entry = artifacts[artifact_name]
        path = _resolve_project_path(entry["path"], project_root)
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"frozen {artifact_name} SHA-256 mismatch")

    tokenizer_manifest = json.loads(
        _resolve_project_path(
            artifacts["tokenizer_manifest"]["path"], project_root
        ).read_text(encoding="utf-8")
    )
    if (
        tokenizer_manifest["data_manifest_sha256"]
        != data_manifest_entry["sha256"]
        or tokenizer_manifest["tokenizer_model_sha256"]
        != artifacts["tokenizer_model"]["sha256"]
    ):
        raise ValueError("frozen tokenizer provenance does not match")

    if verify_provenance:
        for relative_path, expected_sha256 in record[
            "provenance_file_sha256"
        ].items():
            path = _resolve_project_path(relative_path, project_root)
            if sha256_file(path) != expected_sha256:
                raise ValueError(
                    f"frozen provenance SHA-256 mismatch: {relative_path}"
                )
    return record
