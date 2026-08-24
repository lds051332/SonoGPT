from __future__ import annotations

import json
from pathlib import Path

import pytest

from sonogpt.data.freeze import create_freeze_record, verify_freeze_record
from sonogpt.data.manifest import sha256_file, write_dataset_bundle
from sonogpt.data.semantic_generator import sample_semantic_cases
from sonogpt.data.split import build_generate_splits


def _create_test_release(tmp_path: Path) -> tuple[Path, Path]:
    data_directory = tmp_path / "data" / "processed" / "candidate"
    data_manifest_path = tmp_path / "data" / "manifests" / "candidate.json"
    data_manifest_path.parent.mkdir(parents=True)
    write_dataset_bundle(
        build_generate_splits(
            sample_semantic_cases(20, seed=701),
            seed=702,
        ),
        output_directory=data_directory,
        manifest_path=data_manifest_path,
    )

    tokenizer_directory = tmp_path / "artifacts" / "tokenizers" / "candidate"
    tokenizer_directory.mkdir(parents=True)
    tokenizer_model_path = tokenizer_directory / "tokenizer.model"
    tokenizer_model_path.write_bytes(b"deterministic-tokenizer")
    tokenizer_manifest_path = tokenizer_directory / "tokenizer_manifest.json"
    tokenizer_manifest_path.write_text(
        json.dumps(
            {
                "data_manifest_sha256": sha256_file(data_manifest_path),
                "tokenizer_model_sha256": sha256_file(tokenizer_model_path),
            }
        ),
        encoding="utf-8",
    )

    freeze_record_path = (
        tmp_path / "data" / "releases" / "frozen_v1.freeze.json"
    )
    create_freeze_record(
        freeze_id="frozen_v1",
        project_root=tmp_path,
        data_directory=data_directory,
        data_manifest_path=data_manifest_path,
        tokenizer_model_path=tokenizer_model_path,
        tokenizer_manifest_path=tokenizer_manifest_path,
        review_outcome="approved_no_changes",
        review_sample_count=50,
        reviewer_role="licensed_ultrasound_physician",
        review_date="2026-08-24",
        output_path=freeze_record_path,
        provenance_paths=(),
    )
    return freeze_record_path, data_directory


def test_freeze_record_preserves_review_and_artifact_identity(
    tmp_path: Path,
) -> None:
    freeze_record_path, _ = _create_test_release(tmp_path)

    record = verify_freeze_record(
        freeze_record_path,
        project_root=tmp_path,
        verify_provenance=True,
    )

    assert record["status"] == "frozen"
    assert record["professional_review"]["review_sample_count"] == 50
    assert record["evaluation_scope"]["challenge_set_included"] is False


def test_freeze_verification_rejects_dataset_mutation(tmp_path: Path) -> None:
    freeze_record_path, data_directory = _create_test_release(tmp_path)
    train_path = data_directory / "train.jsonl"
    train_path.write_text(
        train_path.read_text(encoding="utf-8") + "tampered\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_freeze_record(freeze_record_path, project_root=tmp_path)


def test_freeze_verification_rejects_record_mutation(tmp_path: Path) -> None:
    freeze_record_path, _ = _create_test_release(tmp_path)
    record = json.loads(freeze_record_path.read_text(encoding="utf-8"))
    record["professional_review"]["review_sample_count"] = 51
    freeze_record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="freeze record SHA-256 mismatch"):
        verify_freeze_record(freeze_record_path, project_root=tmp_path)
