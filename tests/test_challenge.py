from __future__ import annotations

import json
from pathlib import Path

import pytest

from sonogpt.data.freeze import create_freeze_record
from sonogpt.data.manifest import sha256_file, write_dataset_bundle
from sonogpt.data.semantic_generator import sample_semantic_cases
from sonogpt.data.split import build_generate_splits
from sonogpt.evaluation.challenge import (
    ChallengeSample,
    verify_challenge_freeze,
    write_frozen_challenge_set,
)
from sonogpt.schemas.domain import ThyroidExam


def _training_freeze(tmp_path: Path) -> tuple[Path, Path]:
    data_directory = tmp_path / "data" / "processed" / "candidate"
    manifest_path = tmp_path / "data" / "manifests" / "candidate.json"
    manifest_path.parent.mkdir(parents=True)
    write_dataset_bundle(
        build_generate_splits(
            sample_semantic_cases(20, seed=801),
            seed=802,
        ),
        output_directory=data_directory,
        manifest_path=manifest_path,
    )
    tokenizer_directory = tmp_path / "artifacts" / "tokenizer"
    tokenizer_directory.mkdir(parents=True)
    tokenizer_model = tokenizer_directory / "tokenizer.model"
    tokenizer_model.write_bytes(b"challenge-test-tokenizer")
    tokenizer_manifest = tokenizer_directory / "manifest.json"
    tokenizer_manifest.write_text(
        json.dumps(
            {
                "data_manifest_sha256": sha256_file(manifest_path),
                "tokenizer_model_sha256": sha256_file(tokenizer_model),
            }
        ),
        encoding="utf-8",
    )
    training_freeze = (
        tmp_path / "data" / "releases" / "training.freeze.json"
    )
    create_freeze_record(
        freeze_id="training_frozen_v1",
        project_root=tmp_path,
        data_directory=data_directory,
        data_manifest_path=manifest_path,
        tokenizer_model_path=tokenizer_model,
        tokenizer_manifest_path=tokenizer_manifest,
        review_outcome="approved_no_changes",
        review_sample_count=10,
        reviewer_role="test_reviewer",
        review_date="2026-08-24",
        output_path=training_freeze,
        provenance_paths=(),
    )
    return training_freeze, data_directory


def _challenge_sample() -> ChallengeSample:
    exam = ThyroidExam.model_validate(
        {
            "nodules": [
                {
                    "location": {"side": "isthmus", "segment": "not_applicable"},
                    "dimensions_mm": [9.5, 4.5],
                    "composition": "mixed_cystic_solid",
                    "echogenicity": "hypoechoic",
                    "shape": "wider_than_tall",
                    "margin": "smooth",
                    "echogenic_foci": "none",
                    "vascularity": "peripheral",
                }
            ]
        }
    )
    return ChallengeSample.from_exam(
        exam,
        reference_report=(
            "峡部见约9.5×4.5mm囊实性低回声结节，横向生长且边缘平整，"
            "内部无明确强回声，血流环绕其周边。"
        ),
        difficulty_tags=("isthmus", "lexical_paraphrase"),
    )


def test_simulated_challenge_is_frozen_and_linked_without_leakage(
    tmp_path: Path,
) -> None:
    training_freeze, _ = _training_freeze(tmp_path)
    builder_path = tmp_path / "builder.py"
    builder_path.write_text("# fixed builder\n", encoding="utf-8")
    challenge_path = tmp_path / "data" / "challenges" / "challenge.jsonl"
    challenge_freeze = (
        tmp_path / "data" / "releases" / "challenge.freeze.json"
    )

    write_frozen_challenge_set(
        (_challenge_sample(),),
        challenge_path=challenge_path,
        freeze_record_path=challenge_freeze,
        training_freeze_record=training_freeze,
        project_root=tmp_path,
        builder_path=builder_path,
    )
    record = verify_challenge_freeze(
        challenge_freeze,
        project_root=tmp_path,
    )

    assert record["authoring"]["human_authored"] is False
    assert record["statistics"]["training_semantic_overlap_count"] == 0
    assert record["training_use_prohibited"] is True


def test_challenge_freeze_rejects_file_mutation(tmp_path: Path) -> None:
    training_freeze, _ = _training_freeze(tmp_path)
    builder_path = tmp_path / "builder.py"
    builder_path.write_text("# fixed builder\n", encoding="utf-8")
    challenge_path = tmp_path / "data" / "challenges" / "challenge.jsonl"
    challenge_freeze = (
        tmp_path / "data" / "releases" / "challenge.freeze.json"
    )
    write_frozen_challenge_set(
        (_challenge_sample(),),
        challenge_path=challenge_path,
        freeze_record_path=challenge_freeze,
        training_freeze_record=training_freeze,
        project_root=tmp_path,
        builder_path=builder_path,
    )
    challenge_path.write_text(
        challenge_path.read_text(encoding="utf-8") + "tampered\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="challenge file SHA-256 mismatch"):
        verify_challenge_freeze(
            challenge_freeze,
            project_root=tmp_path,
        )


def test_challenge_rejects_semantic_leakage(tmp_path: Path) -> None:
    training_freeze, data_directory = _training_freeze(tmp_path)
    frozen_row = json.loads(
        (data_directory / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    leaked_exam = ThyroidExam.model_validate_json(frozen_row["input"])
    leaked_sample = ChallengeSample.from_exam(
        leaked_exam,
        reference_report="这是独立措辞，但结构化病例已经出现在训练数据中。",
        difficulty_tags=("leakage_test",),
    )
    builder_path = tmp_path / "builder.py"
    builder_path.write_text("# fixed builder\n", encoding="utf-8")

    with pytest.raises(ValueError, match="leaks from frozen data"):
        write_frozen_challenge_set(
            (leaked_sample,),
            challenge_path=tmp_path / "challenge.jsonl",
            freeze_record_path=tmp_path / "challenge.freeze.json",
            training_freeze_record=training_freeze,
            project_root=tmp_path,
            builder_path=builder_path,
        )
