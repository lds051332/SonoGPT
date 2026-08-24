from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sonogpt.data.manifest import (
    compute_dataset_statistics,
    verify_manifest,
    write_dataset_bundle,
)
from sonogpt.data.semantic_generator import sample_semantic_cases
from sonogpt.data.split import build_generate_splits


def test_statistics_count_semantics_once_and_templates_per_sample() -> None:
    splits = build_generate_splits(sample_semantic_cases(20, seed=21), seed=22)
    statistics = compute_dataset_statistics(splits.all_samples())

    assert statistics["semantic_case_count"] == 20
    assert statistics["sample_count"] == 20 * 3 + 2
    assert statistics["task_counts"] == {"generate": 62}
    assert statistics["duplicates"]["sample_id_rows"] == 0  # type: ignore[index]
    assert set(statistics["template_family_counts"]) == {  # type: ignore[arg-type]
        "location_first_v2",
        "descriptor_first_v2",
        "dimensions_first_v2",
        "flow_first_v2",
    }


def test_manifest_hashes_are_reproducible_and_verifiable(tmp_path: Path) -> None:
    splits = build_generate_splits(sample_semantic_cases(20, seed=31), seed=32)
    first_data = tmp_path / "first"
    second_data = tmp_path / "second"
    first_manifest = tmp_path / "first.manifest.json"
    second_manifest = tmp_path / "second.manifest.json"

    first = write_dataset_bundle(
        splits,
        output_directory=first_data,
        manifest_path=first_manifest,
    )
    second = write_dataset_bundle(
        splits,
        output_directory=second_data,
        manifest_path=second_manifest,
    )

    assert first == second
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    verify_manifest(first_manifest, first_data)

    payload = json.loads(first_manifest.read_text(encoding="utf-8"))
    for entry in payload["files"].values():
        file_bytes = (first_data / entry["path"]).read_bytes()
        assert hashlib.sha256(file_bytes).hexdigest() == entry["sha256"]


def test_manifest_verification_detects_tampering(tmp_path: Path) -> None:
    splits = build_generate_splits(sample_semantic_cases(10, seed=41), seed=42)
    data_directory = tmp_path / "data"
    manifest_path = tmp_path / "manifest.json"
    write_dataset_bundle(
        splits,
        output_directory=data_directory,
        manifest_path=manifest_path,
    )
    train_path = data_directory / "train.jsonl"
    train_path.write_text(
        train_path.read_text(encoding="utf-8") + "tampered\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_manifest(manifest_path, data_directory)
