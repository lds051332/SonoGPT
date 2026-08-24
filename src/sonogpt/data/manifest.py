"""Dataset statistics, deterministic JSONL writing, and SHA-256 manifests."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from sonogpt.data.constraints import (
    SYNTHETIC_CONSTRAINTS_VERSION,
    assert_synthetic_case_valid,
)
from sonogpt.data.renderers import (
    RENDERER_VERSION,
    GeneratedSample,
    render_with_family,
    sample_id_for,
)
from sonogpt.data.semantic_generator import (
    GENERATOR_VERSION,
    canonical_exam_json,
    semantic_case_id_for,
)
from sonogpt.data.split import GenerateDatasetSplits, validate_no_leakage
from sonogpt.schemas.domain import ObservationState, ThyroidExam

MANIFEST_VERSION = "1.1.0"
DATASET_VERSION = "1.1.0"


def sha256_file(path: Path) -> str:
    """Hash a file without loading the whole file into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _length_statistics(lengths: list[int]) -> dict[str, int | float | None]:
    if not lengths:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }

    ordered = sorted(lengths)

    def percentile(percent: int) -> int:
        index = math_ceil_div(percent * len(ordered), 100) - 1
        return ordered[max(0, index)]

    return {
        "min": ordered[0],
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 3),
        "p50": percentile(50),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
    }


def math_ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _validated_unique_exams(
    samples: tuple[GeneratedSample, ...],
) -> dict[str, ThyroidExam]:
    case_inputs: dict[str, str] = {}
    exams: dict[str, ThyroidExam] = {}
    for sample in samples:
        previous_input = case_inputs.setdefault(sample.semantic_case_id, sample.input)
        if previous_input != sample.input:
            raise ValueError("one semantic_case_id maps to multiple canonical inputs")

        exam = ThyroidExam.model_validate_json(sample.input)
        if canonical_exam_json(exam) != sample.input:
            raise ValueError("sample input is not canonical JSON")
        if semantic_case_id_for(exam) != sample.semantic_case_id:
            raise ValueError("sample semantic_case_id does not match its input")
        if sample.sample_id != sample_id_for(
            sample.semantic_case_id, sample.template_family
        ):
            raise ValueError("sample_id does not match its semantic/template identity")
        if sample.target != render_with_family(exam, sample.template_family):
            raise ValueError("sample target does not match its template family")
        if sample.schema_version != exam.schema_version:
            raise ValueError("sample schema_version does not match its input")
        if (
            sample.generator_version != GENERATOR_VERSION
            or sample.renderer_version != RENDERER_VERSION
        ):
            raise ValueError("sample generator or renderer version is unsupported")
        assert_synthetic_case_valid(exam)
        exams.setdefault(sample.semantic_case_id, exam)
    return exams


def compute_dataset_statistics(
    samples: Iterable[GeneratedSample],
) -> dict[str, object]:
    """Compute sample-level text counts and case-level semantic distributions."""

    sample_rows = tuple(samples)
    exams = _validated_unique_exams(sample_rows)
    field_counts: dict[str, Counter[str]] = {
        "location.side": Counter(),
        "location.segment": Counter(),
        "composition": Counter(),
        "echogenicity": Counter(),
        "shape": Counter(),
        "margin": Counter(),
        "echogenic_foci": Counter(),
        "vascularity": Counter(),
        "lymph_nodes": Counter(),
    }
    missing_states: Counter[str] = Counter()
    dimension_states: Counter[str] = Counter()
    dimension_counts: Counter[str] = Counter()
    dimension_values: list[float] = []

    for exam in exams.values():
        nodule = exam.nodules[0]
        values = {
            "location.side": nodule.location.side,
            "location.segment": nodule.location.segment,
            "composition": nodule.composition,
            "echogenicity": nodule.echogenicity,
            "shape": nodule.shape,
            "margin": nodule.margin,
            "echogenic_foci": nodule.echogenic_foci,
            "vascularity": nodule.vascularity,
            "lymph_nodes": exam.lymph_nodes,
        }
        for field_name, value in values.items():
            field_counts[field_name][value.value] += 1
            if value.value in {
                ObservationState.NOT_MENTIONED.value,
                ObservationState.UNKNOWN.value,
                ObservationState.NOT_APPLICABLE.value,
            }:
                missing_states[value.value] += 1

        if isinstance(nodule.dimensions_mm, list):
            dimension_counts[str(len(nodule.dimensions_mm))] += 1
            dimension_values.extend(nodule.dimensions_mm)
        else:
            state = nodule.dimensions_mm.value
            dimension_states[state] += 1
            missing_states[state] += 1

    sample_ids = [sample.sample_id for sample in sample_rows]
    inputs = [sample.input for sample in sample_rows]
    targets = [sample.target for sample in sample_rows]
    return {
        "sample_count": len(sample_rows),
        "semantic_case_count": len(exams),
        "task_counts": _sorted_counts(
            Counter(sample.task for sample in sample_rows)
        ),
        "source_counts": _sorted_counts(
            Counter(sample.source for sample in sample_rows)
        ),
        "template_family_counts": _sorted_counts(
            Counter(sample.template_family for sample in sample_rows)
        ),
        "semantic_field_counts": {
            field_name: _sorted_counts(counts)
            for field_name, counts in sorted(field_counts.items())
        },
        "missing_state_counts": _sorted_counts(missing_states),
        "dimensions": {
            "dimension_count_counts": _sorted_counts(dimension_counts),
            "state_counts": _sorted_counts(dimension_states),
            "value_min_mm": min(dimension_values) if dimension_values else None,
            "value_max_mm": max(dimension_values) if dimension_values else None,
            "unit_case_counts": {
                "mm": sum(dimension_counts.values()),
                "other": 0,
            },
        },
        "input_char_lengths": _length_statistics(
            [len(value) for value in inputs]
        ),
        "target_char_lengths": _length_statistics(
            [len(value) for value in targets]
        ),
        "duplicates": {
            "sample_id_rows": len(sample_ids) - len(set(sample_ids)),
            "semantic_case_rows": len(sample_rows) - len(exams),
            "input_rows": len(inputs) - len(set(inputs)),
            "target_rows": len(targets) - len(set(targets)),
        },
    }


def _write_json(path: Path, payload: object) -> None:
    serialized = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True
    )
    path.write_text(serialized + "\n", encoding="utf-8")


def _write_jsonl(path: Path, samples: tuple[GeneratedSample, ...]) -> None:
    lines = [
        json.dumps(
            sample.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        for sample in samples
    ]
    content = "\n".join(lines)
    if lines:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def write_dataset_bundle(
    splits: GenerateDatasetSplits,
    *,
    output_directory: Path,
    manifest_path: Path,
    dataset_name: str = "sonogpt_single_nodule_generate_v1",
) -> dict[str, object]:
    """Write deterministic split files, statistics, and a hash manifest."""

    validate_no_leakage(splits)
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    split_samples = {
        "train": splits.train,
        "validation": splits.validation,
        "test_seen_templates": splits.test_seen_templates,
        "test_heldout_templates": splits.test_heldout_templates,
    }
    file_entries: dict[str, dict[str, object]] = {}
    for split_name, samples in split_samples.items():
        file_path = output_directory / f"{split_name}.jsonl"
        _write_jsonl(file_path, samples)
        file_entries[split_name] = {
            "path": file_path.name,
            "sha256": sha256_file(file_path),
            "bytes": file_path.stat().st_size,
            "records": len(samples),
        }

    all_samples = splits.all_samples()
    statistics = compute_dataset_statistics(all_samples)
    statistics["splits"] = {
        name: {
            "sample_count": len(samples),
            "semantic_case_count": len(_validated_unique_exams(samples)),
        }
        for name, samples in split_samples.items()
    }
    statistics_path = output_directory / "statistics.json"
    _write_json(statistics_path, statistics)
    file_entries["statistics"] = {
        "path": statistics_path.name,
        "sha256": sha256_file(statistics_path),
        "bytes": statistics_path.stat().st_size,
        "records": None,
    }

    seeds = sorted({sample.seed for sample in all_samples})
    schema_versions = sorted({sample.schema_version for sample in all_samples})
    manifest: dict[str, object] = {
        "manifest_version": MANIFEST_VERSION,
        "dataset_version": DATASET_VERSION,
        "dataset_name": dataset_name,
        "task": "generate",
        "source": "synthetic",
        "generator_version": GENERATOR_VERSION,
        "synthetic_constraints_version": SYNTHETIC_CONSTRAINTS_VERSION,
        "renderer_version": RENDERER_VERSION,
        "dimension_order": [
            "transverse_mm",
            "anteroposterior_mm",
            "longitudinal_mm_if_present",
        ],
        "schema_versions": schema_versions,
        "seeds": seeds,
        "semantic_case_count": statistics["semantic_case_count"],
        "seen_template_families": sorted(
            {
                sample.template_family
                for sample in (
                    splits.train
                    + splits.validation
                    + splits.test_seen_templates
                )
            }
        ),
        "heldout_template_families": sorted(
            {sample.template_family for sample in splits.test_heldout_templates}
        ),
        "files": file_entries,
    }
    _write_json(manifest_path, manifest)
    return manifest


def verify_manifest(manifest_path: Path, data_directory: Path) -> None:
    """Recompute every listed file hash and reject unsafe manifest paths."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["files"].values():
        relative_path = Path(entry["path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("manifest contains an unsafe file path")
        file_path = data_directory / relative_path
        if sha256_file(file_path) != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {relative_path}")
