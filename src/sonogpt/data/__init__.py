"""Synthetic data generation and leakage-resistant dataset utilities."""

from sonogpt.data.constraints import (
    SYNTHETIC_CONSTRAINTS_VERSION,
    assert_synthetic_case_valid,
    synthetic_constraint_violations,
)
from sonogpt.data.dataset import (
    CausalLMBatch,
    EncodedGenerateSample,
    SequenceTooLongError,
    collate_encoded_samples,
    encode_generate_sample,
)
from sonogpt.data.freeze import (
    FREEZE_RECORD_VERSION,
    create_freeze_record,
    verify_freeze_record,
)
from sonogpt.data.manifest import (
    compute_dataset_statistics,
    sha256_file,
    verify_manifest,
    write_dataset_bundle,
)
from sonogpt.data.renderers import (
    TEMPLATE_FAMILIES,
    GeneratedSample,
    render_case,
    render_cases,
    render_with_family,
    sample_id_for,
)
from sonogpt.data.semantic_generator import (
    GENERATOR_VERSION,
    SemanticCase,
    canonical_exam_json,
    sample_semantic_cases,
    semantic_case_id_for,
)
from sonogpt.data.split import (
    GenerateDatasetSplits,
    build_generate_splits,
    split_semantic_cases,
    validate_no_leakage,
)

__all__ = [
    "GENERATOR_VERSION",
    "FREEZE_RECORD_VERSION",
    "SYNTHETIC_CONSTRAINTS_VERSION",
    "TEMPLATE_FAMILIES",
    "CausalLMBatch",
    "EncodedGenerateSample",
    "GenerateDatasetSplits",
    "GeneratedSample",
    "SemanticCase",
    "SequenceTooLongError",
    "build_generate_splits",
    "assert_synthetic_case_valid",
    "canonical_exam_json",
    "collate_encoded_samples",
    "compute_dataset_statistics",
    "create_freeze_record",
    "encode_generate_sample",
    "render_case",
    "render_cases",
    "render_with_family",
    "sample_id_for",
    "sample_semantic_cases",
    "semantic_case_id_for",
    "sha256_file",
    "split_semantic_cases",
    "synthetic_constraint_violations",
    "validate_no_leakage",
    "verify_manifest",
    "verify_freeze_record",
    "write_dataset_bundle",
]
