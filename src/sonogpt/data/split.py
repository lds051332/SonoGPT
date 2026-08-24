"""Leakage-resistant semantic-case and template-family splitting."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from sonogpt.data.renderers import (
    TEMPLATE_FAMILIES,
    GeneratedSample,
    render_cases,
)
from sonogpt.data.semantic_generator import DEFAULT_SEED, SemanticCase


@dataclass(frozen=True)
class SemanticCaseSplits:
    """Top-level partitions; each semantic case occurs in exactly one."""

    train: tuple[SemanticCase, ...]
    validation: tuple[SemanticCase, ...]
    test: tuple[SemanticCase, ...]


@dataclass(frozen=True)
class GenerateDatasetSplits:
    """Generate-task rows with a template holdout nested under test cases."""

    train: tuple[GeneratedSample, ...]
    validation: tuple[GeneratedSample, ...]
    test_seen_templates: tuple[GeneratedSample, ...]
    test_heldout_templates: tuple[GeneratedSample, ...]

    def all_samples(self) -> tuple[GeneratedSample, ...]:
        return (
            self.train
            + self.validation
            + self.test_seen_templates
            + self.test_heldout_templates
        )


def _partition_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, ...]:
    raw_counts = tuple(total * ratio for ratio in ratios)
    counts = [math.floor(value) for value in raw_counts]
    remaining = total - sum(counts)
    order = sorted(
        range(len(ratios)),
        key=lambda index: (raw_counts[index] - counts[index], -index),
        reverse=True,
    )
    for index in order[:remaining]:
        counts[index] += 1
    return tuple(counts)


def split_semantic_cases(
    semantic_cases: tuple[SemanticCase, ...],
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> SemanticCaseSplits:
    """Assign complete semantic groups using a stable seeded ordering."""

    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(not math.isfinite(ratio) or ratio < 0 for ratio in ratios):
        raise ValueError("split ratios must be finite and non-negative")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("split ratios must sum to 1")

    case_ids = [case.semantic_case_id for case in semantic_cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("semantic_cases must not contain duplicate IDs")

    ordered = tuple(
        sorted(
            semantic_cases,
            key=lambda case: hashlib.sha256(
                f"{seed}:{case.semantic_case_id}".encode("utf-8")
            ).digest(),
        )
    )
    train_count, validation_count, _ = _partition_counts(len(ordered), ratios)
    validation_end = train_count + validation_count
    return SemanticCaseSplits(
        train=ordered[:train_count],
        validation=ordered[train_count:validation_end],
        test=ordered[validation_end:],
    )


def _case_ids(samples: tuple[GeneratedSample, ...]) -> set[str]:
    return {sample.semantic_case_id for sample in samples}


def validate_no_leakage(splits: GenerateDatasetSplits) -> None:
    """Raise if semantic groups or held-out template families leak into training."""

    train_ids = _case_ids(splits.train)
    validation_ids = _case_ids(splits.validation)
    test_seen_ids = _case_ids(splits.test_seen_templates)
    test_heldout_ids = _case_ids(splits.test_heldout_templates)

    if train_ids & validation_ids or train_ids & test_seen_ids:
        raise ValueError("semantic_case_id leakage involving the training split")
    if validation_ids & test_seen_ids:
        raise ValueError("semantic_case_id leakage between validation and test")
    if test_seen_ids != test_heldout_ids:
        raise ValueError("seen and held-out template tests must use the same test cases")

    train_families = {sample.template_family for sample in splits.train}
    heldout_families = {
        sample.template_family for sample in splits.test_heldout_templates
    }
    non_holdout_families = train_families | {
        sample.template_family
        for sample in splits.validation + splits.test_seen_templates
    }
    if non_holdout_families & heldout_families:
        raise ValueError("held-out template family leaked outside its test view")

    sample_ids = [sample.sample_id for sample in splits.all_samples()]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample IDs must be unique across dataset splits")


def build_generate_splits(
    semantic_cases: tuple[SemanticCase, ...],
    *,
    heldout_template_families: tuple[str, ...] = ("flow_first_v2",),
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> GenerateDatasetSplits:
    """Build seen-template and held-out-template rows without train leakage.

    The two test views share only test semantic cases. This isolates word-order
    generalization while keeping every test meaning absent from training.
    """

    if not heldout_template_families:
        raise ValueError("at least one template family must be held out")
    if len(heldout_template_families) != len(set(heldout_template_families)):
        raise ValueError("heldout_template_families must not contain duplicates")
    unknown = set(heldout_template_families).difference(TEMPLATE_FAMILIES)
    if unknown:
        raise ValueError(f"unknown held-out template families: {sorted(unknown)}")

    seen_families = tuple(
        family
        for family in TEMPLATE_FAMILIES
        if family not in heldout_template_families
    )
    if not seen_families:
        raise ValueError("at least one template family must remain for training")

    case_splits = split_semantic_cases(
        semantic_cases,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    splits = GenerateDatasetSplits(
        train=render_cases(case_splits.train, seen_families, seed),
        validation=render_cases(case_splits.validation, seen_families, seed),
        test_seen_templates=render_cases(case_splits.test, seen_families, seed),
        test_heldout_templates=render_cases(
            case_splits.test, heldout_template_families, seed
        ),
    )
    validate_no_leakage(splits)
    return splits
