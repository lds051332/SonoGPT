from __future__ import annotations

from sonogpt.data.renderers import TEMPLATE_FAMILIES
from sonogpt.data.semantic_generator import sample_semantic_cases
from sonogpt.data.split import build_generate_splits, split_semantic_cases


def _ids(samples: tuple[object, ...]) -> set[str]:
    return {sample.semantic_case_id for sample in samples}  # type: ignore[attr-defined]


def test_semantic_case_split_is_deterministic_and_grouped() -> None:
    cases = sample_semantic_cases(30, seed=7)
    first = split_semantic_cases(cases, seed=99)
    second = split_semantic_cases(cases, seed=99)

    assert first == second
    assert (len(first.train), len(first.validation), len(first.test)) == (24, 3, 3)
    train_ids = {case.semantic_case_id for case in first.train}
    validation_ids = {case.semantic_case_id for case in first.validation}
    test_ids = {case.semantic_case_id for case in first.test}
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)


def test_generate_split_holds_out_meanings_and_template_family() -> None:
    cases = sample_semantic_cases(30, seed=7)
    splits = build_generate_splits(
        cases,
        heldout_template_families=("flow_first_v2",),
        seed=99,
    )

    assert len(splits.train) == 24 * 3
    assert len(splits.validation) == 3 * 3
    assert len(splits.test_seen_templates) == 3 * 3
    assert len(splits.test_heldout_templates) == 3

    train_ids = _ids(splits.train)
    validation_ids = _ids(splits.validation)
    test_ids = _ids(splits.test_seen_templates)
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
    assert test_ids == _ids(splits.test_heldout_templates)

    assert {
        sample.template_family for sample in splits.train
    } == set(TEMPLATE_FAMILIES) - {"flow_first_v2"}
    assert {
        sample.template_family for sample in splits.test_heldout_templates
    } == {"flow_first_v2"}


def test_all_rewrites_of_a_case_keep_the_same_canonical_input() -> None:
    splits = build_generate_splits(sample_semantic_cases(20, seed=11), seed=12)
    inputs_by_case: dict[str, set[str]] = {}

    for sample in splits.all_samples():
        inputs_by_case.setdefault(sample.semantic_case_id, set()).add(sample.input)

    assert all(len(inputs) == 1 for inputs in inputs_by_case.values())
