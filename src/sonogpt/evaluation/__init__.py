"""Evaluation datasets and metrics kept separate from model training."""

from sonogpt.evaluation.challenge import (
    CHALLENGE_AUTHORING_METHOD,
    CHALLENGE_SET_VERSION,
    ChallengeSample,
    challenge_id_for,
    verify_challenge_freeze,
    write_frozen_challenge_set,
)
from sonogpt.evaluation.metrics import aggregate_scores, score_generated_report
from sonogpt.evaluation.pipeline import (
    ALL_SPLITS,
    EvaluationExample,
    evaluate_examples,
    run_generate_evaluation,
)
from sonogpt.evaluation.report_parser import parse_report

__all__ = [
    "ALL_SPLITS",
    "CHALLENGE_AUTHORING_METHOD",
    "CHALLENGE_SET_VERSION",
    "ChallengeSample",
    "EvaluationExample",
    "aggregate_scores",
    "challenge_id_for",
    "evaluate_examples",
    "parse_report",
    "run_generate_evaluation",
    "score_generated_report",
    "verify_challenge_freeze",
    "write_frozen_challenge_set",
]
