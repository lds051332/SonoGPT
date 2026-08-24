"""Create or verify a reviewed SonoGPT frozen data release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sonogpt.data.freeze import create_freeze_record, verify_freeze_record

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREEZE_RECORD = (
    PROJECT_ROOT
    / "data"
    / "releases"
    / "synthetic_v1_5k_frozen_v1.freeze.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--freeze-record",
        type=Path,
        default=DEFAULT_FREEZE_RECORD,
    )
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "synthetic_v1_5k_candidate_v2",
    )
    parser.add_argument(
        "--data-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "manifests"
        / "synthetic_v1_5k_candidate_v2.manifest.json",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "tokenizers"
        / "sonogpt_bpe_1807_candidate_v2"
        / "sonogpt_bpe.model",
    )
    parser.add_argument(
        "--tokenizer-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "tokenizers"
        / "sonogpt_bpe_1807_candidate_v2"
        / "tokenizer_manifest.json",
    )
    parser.add_argument(
        "--freeze-id",
        default="synthetic_v1_5k_frozen_v1",
    )
    parser.add_argument(
        "--review-outcome",
        choices=("approved_no_changes", "approved_after_changes"),
    )
    parser.add_argument("--review-sample-count", type=int)
    parser.add_argument("--reviewer-role")
    parser.add_argument("--review-date")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify an existing record instead of creating one",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify:
        record = verify_freeze_record(
            args.freeze_record,
            project_root=PROJECT_ROOT,
            verify_provenance=True,
        )
    else:
        required = {
            "--review-outcome": args.review_outcome,
            "--review-sample-count": args.review_sample_count,
            "--reviewer-role": args.reviewer_role,
            "--review-date": args.review_date,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "creating a freeze record requires " + ", ".join(missing)
            )
        record = create_freeze_record(
            freeze_id=args.freeze_id,
            project_root=PROJECT_ROOT,
            data_directory=args.data_directory,
            data_manifest_path=args.data_manifest,
            tokenizer_model_path=args.tokenizer_model,
            tokenizer_manifest_path=args.tokenizer_manifest,
            review_outcome=args.review_outcome,
            review_sample_count=args.review_sample_count,
            reviewer_role=args.reviewer_role,
            review_date=args.review_date,
            output_path=args.freeze_record,
        )
    print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
