"""Evaluate frozen generate-task checkpoints on template and challenge splits."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from sonogpt.evaluation.pipeline import ALL_SPLITS, run_generate_evaluation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREEZE_RECORD = (
    PROJECT_ROOT / "data" / "releases" / "synthetic_v1_5k_frozen_v1.freeze.json"
)
DEFAULT_CHALLENGE_FREEZE = (
    PROJECT_ROOT
    / "data"
    / "releases"
    / "simulated_human_challenge_v1.freeze.json"
)
DEFAULT_PRIMARY = (
    PROJECT_ROOT / "artifacts" / "training" / "sonogpt_16m_m3" / "step_00004900.pt"
)
DEFAULT_COMPARISON = (
    PROJECT_ROOT / "artifacts" / "training" / "sonogpt_16m_m3" / "step_00005000.pt"
)


def parse_args() -> argparse.Namespace:
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--freeze-record", type=Path, default=DEFAULT_FREEZE_RECORD)
    parser.add_argument(
        "--challenge-freeze-record",
        type=Path,
        default=DEFAULT_CHALLENGE_FREEZE,
    )
    parser.add_argument("--primary-checkpoint", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument(
        "--comparison-checkpoint",
        type=Path,
        default=DEFAULT_COMPARISON,
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="evaluate only the primary checkpoint",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=ALL_SPLITS,
        default=list(ALL_SPLITS),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, help="optional per-split sample cap")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "evaluation"
            / f"sonogpt_16m_m3_generate_{stamp}.json"
        ),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "evaluation"
            / f"sonogpt_16m_m3_generate_{stamp}.md"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = [args.primary_checkpoint]
    if not args.skip_comparison:
        checkpoints.append(args.comparison_checkpoint)

    def progress(completed: int, total: int, split: str) -> None:
        if completed == 1 or completed == total or completed % 50 == 0:
            print(f"{split}: {completed}/{total}", flush=True)

    report = run_generate_evaluation(
        project_root=args.project_root,
        freeze_record_path=args.freeze_record,
        challenge_freeze_path=args.challenge_freeze_record,
        checkpoint_paths=checkpoints,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        device_name=args.device,
        limit=args.limit,
        batch_size=args.batch_size,
        splits=args.splits,
        progress_callback=progress,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
