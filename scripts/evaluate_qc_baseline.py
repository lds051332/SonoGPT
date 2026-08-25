"""Evaluate versioned QC rules on clean pairs and injected defects."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from sonogpt.evaluation.qc_baseline import run_qc_baseline_evaluation

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--freeze-record",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "releases"
        / "synthetic_v1_5k_frozen_v1.freeze.json",
    )
    parser.add_argument(
        "--challenge-freeze-record",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "releases"
        / "simulated_human_challenge_v1.freeze.json",
    )
    parser.add_argument(
        "--fixture-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "fixtures" / "single_nodule_v1.jsonl",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--defect-limit", type=int, default=20)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "baselines" / f"qc_rules_{stamp}.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=PROJECT_ROOT / "reports" / "baselines" / f"qc_rules_{stamp}.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_qc_baseline_evaluation(
        project_root=args.project_root,
        freeze_record_path=args.freeze_record,
        challenge_freeze_path=args.challenge_freeze_record,
        fixture_path=args.fixture_path,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        limit=args.limit,
        defect_limit=args.defect_limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
