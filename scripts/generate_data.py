"""Generate the reproducible SonoGPT V1 JSON-to-report dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sonogpt.data.manifest import sha256_file, write_dataset_bundle
from sonogpt.data.semantic_generator import DEFAULT_SEED, sample_semantic_cases
from sonogpt.data.split import build_generate_splits

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "synthetic_v1_small",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "manifests"
        / "synthetic_v1_small.manifest.json",
    )
    parser.add_argument(
        "--heldout-template-families",
        nargs="+",
        default=["flow_first_v2"],
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = sample_semantic_cases(args.count, args.seed)
    splits = build_generate_splits(
        cases,
        heldout_template_families=tuple(args.heldout_template_families),
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    manifest = write_dataset_bundle(
        splits,
        output_directory=args.output_directory,
        manifest_path=args.manifest_path,
    )
    summary = {
        "manifest": str(args.manifest_path),
        "manifest_sha256": sha256_file(args.manifest_path),
        "semantic_case_count": manifest["semantic_case_count"],
        "files": manifest["files"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
