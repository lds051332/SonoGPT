"""Compare complete checkpoint values independent of container metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sonogpt.training.checkpoint import (
    semantic_checkpoint_sha256,
    sha256_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    first_semantic = semantic_checkpoint_sha256(args.first)
    second_semantic = semantic_checkpoint_sha256(args.second)
    report = {
        "exact_state_match": first_semantic == second_semantic,
        "first": {
            "path": str(args.first),
            "file_sha256": sha256_path(args.first),
            "semantic_sha256": first_semantic,
        },
        "second": {
            "path": str(args.second),
            "file_sha256": sha256_path(args.second),
            "semantic_sha256": second_semantic,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["exact_state_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
