"""Run local SonoGPT inference: generate, rule extract, or QC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sonogpt.inference.engine import InferenceEngine
from sonogpt.schemas.domain import ThyroidExam

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREEZE_RECORD = (
    PROJECT_ROOT / "data" / "releases" / "synthetic_v1_5k_frozen_v1.freeze.json"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "artifacts" / "training" / "sonogpt_16m_m3" / "step_00004900.pt"
)


def _read_text_argument(value: str | None, path: Path | None) -> str:
    if path is not None:
        return path.read_text(encoding="utf-8").strip()
    if value == "-":
        return sys.stdin.read().strip()
    if value:
        return value.strip()
    raise ValueError("provide --text, --text-file, or '-' for stdin")


def _read_json_argument(value: str | None, path: Path | None) -> dict[str, Any]:
    if path is not None:
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif value == "-":
        payload = json.loads(sys.stdin.read())
    elif value:
        payload = json.loads(value)
    else:
        raise ValueError("provide --json, --json-file, or '-' for stdin")
    if not isinstance(payload, dict):
        raise ValueError("JSON input must be an object")
    return payload


def _engine(args: argparse.Namespace) -> InferenceEngine:
    return InferenceEngine(
        project_root=args.project_root,
        freeze_record_path=args.freeze_record,
        checkpoint_path=args.checkpoint,
        device_name=args.device,
    )


def parse_args() -> argparse.Namespace:
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    shared.add_argument("--freeze-record", type=Path, default=DEFAULT_FREEZE_RECORD)
    shared.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    shared.add_argument("--device", default="auto")
    shared.add_argument(
        "--output",
        type=Path,
        help="write JSON to this path in addition to stdout",
    )

    parser = argparse.ArgumentParser(
        description="SonoGPT local inference (learning demo, not clinical use)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "info",
        parents=[shared],
        help="print model, tokenizer, schema, and rule versions",
    )

    generate = sub.add_parser(
        "generate",
        parents=[shared],
        help="JSON → report with QC and template fallback",
    )
    generate.add_argument("--json")
    generate.add_argument("--json-file", type=Path)
    generate.add_argument("--no-fallback", action="store_true")

    extract = sub.add_parser(
        "extract",
        parents=[shared],
        help="report → JSON using the rule baseline",
    )
    extract.add_argument("--text")
    extract.add_argument("--text-file", type=Path)

    qc = sub.add_parser(
        "qc",
        parents=[shared],
        help="run QC on a report, optionally with structure",
    )
    qc.add_argument("--text")
    qc.add_argument("--text-file", type=Path)
    qc.add_argument("--json")
    qc.add_argument("--json-file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = _engine(args)
    if args.command == "info":
        payload: dict[str, Any] = engine.info(load_model=True).to_dict()
    elif args.command == "generate":
        exam = ThyroidExam.model_validate(
            _read_json_argument(args.json, args.json_file)
        )
        payload = engine.generate(
            exam, fallback_template=not args.no_fallback
        )
    elif args.command == "extract":
        payload = engine.extract(_read_text_argument(args.text, args.text_file))
    elif args.command == "qc":
        structure = None
        if args.json or args.json_file:
            structure = ThyroidExam.model_validate(
                _read_json_argument(args.json, args.json_file)
            )
        payload = engine.qc(
            _read_text_argument(args.text, args.text_file),
            structure,
        )
    else:
        raise ValueError(f"unknown command: {args.command}")
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
