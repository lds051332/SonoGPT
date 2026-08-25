"""Start the local SonoGPT demo website."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from sonogpt.inference.engine import InferenceEngine
from sonogpt.web.app import create_app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREEZE_RECORD = (
    PROJECT_ROOT / "data" / "releases" / "synthetic_v1_5k_frozen_v1.freeze.json"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "artifacts" / "training" / "sonogpt_16m_m3" / "step_00004900.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SonoGPT local website (learning demo, not clinical use)"
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--freeze-record", type=Path, default=DEFAULT_FREEZE_RECORD)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--preload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="load the 15M checkpoint at startup (default: true)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = InferenceEngine(
        project_root=args.project_root,
        freeze_record_path=args.freeze_record,
        checkpoint_path=args.checkpoint,
        device_name=args.device,
    )
    preload = bool(args.preload and args.checkpoint.is_file())
    if not args.checkpoint.is_file():
        print(
            "checkpoint missing; template preview still works. "
            "Place step_00004900.pt under artifacts/training/sonogpt_16m_m3/ (not stored in Git)."
        )
    app = create_app(engine, preload=preload)
    print("SonoGPT local demo — not for clinical diagnosis", flush=True)
    print(f"open http://{args.host}:{args.port}/", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
