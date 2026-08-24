"""Run visible RTX batch-size A/B benchmarks and aggregate results."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro-batch-sizes", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--effective-batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "training"
        / "rtx2060_batch_ab_20260824",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "benchmarks"
        / "rtx2060_batch_ab_20260824.json",
    )
    parser.add_argument("--max-gpu-temperature", type=float, default=85.0)
    return parser.parse_args()


def _emit(payload: object) -> None:
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.effective_batch_size <= 0:
        raise ValueError("steps and effective batch size must be positive")
    if len(args.micro_batch_sizes) != len(set(args.micro_batch_sizes)):
        raise ValueError("micro batch sizes must be unique")

    results: list[dict[str, object]] = []
    for run_index, micro_batch_size in enumerate(
        args.micro_batch_sizes, start=1
    ):
        if (
            micro_batch_size <= 0
            or args.effective_batch_size % micro_batch_size != 0
        ):
            raise ValueError(
                "each micro batch size must divide effective batch size"
            )
        accumulation_steps = (
            args.effective_batch_size // micro_batch_size
        )
        output_directory = args.output_root / f"micro_{micro_batch_size}"
        if (output_directory / "latest.json").exists():
            raise FileExistsError(
                f"benchmark output already exists: {output_directory}"
            )
        _emit(
            {
                "event": "benchmark_variant_start",
                "variant": run_index,
                "variant_count": len(args.micro_batch_sizes),
                "micro_batch_size": micro_batch_size,
                "gradient_accumulation_steps": accumulation_steps,
                "steps": args.steps,
            }
        )
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "train_model.py"),
                "--device",
                "cuda",
                "--max-steps",
                str(args.steps),
                "--micro-batch-size",
                str(micro_batch_size),
                "--gradient-accumulation-steps",
                str(accumulation_steps),
                "--max-gpu-temperature",
                str(args.max_gpu_temperature),
                "--output-directory",
                str(output_directory),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
        summary = json.loads(
            (output_directory / "summary.json").read_text(encoding="utf-8")
        )
        variant = {
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "effective_batch_size": args.effective_batch_size,
            "output_directory": str(output_directory),
            "elapsed_seconds": summary["elapsed_seconds"],
            "runtime": summary["runtime"],
            "cuda_memory": summary["cuda_memory"],
            "gpu_telemetry": summary["gpu_telemetry"],
            "validation": summary["final_validation"],
            "skipped_optimizer_steps": summary["state"][
                "skipped_optimizer_steps"
            ],
        }
        results.append(variant)
        _emit({"event": "benchmark_variant_complete", **variant})

    eligible = [
        result
        for result in results
        if result["skipped_optimizer_steps"] == 0
        and result["gpu_telemetry"]["max_temperature_c"]  # type: ignore[index]
        < args.max_gpu_temperature
    ]
    if not eligible:
        raise RuntimeError("no batch-size variant passed safety checks")
    recommended = max(
        eligible,
        key=lambda result: result["runtime"][  # type: ignore[index]
            "target_tokens_per_second"
        ],
    )
    report = {
        "benchmark_version": "1.0.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "device": "NVIDIA GeForce RTX 2060 Laptop",
        "steps_per_variant": args.steps,
        "effective_batch_size": args.effective_batch_size,
        "max_gpu_temperature_c": args.max_gpu_temperature,
        "results": results,
        "recommended_micro_batch_size": recommended["micro_batch_size"],
        "recommended_gradient_accumulation_steps": recommended[
            "gradient_accumulation_steps"
        ],
        "selection_metric": "target_tokens_per_second",
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _emit({"event": "benchmark_complete", "report": report})


if __name__ == "__main__":
    main()
