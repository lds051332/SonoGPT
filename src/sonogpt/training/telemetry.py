"""Lightweight NVIDIA telemetry sampling for local training benchmarks."""

from __future__ import annotations

import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GpuTelemetrySample:
    elapsed_seconds: float
    temperature_c: float
    utilization_percent: float
    memory_used_mib: float
    power_w: float | None

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


class NvidiaSmiMonitor:
    def __init__(self, *, device_index: int = 0, interval_seconds: float = 1.0):
        if device_index < 0 or interval_seconds <= 0:
            raise ValueError("invalid telemetry monitor configuration")
        self.device_index = device_index
        self.interval_seconds = interval_seconds
        self.samples: list[GpuTelemetrySample] = []
        self.query_errors = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0

    @staticmethod
    def parse_query_row(
        row: str, *, elapsed_seconds: float
    ) -> GpuTelemetrySample:
        values = [value.strip() for value in row.strip().split(",")]
        if len(values) != 4:
            raise ValueError("unexpected nvidia-smi query output")
        power_w = None if values[3] == "[N/A]" else float(values[3])
        return GpuTelemetrySample(
            elapsed_seconds=elapsed_seconds,
            temperature_c=float(values[0]),
            utilization_percent=float(values[1]),
            memory_used_mib=float(values[2]),
            power_w=power_w,
        )

    def _query(self) -> GpuTelemetrySample:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={self.device_index}",
                "--query-gpu=temperature.gpu,utilization.gpu,"
                "memory.used,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return self.parse_query_row(
            completed.stdout,
            elapsed_seconds=time.perf_counter() - self._started_at,
        )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.samples.append(self._query())
            except (OSError, ValueError, subprocess.SubprocessError):
                self.query_errors += 1
            self._stop_event.wait(self.interval_seconds)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("telemetry monitor has already been started")
        self._started_at = time.perf_counter()
        self._thread = threading.Thread(
            target=self._run,
            name="sonogpt-nvidia-smi-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=max(10.0, self.interval_seconds * 2))
        if self._thread.is_alive():
            raise RuntimeError("telemetry monitor did not stop")

    def summary(self) -> dict[str, int | float | None]:
        if not self.samples:
            return {
                "sample_count": 0,
                "query_errors": self.query_errors,
                "max_temperature_c": None,
                "mean_temperature_c": None,
                "max_utilization_percent": None,
                "mean_utilization_percent": None,
                "max_memory_used_mib": None,
                "max_power_w": None,
                "mean_power_w": None,
            }
        power_values = [
            sample.power_w
            for sample in self.samples
            if sample.power_w is not None
        ]
        return {
            "sample_count": len(self.samples),
            "query_errors": self.query_errors,
            "max_temperature_c": max(
                sample.temperature_c for sample in self.samples
            ),
            "mean_temperature_c": statistics.fmean(
                sample.temperature_c for sample in self.samples
            ),
            "max_utilization_percent": max(
                sample.utilization_percent for sample in self.samples
            ),
            "mean_utilization_percent": statistics.fmean(
                sample.utilization_percent for sample in self.samples
            ),
            "max_memory_used_mib": max(
                sample.memory_used_mib for sample in self.samples
            ),
            "max_power_w": max(power_values) if power_values else None,
            "mean_power_w": (
                statistics.fmean(power_values) if power_values else None
            ),
        }
