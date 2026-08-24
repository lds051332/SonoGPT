"""Threaded progress heartbeats that remain visible during long phases."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressSnapshot:
    phase: str
    status: str
    completed_steps: int | None
    total_steps: int | None
    percent: float | None
    steps_per_second: float | None
    eta_seconds: float | None
    elapsed_seconds: float
    seconds_since_progress: float
    detail: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "event": "progress",
            "phase": self.phase,
            "status": self.status,
            "completed_steps": self.completed_steps,
            "total_steps": self.total_steps,
            "percent": self.percent,
            "steps_per_second": self.steps_per_second,
            "eta_seconds": self.eta_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "seconds_since_progress": self.seconds_since_progress,
            "detail": self.detail,
        }


class ProgressReporter:
    """Emit phase transitions and periodic liveness heartbeats."""

    def __init__(
        self,
        emit: Callable[[dict[str, object]], None],
        *,
        heartbeat_interval_seconds: float = 5.0,
        stall_warning_seconds: float = 60.0,
    ):
        if heartbeat_interval_seconds <= 0 or stall_warning_seconds <= 0:
            raise ValueError("progress intervals must be positive")
        self.emit = emit
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.stall_warning_seconds = stall_warning_seconds
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0
        self._last_progress_at = 0.0
        self._phase_started_at = 0.0
        self._phase_start_step: int | None = None
        self._phase = "starting"
        self._completed_steps: int | None = None
        self._total_steps: int | None = None
        self._detail: str | None = None
        self._terminal_status: str | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("progress reporter has already been started")
        now = time.perf_counter()
        self._started_at = now
        self._last_progress_at = now
        self._phase_started_at = now
        self._thread = threading.Thread(
            target=self._run,
            name="sonogpt-progress-heartbeat",
            daemon=True,
        )
        self._thread.start()
        self.emit(self.snapshot().to_dict())

    def update(
        self,
        phase: str,
        *,
        completed_steps: int | None = None,
        total_steps: int | None = None,
        detail: str | None = None,
    ) -> None:
        if not phase:
            raise ValueError("progress phase must be non-empty")
        now = time.perf_counter()
        with self._lock:
            phase_changed = phase != self._phase
            if phase_changed:
                self._phase_started_at = now
                self._phase_start_step = completed_steps
            elif self._phase_start_step is None and completed_steps is not None:
                self._phase_start_step = completed_steps
            self._phase = phase
            self._completed_steps = completed_steps
            self._total_steps = total_steps
            self._detail = detail
            self._last_progress_at = now
        if phase_changed:
            self.emit(self.snapshot().to_dict())

    def snapshot(self) -> ProgressSnapshot:
        now = time.perf_counter()
        with self._lock:
            elapsed = max(0.0, now - self._started_at)
            since_progress = max(0.0, now - self._last_progress_at)
            phase_elapsed = max(0.0, now - self._phase_started_at)
            completed_steps = self._completed_steps
            total_steps = self._total_steps
            step_delta = (
                completed_steps - self._phase_start_step
                if completed_steps is not None
                and self._phase_start_step is not None
                else None
            )
            steps_per_second = (
                step_delta / phase_elapsed
                if step_delta is not None
                and step_delta > 0
                and phase_elapsed > 0
                else None
            )
            eta_seconds = (
                (total_steps - completed_steps) / steps_per_second
                if total_steps is not None
                and completed_steps is not None
                and steps_per_second is not None
                and total_steps >= completed_steps
                else None
            )
            percent = (
                100.0 * completed_steps / total_steps
                if total_steps is not None
                and total_steps > 0
                and completed_steps is not None
                else None
            )
            status = self._terminal_status or (
                "warning_possible_stall"
                if since_progress >= self.stall_warning_seconds
                else "running"
            )
            return ProgressSnapshot(
                phase=self._phase,
                status=status,
                completed_steps=completed_steps,
                total_steps=total_steps,
                percent=percent,
                steps_per_second=steps_per_second,
                eta_seconds=eta_seconds,
                elapsed_seconds=elapsed,
                seconds_since_progress=since_progress,
                detail=self._detail,
            )

    def _run(self) -> None:
        while not self._stop_event.wait(self.heartbeat_interval_seconds):
            self.emit(self.snapshot().to_dict())

    def close(self, *, status: str) -> None:
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError("invalid terminal progress status")
        with self._lock:
            self._terminal_status = status
            self._last_progress_at = time.perf_counter()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(
                timeout=max(10.0, self.heartbeat_interval_seconds * 2)
            )
            if self._thread.is_alive():
                raise RuntimeError("progress reporter did not stop")
        self.emit(self.snapshot().to_dict())
