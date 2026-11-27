"""Structured health, failure, and decision observability metrics for operator runbooks."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class EngineMetricsRecorder:
    """Captures stage timings, validation counters, promotion decisions, and structured failures."""

    def __init__(self) -> None:
        self.started_at_utc = datetime.now(UTC)
        self.counters: dict[str, int] = defaultdict(int)
        self.stage_durations_ms: dict[str, float] = {}
        self.decisions: list[dict[str, Any]] = []
        self.failures: list[dict[str, Any]] = []

    def increment(self, metric_name: str, amount: int = 1) -> None:
        self.counters[metric_name] += amount

    def record_decision(self, decision_type: str, subject: str, outcome: str, details: dict[str, Any]) -> None:
        self.decisions.append(
            {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "decision_type": decision_type,
                "subject": subject,
                "outcome": outcome,
                "details": details,
            }
        )

    def record_failure(self, stage: str, error_code: str, message: str, details: dict[str, Any] | None = None) -> None:
        self.increment("errors_total")
        self.failures.append(
            {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "stage": stage,
                "error_code": error_code,
                "message": message,
                "details": details or {},
            }
        )

    @contextmanager
    def time_stage(self, stage_name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.stage_durations_ms[stage_name] = round(elapsed_ms, 2)

    def snapshot(self) -> dict[str, Any]:
        return {
            "started_at_utc": self.started_at_utc.isoformat(),
            "captured_at_utc": datetime.now(UTC).isoformat(),
            "status": "healthy" if self.counters.get("fatal_errors", 0) == 0 else "degraded",
            "counters": dict(sorted(self.counters.items())),
            "stage_durations_ms": dict(sorted(self.stage_durations_ms.items())),
            "decisions": self.decisions,
            "failures": self.failures,
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.snapshot(), fh, indent=2, sort_keys=True)
        return path
