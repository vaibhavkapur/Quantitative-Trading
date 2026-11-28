"""Structured diagnostics, event logging, and operational counters."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from typing import Any


@dataclass
class DiagnosticEvent:
    """Single structured log event."""

    timestamp_utc: str
    level: str
    event: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "level": self.level,
            "event": self.event,
            "details": self.details,
        }


class DiagnosticTracker:
    """Collects structured JSON events and counters across pipeline stages."""

    def __init__(self, *, deterministic_clock: bool = False) -> None:
        self._counters: Counter[str] = Counter()
        self._events: list[DiagnosticEvent] = []
        self._deterministic_clock = deterministic_clock
        self._seq = 0

    def inc(self, name: str, amount: int = 1) -> None:
        self._counters[name] += amount

    def set_counter(self, name: str, value: int) -> None:
        self._counters[name] = value

    def get_counter(self, name: str) -> int:
        return int(self._counters.get(name, 0))

    def log(self, level: str, event: str, **details: Any) -> None:
        self._seq += 1
        ts = (
            f"seq:{self._seq:06d}"
            if self._deterministic_clock
            else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        self._events.append(
            DiagnosticEvent(
                timestamp_utc=ts,
                level=level.upper(),
                event=event,
                details=details,
            )
        )

    @property
    def counters(self) -> dict[str, int]:
        return {k: int(v) for k, v in sorted(self._counters.items())}

    @property
    def events(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._events]

    def to_dict(self) -> dict[str, Any]:
        return {
            "counters": self.counters,
            "event_count": len(self._events),
            "events": self.events,
        }

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(e.to_dict(), sort_keys=True) for e in self._events)
