"""Versioned contracts for one ordered, point-in-time market series."""
import csv
import json
import math
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Optional

FEATURES = ("annual_vol", "trend_t", "spread_bps", "benchmark_corr")
STATES = ("rising", "falling", "stressed")
ENGINE_VERSION = "1.0"
FEATURE_VERSION = "trailing-daily.v1"


class ContractError(ValueError):
    """Expected operator-visible input or configuration error."""


def timestamp(value):
    try:
        dt = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        if dt.strftime("%Y-%m-%dT%H:%M:%SZ") == value:
            return dt
    except (ValueError, TypeError):
        pass
    raise ContractError("expected a UTC timestamp YYYY-MM-DDTHH:MM:SSZ")


@dataclass(frozen=True)
class Settings:
    version: str
    series: str
    train_through: str
    window: int
    max_gap_days: int
    stress_vol: float
    stress_spread_bps: float
    confidence_floor: float
    confirm_bars: int
    ood_rms: float
    em_iterations: int
    cost_bps: float

    def __post_init__(self):
        if self.version != "daily.v1" or not isinstance(self.series, str) or not self.series:
            raise ContractError("invalid settings version or series")
        timestamp(self.train_through)
        for name, low in (("window", 3), ("confirm_bars", 1), ("max_gap_days", 1), ("em_iterations", 1)):
            val = getattr(self, name)
            if type(val) is not int or val < low:
                raise ContractError("invalid " + name)
        for name in ("stress_vol", "stress_spread_bps", "ood_rms", "cost_bps", "confidence_floor"):
            val = getattr(self, name)
            if type(val) not in (float, int) or not math.isfinite(val) or val < 0:
                raise ContractError("invalid " + name)
            if val == 0 and name != "cost_bps":
                raise ContractError("expected positive " + name)
        if self.confidence_floor > 1 or self.cost_bps >= 10000:
            raise ContractError("probability or cost out of bounds")

    @classmethod
    def load(cls, path):
        try:
            return cls(**json.loads(Path(path).read_text()))
        except TypeError as exc:
            raise ContractError("settings fields mismatch") from exc


@dataclass(frozen=True)
class Bar:
    series: str
    event_at: str
    available_at: str
    close: Optional[float]
    benchmark_close: Optional[float]
    spread_bps: Optional[float]

    def __post_init__(self):
        if not isinstance(self.series, str) or not self.series:
            raise ContractError("missing series")
        timestamp(self.event_at)
        timestamp(self.available_at)
        if self.event_at > self.available_at:
            raise ContractError("bar available before its event")
        for name in ("close", "benchmark_close", "spread_bps"):
            val = getattr(self, name)
            if val is None:
                continue
            if type(val) not in (float, int) or not math.isfinite(val) or val < 0 or (name != "spread_bps" and val == 0):
                raise ContractError("invalid " + name)


def read_bars(path):
    with open(path, newline="") as stream:
        yield from parse_bars(stream)


def parse_bars(stream):
    rows = csv.DictReader(stream)
    if rows.fieldnames != [field.name for field in fields(Bar)]:
        raise ContractError("invalid bar CSV header")
    for i, row in enumerate(rows, 2):
        try:
            for name in ("close", "benchmark_close", "spread_bps"):
                row[name] = None if row[name] == "" else float(row[name])
            yield Bar(**row)
        except (ValueError, TypeError) as exc:
            raise ContractError(f"bar CSV line {i}: {exc}") from exc
