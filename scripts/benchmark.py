"""Measure local fit and online inference on the permitted fixture."""
import json
import platform
import sqlite3
from math import ceil
from pathlib import Path
from statistics import median
from time import perf_counter_ns

from regime.domain import Settings, read_bars
from regime.engine import OnlineEngine

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = Settings.load(ROOT / "config/daily.v1.json")
    bars = list(read_bars(ROOT / "data/fixtures/daily.csv"))
    train = [b for b in bars if b.available_at <= cfg.train_through]
    samples, fits = [], []
    for _ in range(5):
        start = perf_counter_ns()
        engine, _rows = OnlineEngine.fit(train, cfg)
        fits.append((perf_counter_ns() - start) / 1e6)
        for bar in bars[len(train):]:
            start = perf_counter_ns()
            engine.observe(bar)
            samples.append((perf_counter_ns() - start) / 1e3)
    print(json.dumps({"python": platform.python_version(), "system": platform.system(),
                      "architecture": platform.machine(), "sqlite": sqlite3.sqlite_version, "runs": len(fits),
                      "training_bars_per_run": len(train), "online_calls": len(samples),
                      "fit_ms_each": fits, "online_us_p50": median(samples),
                      "online_us_p95": sorted(samples)[ceil(.95*len(samples))-1]}, indent=2))


if __name__ == "__main__":
    main()
