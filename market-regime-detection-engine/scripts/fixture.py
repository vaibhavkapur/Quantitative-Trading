"""Produce synthetic weekday closes. No provider data or third-party fixtures."""
import argparse
import csv
import hashlib
import io
import json
import math
import random
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 80717


def fixture():
    rng = random.Random(SEED)
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(("series", "event_at", "available_at", "close", "benchmark_close", "spread_bps"))
    day = date(2023, 1, 2)
    asset, bench = 100.0, 100.0
    for i in range(455):
        while day.weekday() >= 5:
            day += timedelta(days=1)
        state = (i // 37) % 3
        mu, sigma, spread, corr = ((0.0018, .004, 3, .32), (-.002, .005, 5, .60),
                                    (-.001, .022, 18, .86))[state]
        if 385 <= i < 410:
            sigma, spread, corr = .044, 44, .95
        common, own = rng.gauss(0, 1), rng.gauss(0, 1)
        asset *= math.exp(mu + sigma * (corr * common + math.sqrt(1 - corr ** 2) * own))
        bench *= math.exp(.0002 + .009 * common)
        spread *= rng.uniform(.8, 1.2)
        writer.writerow(("SYNTH-USD", f"{day}T16:00:00Z", f"{day}T16:05:00Z",
                         "" if i == 332 else f"{asset:.6f}", f"{bench:.6f}", f"{spread:.5f}"))
        day += timedelta(days=1)
    return out.getvalue().encode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = fixture()
    manifest = {"schema": "bars.v1", "series": "SYNTH-USD", "rows": 455, "seed": SEED,
                "source": "Locally generated synthetic USD daily closes and quoted spread",
                "generator": "scripts/fixture.py", "entitlements": "Synthetic project fixture; no vendor inputs",
                "sha256": hashlib.sha256(payload).hexdigest()}
    for name, data in (("daily.csv", payload), ("manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())):
        path = ROOT / "data" / "fixtures" / name
        if args.check:
            if path.read_bytes() != data:
                raise SystemExit(f"fixture mismatch: {name}")
        else:
            path.write_bytes(data)
