#!/usr/bin/env python3
"""Reproducible demo script: generates deterministic fixtures and runs the full benchmark."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from volatility_forecasting.fixtures import write_all_fixtures
from volatility_forecasting.pipeline import run_pipeline


def main() -> int:
    fixtures_dir = ROOT / "data" / "fixtures"
    write_all_fixtures(fixtures_dir)

    config_path = ROOT / "config" / "default.yaml"
    result = run_pipeline(config_path, base_dir=ROOT, export_reports=True)

    print(f"Run ID:            {result.run_id}")
    print(f"Config Fingerprint:{result.config_fingerprint}")
    print(f"Horizon (days):    {result.horizon_days}")
    print(f"Splits:            {len(result.splits)}")
    print(f"All Nonnegative:   {bool((result.model_comparison['min_pred_var'] >= 0.0).all())}")
    print("\nOOS Model Comparison (sorted by QLIKE):")
    cols = [
        "model",
        "qlike",
        "qlike_vs_baseline_pct",
        "mae_vol",
        "mae_vol_vs_baseline_pct",
        "mz_slope",
        "coverage_80",
        "min_pred_var",
    ]
    print(result.model_comparison.select(cols))
    print("\nExported Artifacts:")
    for key, val in sorted(result.output_paths.items()):
        print(f"  - {key}: {val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
