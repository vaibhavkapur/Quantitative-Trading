#!/usr/bin/env python3
"""Run the end-to-end statistical arbitrage walk-forward demo and emit versioned reports."""

from __future__ import annotations

import json
from pathlib import Path

from statarb.fixtures import write_all_fixtures
from statarb.pipeline import execute_research_pipeline


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    fixtures_dir = project_root / "data" / "fixtures"
    if not (fixtures_dir / "synthetic_universe_prices.parquet").exists():
        write_all_fixtures(fixtures_dir)

    reports_dir = project_root / "reports"
    demo_report = execute_research_pipeline(
        project_root / "config" / "default.toml",
        project_root=project_root,
        output_dir=reports_dir,
        run_trial_sweep=True,
    )
    ref_report = execute_research_pipeline(
        project_root / "config" / "reference.toml",
        project_root=project_root,
        output_dir=reports_dir,
        run_trial_sweep=True,
    )
    print(
        json.dumps(
            {
                "demo_run_id": demo_report["run_id"],
                "demo_performance": demo_report["performance"],
                "demo_reconciliation": demo_report["reconciliation"],
                "reference_run_id": ref_report["run_id"],
                "reference_performance": ref_report["performance"],
                "reference_reconciliation": ref_report["reconciliation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
