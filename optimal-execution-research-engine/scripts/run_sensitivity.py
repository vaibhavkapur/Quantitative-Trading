#!/usr/bin/env python3
"""Run liquidity and urgency regime sensitivity analysis and write reports."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from optimal_execution.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "sensitivity",
                "--scenario",
                str(ROOT / "config" / "default_scenario.json"),
                "--grid",
                str(ROOT / "config" / "sensitivity_grid.json"),
                "--output-dir",
                str(ROOT / "reports"),
            ]
        )
    )
