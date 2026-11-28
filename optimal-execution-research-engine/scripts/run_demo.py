#!/usr/bin/env python3
"""Run the reproducible baseline multi-strategy optimal execution demo."""

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
                "demo",
                "--scenario",
                str(ROOT / "config" / "default_scenario.json"),
                "--output-dir",
                str(ROOT / "reports"),
                "--risk-aversion",
                "1e-6",
                "--target-pov",
                "0.12",
            ]
        )
    )
