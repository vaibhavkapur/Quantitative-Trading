"""Run the complete acceptance verification pipeline (validate, pytest, demo, replay)."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print(f"\n==> Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    py = sys.executable
    _run([py, "-m", "cross_asset_factor_engine", "validate", "--dataset", "data/fixtures/valid_small"])
    _run([py, "-m", "cross_asset_factor_engine", "validate", "--dataset", "data/fixtures/synthetic_multi_year"])
    _run([py, "-m", "pytest", "tests", "-v"])
    _run([py, "scripts/run_demo.py"])
    _run([py, "-m", "cross_asset_factor_engine", "verify-replay", "--config", "config/default.yaml"])
    print("\nALL ACCEPTANCE CHECKS PASSED.")


if __name__ == "__main__":
    main()
