#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[PAE validate] Running unit, integration, scenario tests and evidence generator..."
PYTHONPATH=src python3 scripts/run_demo_and_evidence.py
echo "[PAE validate] Complete. See reports/validation_report.md and reports/test_report.txt"
