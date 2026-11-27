#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p reports/evidence

PYTHON_BIN=".venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  bash scripts/setup.sh
fi

echo "==> 1. Running Ruff lint checks..."
.venv/bin/ruff check src tests scripts | tee reports/evidence/ruff_output.txt

echo "==> 2. Running pytest unit, integration, and 8 required scenario tests..."
.venv/bin/pytest -v --junitxml=reports/evidence/pytest_junit.xml | tee reports/evidence/pytest_output.txt

echo "==> 3. Running end-to-end demo, COD report generation, and recovery benchmark..."
"$PYTHON_BIN" scripts/demo.py

echo "==> 4. Verifying cross-run reproducibility..."
"$PYTHON_BIN" scripts/reproduce.py

echo "==> All validation checks passed."
