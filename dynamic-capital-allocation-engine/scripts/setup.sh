#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[DCAE setup] Checking Python runtime..."
python3 --version
python3 -c 'import sys; assert sys.version_info >= (3, 9), "Python 3.9+ required"'

echo "[DCAE setup] Generating deterministic synthetic fixtures..."
PYTHONPATH="$ROOT_DIR/src" python3 scripts/generate_fixtures.py

echo "[DCAE setup] Verifying import of dcae package..."
PYTHONPATH="$ROOT_DIR/src" python3 -c "import dcae; print('dcae version:', dcae.__version__)"

echo "[DCAE setup] Complete. Run tests with: PYTHONPATH=src python3 -m unittest discover -s tests -v"
