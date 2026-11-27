#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[PAE setup] Checking Python runtime..."
python3 -c 'import sys; assert sys.version_info >= (3, 9), "Python 3.9+ required"; print(f"Python {sys.version.split()[0]} OK")'

echo "[PAE setup] Generating deterministic synthetic fixtures..."
python3 data/fixtures/generate_fixtures.py

echo "[PAE setup] Verifying package import..."
PYTHONPATH=src python3 -c 'import pae; from pae.config import AttributionConfig; cfg = AttributionConfig.from_file(); print(f"PAE {pae.__version__} methodology={cfg.methodology_version} config_hash={cfg.config_hash()[:12]}")'
echo "[PAE setup] Complete."
