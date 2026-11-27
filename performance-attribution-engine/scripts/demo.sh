#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[PAE demo] Running reproducible end-to-end attribution demo (v1 -> review -> approve -> v2 correction)..."
PYTHONPATH=src python3 scripts/run_demo_and_evidence.py --demo-only
echo "[PAE demo] Complete. Exports written to reports/demo_exports/"
