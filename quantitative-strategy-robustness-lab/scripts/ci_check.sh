#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi

echo "==> [1/4] Generating deterministic fixtures"
PYTHONPATH="${PROJECT_DIR}/src" "${PYTHON}" "${PROJECT_DIR}/scripts/generate_fixtures.py"

echo "==> [2/4] Running unit, integration, boundary, and reproducibility test suite"
PYTHONPATH="${PROJECT_DIR}/src" "${PYTHON}" -m pytest "${PROJECT_DIR}/tests" -v

echo "==> [3/4] Running reproducible end-to-end demo & replay check"
PYTHONPATH="${PROJECT_DIR}/src" "${PYTHON}" "${PROJECT_DIR}/scripts/run_demo.py"

echo "==> [4/4] Verifying CLI verify-registry and replay commands"
PYTHONPATH="${PROJECT_DIR}/src" "${PYTHON}" -m robustness_lab.cli verify-registry \
  --ledger "${PROJECT_DIR}/data/fixtures/valid/known_signal_experiment/registry.jsonl"
PYTHONPATH="${PROJECT_DIR}/src" "${PYTHON}" -m robustness_lab.cli replay \
  --experiment-dir "${PROJECT_DIR}/data/fixtures/valid/known_signal_experiment" \
  --policy "${PROJECT_DIR}/config/default_policy.json" \
  --baseline-report "${PROJECT_DIR}/reports/baseline/known_signal.json"

echo "ALL CI CHECKS PASSED"
