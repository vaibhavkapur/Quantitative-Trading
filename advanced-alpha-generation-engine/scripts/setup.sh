#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if command -v uv >/dev/null 2>&1; then
  UV_CMD="uv"
else
  UV_CMD="python3 -m uv"
fi

echo "==> Syncing Python dependencies from uv.lock..."
$UV_CMD sync --python 3.12 --extra dev --extra boosting

echo "==> Generating deterministic synthetic PIT fixtures..."
.venv/bin/python data/fixtures/generate_fixtures.py

echo "==> Setup complete. Run '.venv/bin/python scripts/validate.py' to verify tests and demo."
