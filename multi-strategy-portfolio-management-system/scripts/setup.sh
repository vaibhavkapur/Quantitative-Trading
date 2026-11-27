#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if command -v uv >/dev/null 2>&1; then
  UV_BIN="uv"
else
  UV_BIN="python3 -m uv"
fi

echo "==> Syncing Python 3.12 virtual environment with dev extras..."
$UV_BIN sync --extra dev --offline 2>/dev/null || $UV_BIN sync --extra dev
echo "==> Setup complete. Activate with: source .venv/bin/activate"
