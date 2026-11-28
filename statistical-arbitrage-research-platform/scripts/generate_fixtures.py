#!/usr/bin/env python3
"""Generate deterministic synthetic fixtures into data/fixtures/."""

from __future__ import annotations

import json
from pathlib import Path

from statarb.fixtures import write_all_fixtures


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    fixtures_dir = project_root / "data" / "fixtures"
    manifest = write_all_fixtures(fixtures_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
