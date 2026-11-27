"""Deterministic reproduction script proving identical replay hashes across fresh builds."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pms.service import PortfolioService


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    hashes: list[str] = []
    for idx in range(2):
        with tempfile.TemporaryDirectory(prefix=f"pms_repro_{idx}_") as tmp:
            svc = PortfolioService.from_workspace(
                root, sqlite_path=Path(tmp) / "repro.sqlite3", load_fixtures=True
            )
            snap, _ = svc.replay()
            hashes.append(snap.state_hash)

    identical = hashes[0] == hashes[1]
    payload = {
        "reproducible_across_clean_builds": identical,
        "run_1_state_hash": hashes[0],
        "run_2_state_hash": hashes[1],
    }
    out_path = root / "reports" / "evidence" / "reproducibility_check.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
