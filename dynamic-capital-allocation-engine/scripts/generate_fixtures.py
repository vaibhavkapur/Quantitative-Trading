#!/usr/bin/env python3
"""Generate deterministic synthetic fixtures and manifest with SHA-256 checksums."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dcae.fixtures import (  # noqa: E402
    build_baseline_snapshot,
    build_scenario_fixtures,
    generate_walkforward_strategies,
)


def _write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    fixtures_dir = ROOT / "data" / "fixtures"
    manifest_entries = []

    # 1. Baseline snapshot
    base_snap = build_baseline_snapshot()
    base_path = fixtures_dir / "baseline_snapshot.json"
    manifest_entries.append(
        {
            "file": str(base_path.relative_to(ROOT)),
            "kind": "baseline_snapshot",
            "entitlement": base_snap.provenance.entitlement,
            "seed": base_snap.provenance.seed,
            "sha256": _write_json(base_path, base_snap.to_dict()),
        }
    )

    # 2. Walk-forward dataset
    wf_strats, wf_prov = generate_walkforward_strategies()
    wf_payload = {
        "provenance": wf_prov.to_dict(),
        "strategies": [
            {
                "strategy_id": s.strategy_id,
                "name": s.name,
                "eligible": s.eligible,
                "risk_budget": s.risk_budget,
                "exposure": s.exposure.to_dict(),
                "returns": [r.to_dict() for r in s.returns],
            }
            for s in wf_strats
        ],
    }
    wf_path = fixtures_dir / "walkforward_dataset.json"
    manifest_entries.append(
        {
            "file": str(wf_path.relative_to(ROOT)),
            "kind": "walkforward_dataset",
            "entitlement": wf_prov.entitlement,
            "seed": wf_prov.seed,
            "sha256": _write_json(wf_path, wf_payload),
        }
    )

    # 3. Scenario fixtures
    for name, snap in sorted(build_scenario_fixtures().items()):
        spath = fixtures_dir / "scenarios" / f"{name}.json"
        manifest_entries.append(
            {
                "file": str(spath.relative_to(ROOT)),
                "kind": f"scenario:{name}",
                "entitlement": snap.provenance.entitlement,
                "seed": snap.provenance.seed,
                "sha256": _write_json(spath, snap.to_dict()),
            }
        )

    manifest = {
        "manifest_version": "1.0.0",
        "license_and_ip_notice": (
            "All fixture datasets in this directory are synthetically generated "
            "for local reproducible testing. No proprietary or unlicensed market "
            "data is included."
        ),
        "files": manifest_entries,
    }
    _write_json(fixtures_dir / "manifest.json", manifest)
    print(f"Generated {len(manifest_entries)} fixture files and manifest.json in {fixtures_dir}")


if __name__ == "__main__":
    main()
