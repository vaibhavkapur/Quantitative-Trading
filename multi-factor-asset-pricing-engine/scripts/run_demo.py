#!/usr/bin/env python3
"""Run the reproducible reference and synthetic multi-factor demos and verify replay."""

from __future__ import annotations

import json
from pathlib import Path

from mfape.engine import AssetPricingEngine

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    engine = AssetPricingEngine()

    print("=== 1. Running hand-calculated reference fixture ===")
    ref_out, ref_written = engine.run_from_config_file(
        PROJECT_ROOT / "config" / "reference_config.json",
        base_dir=PROJECT_ROOT,
        write_outputs=True,
    )
    print(f"Reference Run ID: {ref_out.run_id}")
    print(f"Reference Artifacts: {json.dumps(ref_written, indent=2)}")

    print("\n=== 2. Running 15-year synthetic FF3 demo (CAPM vs FF3 vs FF4_MOM) ===")
    demo_out, demo_written = engine.run_from_config_file(
        PROJECT_ROOT / "config" / "default_config.json",
        base_dir=PROJECT_ROOT,
        write_outputs=True,
    )
    print(f"Demo Run ID: {demo_out.run_id}")
    print(f"Demo Artifacts: {json.dumps(demo_written, indent=2)}")

    print("\n=== 3. Verifying deterministic replay from saved manifest ===")
    replayed_out, _ = engine.replay_from_manifest(
        Path(demo_written["replay_manifest"]),
        base_dir=PROJECT_ROOT,
        verify_fingerprints=True,
        write_outputs=False,
    )
    assert replayed_out.run_id == demo_out.run_id
    assert replayed_out.comparison.rows == demo_out.comparison.rows
    print(f"Replay verified cleanly: run_id={replayed_out.run_id}")

    print("\n=== Model Comparison Summary (Synthetic FF3 Demo) ===")
    for row in demo_out.comparison.rows:
        print(
            f"  {row['model_name']:8s} | TS R2={row['ts_mean_r2']:.4f} | "
            f"TS |alpha|={row['ts_mean_abs_alpha']:.5f} | GRS p={row['ts_grs_pvalue']:.4f} | "
            f"FM CS R2={row['fm_cs_r2']:.4f} | OOS CS R2={row['oos_cs_r2']:.4f} | "
            f"OOS CS RMSE={row['oos_cs_rmse']:.5f} | OOS TS R2={row['oos_ts_r2_zero_alpha']:.4f}"
        )


if __name__ == "__main__":
    main()
