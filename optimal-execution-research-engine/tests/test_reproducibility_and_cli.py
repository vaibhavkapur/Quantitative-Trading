"""Reproducibility, replay manifest, stable identifiers, and CLI operational tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from optimal_execution.cli import main as cli_main
from optimal_execution.contracts import ScheduleRequest, StrategyType
from optimal_execution.io import (
    build_run_metadata,
    load_scenario,
    replay_from_manifest,
    save_replay_manifest,
)
from optimal_execution.schedules import generate_schedule
from optimal_execution.simulator import simulate_deterministic, simulate_monte_carlo

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "data" / "fixtures"


def test_fixed_seed_bit_exact_reproducibility() -> None:
    scenario = load_scenario(CONFIG_DIR / "default_scenario.json")
    sched = generate_schedule(
        scenario,
        ScheduleRequest(strategy=StrategyType.ALMGREN_CHRISS_ANALYTIC, risk_aversion_lambda=1e-6),
    )
    mc1, paths1 = simulate_monte_carlo(scenario, sched, num_paths=500, seed=2026)
    mc2, paths2 = simulate_monte_carlo(scenario, sched, num_paths=500, seed=2026)
    mc3, paths3 = simulate_monte_carlo(scenario, sched, num_paths=500, seed=9999)

    assert mc1 == mc2
    assert np.array_equal(paths1, paths2)
    assert not np.array_equal(paths1, paths3)


def test_stable_run_id_and_replay_manifest(tmp_path: Path) -> None:
    scenario = load_scenario(CONFIG_DIR / "default_scenario.json")
    requests = [
        ScheduleRequest(strategy=StrategyType.TWAP),
        ScheduleRequest(strategy=StrategyType.ALMGREN_CHRISS_ANALYTIC, risk_aversion_lambda=1e-6),
    ]
    extra = {
        "schedule_requests": [
            {
                "strategy": r.strategy.value,
                "risk_aversion_lambda": r.risk_aversion_lambda,
                "target_participation_rate": r.target_participation_rate,
                "enforce_capacity_in_optimizer": r.enforce_capacity_in_optimizer,
            }
            for r in requests
        ]
    }
    meta1 = build_run_metadata(scenario, extra_config=extra)
    meta2 = build_run_metadata(scenario, extra_config=extra)
    meta_diff_seed = build_run_metadata(replace(scenario, seed=scenario.seed + 1), extra_config=extra)

    assert meta1.run_id == meta2.run_id
    assert meta1.config_sha256 == meta2.config_sha256
    assert meta1.run_id != meta_diff_seed.run_id

    manifest_path = save_replay_manifest(tmp_path / "manifest.json", scenario, requests, meta1)
    rep_scenario, rep_dets, rep_mcs, rep_meta = replay_from_manifest(manifest_path)

    assert rep_scenario == scenario
    assert rep_meta.run_id == meta1.run_id
    assert len(rep_dets) == 2
    assert len(rep_mcs) == 2

    # Direct simulation matches replayed results bit-for-bit
    for req, det, mc in zip(requests, rep_dets, rep_mcs):
        sched = generate_schedule(scenario, req)
        assert det == simulate_deterministic(scenario, sched)
        direct_mc, _ = simulate_monte_carlo(scenario, sched)
        assert mc == direct_mc


def test_cli_demo_validate_compare_replay_and_error_exit_codes(tmp_path: Path) -> None:
    scenario_path = str(CONFIG_DIR / "default_scenario.json")
    out_dir = tmp_path / "out"

    # Valid fixture validation
    assert cli_main(["validate", "--scenario", scenario_path]) == 0
    # Invalid fixture returns non-zero (no silent success)
    assert cli_main(["validate", "--scenario", str(FIXTURES_DIR / "invalid_negative_quantity.json")]) == 1
    # Missing file returns non-zero (no silent success)
    assert cli_main(["validate", "--scenario", str(tmp_path / "missing.json")]) == 1

    # Demo generates artifacts + manifest
    assert cli_main(["demo", "--scenario", scenario_path, "--output-dir", str(out_dir)]) == 0
    assert (out_dir / "baseline_demo_results.json").exists()
    assert (out_dir / "baseline_demo_summary.csv").exists()
    assert (out_dir / "baseline_replay_manifest.json").exists()

    # Replay succeeds
    assert cli_main(["replay", "--manifest", str(out_dir / "baseline_replay_manifest.json")]) == 0

    # Analytic vs numeric comparison succeeds
    cmp_out = out_dir / "cmp.json"
    assert cli_main(["compare", "--scenario", scenario_path, "--output", str(cmp_out)]) == 0
    assert cmp_out.exists()
