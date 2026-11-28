"""End-to-end tests for evaluation metrics, multiple testing, benchmarks, and reproducibility."""

from __future__ import annotations

from pathlib import Path

from statarb.cli import main as cli_main
from statarb.fixtures import write_all_fixtures
from statarb.pipeline import execute_research_pipeline


def _write_test_project(tmp_path: Path) -> tuple[Path, Path]:
    fixtures_dir = tmp_path / "data" / "fixtures"
    write_all_fixtures(fixtures_dir)
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "test_demo.toml"
    cfg_path.write_text(
        """
schema_version = "1.0.0"
run_name = "repro_demo"
seed = 42

[data]
prices_path = "data/fixtures/synthetic_universe_prices.parquet"
borrow_path = "data/fixtures/synthetic_universe_borrow.parquet"
use_log_prices = true

[universe]
min_price = 5.0
min_adv_usd = 1000000.0
min_history_days = 120
require_active = true
same_sector_only = true

[selection]
max_adf_pvalue = 0.05
fdr_alpha = 0.10
multiple_testing_method = "bh"
min_half_life_days = 2.0
max_half_life_days = 40.0
min_correlation = 0.50
max_pairs_per_fold = 4
include_baskets = true
basket_size = 3
max_baskets_per_fold = 1

[signals]
z_entry = 1.75
z_exit = 0.25
z_stop = 3.50
max_holding_days = 30

[execution]
initial_capital = 1000000.0
commission_bps = 1.5
slippage_bps = 3.0
default_borrow_fee_bps_annual = 50.0
enforce_borrow_availability = true

[walk_forward]
train_window_days = 180
test_window_days = 60
step_days = 60
embargo_days = 1

[evaluation]
bootstrap_samples = 150
bootstrap_block_size = 10
confidence_level = 0.95
sweep_z_entries = [1.5, 1.75, 2.0]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return tmp_path, cfg_path


def test_end_to_end_reproducibility_and_required_report_fields(tmp_path: Path) -> None:
    root, cfg_path = _write_test_project(tmp_path)
    out1 = tmp_path / "reports_run1"
    out2 = tmp_path / "reports_run2"

    rep1 = execute_research_pipeline(cfg_path, project_root=root, output_dir=out1, quiet=True)
    rep2 = execute_research_pipeline(cfg_path, project_root=root, output_dir=out2, quiet=True)

    # Deterministic identity across reruns with same versioned inputs, config, and seed
    assert rep1["run_id"] == rep2["run_id"]
    assert rep1["config_hash"] == rep2["config_hash"]
    assert rep1["performance"] == rep2["performance"]
    assert rep1["uncertainty"] == rep2["uncertainty"]
    assert rep1["multiple_testing"] == rep2["multiple_testing"]
    assert rep1["stability"] == rep2["stability"]
    assert rep1["benchmarks"] == rep2["benchmarks"]
    assert rep1["reconciliation"] == rep2["reconciliation"]
    assert rep1["diagnostics"] == rep2["diagnostics"]
    assert rep1["folds"] == rep2["folds"]

    # Required acceptance evidence fields
    assert rep1["reconciliation"]["passed"] is True
    assert rep1["performance"]["n_oos_days"] > 0
    assert rep1["uncertainty"]["sharpe_ci_low"] <= rep1["uncertainty"]["sharpe_ci_high"]
    assert rep1["uncertainty"]["ann_return_ci_low"] <= rep1["uncertainty"]["ann_return_ci_high"]
    assert rep1["multiple_testing"]["total_pair_hypotheses_tested"] > 0
    assert len(rep1["multiple_testing"]["attempted_configurations"]) == 3
    assert 0.0 <= rep1["multiple_testing"]["deflated_sharpe"]["deflated_sharpe_prob"] <= 1.0
    assert "equal_weight_universe" in rep1["benchmarks"]
    assert "zero_cost_strategy" in rep1["benchmarks"]

    # Raw output files written
    assert (out1 / f"{rep1['run_id']}_summary.json").exists()
    assert (out1 / f"{rep1['run_id']}_equity_curve.csv").exists()


def test_cli_validate_and_run_smoke(tmp_path: Path) -> None:
    root, cfg_path = _write_test_project(tmp_path)
    out_dir = tmp_path / "cli_reports"

    rc_val = cli_main(["validate", "--config", str(cfg_path), "--project-root", str(root)])
    assert rc_val == 0

    rc_run = cli_main(
        [
            "run",
            "--config",
            str(cfg_path),
            "--project-root",
            str(root),
            "--output-dir",
            str(out_dir),
            "--no-sweep",
            "--quiet",
        ]
    )
    assert rc_run == 0
    assert (out_dir / "repro_demo_latest.json").exists()
