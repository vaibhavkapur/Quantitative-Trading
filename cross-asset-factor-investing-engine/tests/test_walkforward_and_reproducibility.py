"""End-to-end walk-forward evaluation, class attribution, constraint audit, and reproducibility tests."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path

from cross_asset_factor_engine import (
    DiagnosticTracker,
    EngineConfig,
    MarketDataset,
    make_run_id,
    run_walk_forward_evaluation,
    save_evaluation_report,
    verify_replay_equivalence,
)


def test_walk_forward_evaluation_end_to_end_and_reproducible_replay(
    multi_year_dataset: MarketDataset,
    default_config: EngineConfig,
    tmp_path: Path,
) -> None:
    res_a = run_walk_forward_evaluation(
        multi_year_dataset, default_config, diagnostics=DiagnosticTracker(deterministic_clock=True)
    )
    res_b = run_walk_forward_evaluation(
        multi_year_dataset, default_config, diagnostics=DiagnosticTracker(deterministic_clock=True)
    )

    # 1. Exact deterministic replay equivalence
    replay = verify_replay_equivalence(res_a, res_b, atol=1e-12)
    assert replay["equivalent"] is True
    assert replay["max_return_diff"] == 0.0
    assert replay["max_weight_diff"] == 0.0
    assert make_run_id(res_a) == make_run_id(res_b)

    # 2. Walk-forward folds respect strict embargo and non-overlapping OOS test windows
    assert len(res_a.folds) >= 4
    prev_test_end = None
    for fold in res_a.folds:
        train_end = date.fromisoformat(fold["train_end"])
        test_start = date.fromisoformat(fold["test_start"])
        test_end = date.fromisoformat(fold["test_end"])
        assert train_end < test_start
        assert test_start <= test_end
        if prev_test_end is not None:
            assert prev_test_end < test_start
        prev_test_end = test_end

    # 3. Every signal has verified as-of provenance with zero look-ahead violations
    assert res_a.provenance_audit["all_have_as_of_provenance"] is True
    assert res_a.provenance_audit["lookahead_violations"] == 0

    # 4. All single factors, blended portfolio, and benchmarks are evaluated with net OOS + by-class turnover
    expected_strategies = {
        "momentum",
        "carry",
        "defensive",
        "blended",
        "benchmark_equal_weight",
        "benchmark_inv_vol",
    }
    assert set(res_a.strategy_metrics.keys()) == expected_strategies
    for strat in expected_strategies:
        m = res_a.strategy_metrics[strat]
        assert m["n_oos_days"] >= 500
        assert m["total_cost_drag_ann"] >= 0.0
        assert m["annualized_turnover"] >= 0.0
        # Class attribution covers all 3 asset classes
        assert set(res_a.class_attribution[strat].keys()) == {"equity", "fx", "commodity"}
        for cls_name in ("equity", "fx", "commodity"):
            cls_m = res_a.class_attribution[strat][cls_name]
            assert "net_ann_return_contribution" in cls_m
            assert "annualized_turnover" in cls_m
            assert cls_m["annualized_turnover"] >= 0.0

    # 5. All constrained portfolios respect EVERY cap across all rebalances with zero violations
    tol = default_config.portfolio.constraint_tolerance
    cons = default_config.portfolio.constraints
    for strat in ("momentum", "carry", "defensive", "blended"):
        audit = res_a.constraint_audit[strat]
        assert audit["violations"] == 0
        assert audit["n_rebalances"] > 0
        assert audit["max_observed_asset_weight"] <= cons.max_asset_weight + tol
        assert audit["max_observed_gross_exposure"] <= cons.max_gross_exposure + tol
        assert audit["min_observed_net_exposure"] >= cons.min_net_exposure - tol
        assert audit["max_observed_net_exposure"] <= cons.max_net_exposure + tol
        assert audit["max_observed_ex_ante_vol"] <= default_config.portfolio.volatility_cap + tol
        assert audit["max_observed_turnover"] <= cons.max_turnover_per_rebalance + tol
        for cls_name, cap in cons.max_class_gross_exposure.items():
            assert audit["max_observed_class_gross"][cls_name] <= cap + tol

    # 6. Machine-readable versioned report artifacts write cleanly
    run_dir = save_evaluation_report(res_a, tmp_path, deterministic_timestamp=True)
    for fname in (
        "summary.json",
        "strategy_comparison.csv",
        "class_attribution.csv",
        "daily_returns.csv",
        "portfolio_weights.csv",
        "signal_provenance_audit.csv",
        "diagnostics.jsonl",
    ):
        assert (run_dir / fname).exists()

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["run_id"] == make_run_id(res_a)
    assert summary["runtime_environment"]["polars_version"]
    assert summary["data_provenance"] == "synthetic_seeded_dgp"
