"""End-to-end reproducible demo runner for the Cross-Asset Factor Investing Engine."""

from __future__ import annotations

from pathlib import Path

from cross_asset_factor_engine import (
    DiagnosticTracker,
    load_config,
    load_dataset_from_dir,
    make_run_id,
    run_walk_forward_evaluation,
    save_evaluation_report,
    verify_replay_equivalence,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg = load_config(PROJECT_ROOT / "config" / "default.yaml")
    dataset_dir = PROJECT_ROOT / cfg.data.dataset_dir
    dataset = load_dataset_from_dir(dataset_dir, base_currency=cfg.base_currency)

    diag = DiagnosticTracker(deterministic_clock=True)
    result = run_walk_forward_evaluation(dataset, cfg, diagnostics=diag)
    run_dir = save_evaluation_report(result, PROJECT_ROOT / cfg.reporting.output_dir)

    # Verify deterministic replay
    replay_result = run_walk_forward_evaluation(
        dataset, cfg, diagnostics=DiagnosticTracker(deterministic_clock=True)
    )
    replay_check = verify_replay_equivalence(result, replay_result)
    assert replay_check["equivalent"], f"Replay check failed: {replay_check}"

    # Also run and archive the small analytical reference fixture baseline
    ref_cfg = load_config(PROJECT_ROOT / "config" / "reference_fixture.yaml")
    ref_dataset = load_dataset_from_dir(
        PROJECT_ROOT / ref_cfg.data.dataset_dir, base_currency=ref_cfg.base_currency
    )
    ref_result = run_walk_forward_evaluation(
        ref_dataset, ref_cfg, diagnostics=DiagnosticTracker(deterministic_clock=True)
    )
    ref_dir = save_evaluation_report(ref_result, PROJECT_ROOT / ref_cfg.reporting.output_dir)

    print(f"Run ID (multi-year synthetic): {make_run_id(result)}")
    print(f"Report directory:              {run_dir}")
    print(f"Reference fixture run dir:     {ref_dir}")
    print(f"Replay equivalence verified:   {replay_check['equivalent']} (max_ret_diff={replay_check['max_return_diff']})")
    print("\nNet OOS Strategy Comparison (Synthetic Multi-Year Fixture):")
    header = f"{'Strategy':<24} {'NetRet':>8} {'Vol':>8} {'NetSR':>7} {'MaxDD':>8} {'AnnTO':>7} {'CostDrag':>9}"
    print(header)
    print("-" * len(header))
    for strat, m in result.strategy_metrics.items():
        print(
            f"{strat:<24} {m['net_ann_return']:>7.2%} {m['ann_volatility']:>7.2%} "
            f"{m['net_sharpe']:>7.2f} {m['max_drawdown']:>7.2%} "
            f"{m['annualized_turnover']:>7.2f} {m['total_cost_drag_ann']:>8.2%}"
        )


if __name__ == "__main__":
    main()
