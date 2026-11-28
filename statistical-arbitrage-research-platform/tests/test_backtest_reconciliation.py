"""Tests for trade-to-position reconciliation, transaction/borrow costs, and borrow outages."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import polars as pl
import pytest

from statarb.backtest import (
    run_walk_forward_backtest,
    verify_trade_position_reconciliation,
)
from statarb.config import PlatformConfig
from statarb.evaluation import compute_performance_summary
from statarb.exceptions import BorrowUnavailableError, ReconciliationError
from statarb.fixtures import (
    generate_reference_pair_fixture,
    generate_synthetic_universe_fixture,
)


def _ref_config(*, strict_borrow: bool = False) -> PlatformConfig:
    return PlatformConfig.model_validate(
        {
            "run_name": "recon_test",
            "seed": 7,
            "data": {
                "prices_path": "ref.csv",
                "borrow_path": "ref_borrow.csv",
                "use_log_prices": False,
            },
            "universe": {
                "min_price": 1.0,
                "min_adv_usd": 100_000.0,
                "min_history_days": 60,
                "require_active": True,
                "same_sector_only": False,
            },
            "selection": {
                "max_adf_pvalue": 0.05,
                "multiple_testing_method": "none",
                "min_half_life_days": 1.0,
                "max_half_life_days": 50.0,
                "min_correlation": 0.30,
                "max_pairs_per_fold": 2,
                "include_baskets": False,
                "max_baskets_per_fold": 0,
            },
            "signals": {
                "z_entry": 1.25,
                "z_exit": 0.25,
                "z_stop": 4.0,
                "max_holding_days": 25,
            },
            "execution": {
                "initial_capital": 100_000.0,
                "gross_leverage_target": 1.0,
                "max_pair_weight": 0.50,
                "commission_bps": 2.0,
                "slippage_bps": 3.0,
                "default_borrow_fee_bps_annual": 50.0,
                "enforce_borrow_availability": True,
                "strict_borrow_errors": strict_borrow,
                "recall_on_borrow_loss": True,
                "execution_delay_days": 1,
            },
            "walk_forward": {
                "train_window_days": 100,
                "test_window_days": 50,
                "step_days": 50,
                "embargo_days": 1,
            },
        }
    )


def test_all_trades_reconcile_and_tampering_raises() -> None:
    bundle = generate_reference_pair_fixture(seed=7, n_days=260)
    cfg = _ref_config()

    res = run_walk_forward_backtest(bundle.prices, bundle.borrow, cfg)
    assert res.reconciliation.passed is True
    assert res.reconciliation.total_trades > 0
    assert res.reconciliation.max_position_diff < 1e-9
    assert res.reconciliation.max_equity_diff < 1e-4

    # Tampering with a single trade quantity must trigger ReconciliationError
    tampered_trades = list(res.trades)
    tampered_trades[0] = replace(tampered_trades[0], quantity=tampered_trades[0].quantity + 1.0)
    with pytest.raises(ReconciliationError):
        verify_trade_position_reconciliation(
            trades=tampered_trades,
            positions=list(res.positions),
            daily_equity=list(res.daily_equity),
            initial_capital=cfg.execution.initial_capital,
        )


def test_costs_and_borrow_accrue_and_drag_net_below_gross() -> None:
    bundle = generate_reference_pair_fixture(seed=7, n_days=260)
    cfg = _ref_config()

    res_net = run_walk_forward_backtest(bundle.prices, bundle.borrow, cfg, zero_cost=False)
    res_zero = run_walk_forward_backtest(bundle.prices, bundle.borrow, cfg, zero_cost=True)

    perf_net = compute_performance_summary(res_net, cfg)
    perf_zero = compute_performance_summary(res_zero, cfg)

    assert perf_net.total_commission > 0.0
    assert perf_net.total_slippage > 0.0
    assert perf_net.total_borrow_cost > 0.0
    assert perf_net.total_return_net < perf_net.total_return_gross
    assert perf_net.total_return_net < perf_zero.total_return_net


def test_unavailable_borrow_blocks_entries_and_strict_mode_raises() -> None:
    """Boundary scenario: unavailable borrow blocks all short trades or raises in strict mode."""
    bundle = generate_reference_pair_fixture(seed=7, n_days=260)
    no_borrow = bundle.borrow.with_columns(
        pl.lit(False).alias("borrow_available"),
        pl.lit(0.0).alias("max_short_notional"),
    )

    # Graceful mode: zero shorts held, explicit diagnostic counter incremented, clean reconciliation
    cfg_graceful = _ref_config(strict_borrow=False)
    res_graceful = run_walk_forward_backtest(bundle.prices, no_borrow, cfg_graceful)
    assert res_graceful.reconciliation.passed is True
    assert res_graceful.diagnostics["borrow_blocked_entries"] > 0
    assert all(snap.quantity >= -1e-10 for snap in res_graceful.positions)

    # Strict mode: raises explicit BorrowUnavailableError
    cfg_strict = _ref_config(strict_borrow=True)
    with pytest.raises(BorrowUnavailableError, match="Borrow unavailable"):
        run_walk_forward_backtest(bundle.prices, no_borrow, cfg_strict)


def test_synthetic_universe_borrow_outage_enforces_no_shorts_during_outage() -> None:
    """During TECH_D borrow outage (days 380..395), zero short positions in TECH_D are held."""
    prices, borrow, meta = generate_synthetic_universe_fixture(seed=42, n_days=600)
    outage_start = date.fromisoformat(meta["planted_structures"]["borrow_outage"]["start_date"])
    outage_end = date.fromisoformat(meta["planted_structures"]["borrow_outage"]["end_date"])

    cfg = PlatformConfig.model_validate(
        {
            "run_name": "outage_check",
            "seed": 42,
            "data": {"prices_path": "p.parquet", "borrow_path": "b.parquet"},
            "signals": {"z_entry": 1.5, "z_exit": 0.25, "z_stop": 3.5},
        }
    )
    res = run_walk_forward_backtest(prices, borrow, cfg)
    assert res.reconciliation.passed is True
    # Final day must have zero orphaned positions after fold-end flatten / delisting liquidation
    last_date = res.daily_equity[-1].date
    assert all(snap.date != last_date for snap in res.positions)

    for snap in res.positions:
        if snap.symbol == "TECH_D" and outage_start <= snap.date <= outage_end:
            assert snap.quantity >= -1e-10, f"Unexpected short TECH_D on outage date {snap.date}"
