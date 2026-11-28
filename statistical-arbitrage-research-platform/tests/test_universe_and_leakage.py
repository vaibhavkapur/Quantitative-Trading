"""Integration tests for point-in-time universe, zero train/test leakage, and corporate actions."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from statarb.backtest import generate_walk_forward_folds, run_walk_forward_backtest
from statarb.cointegration import select_spreads_on_training_window
from statarb.config import PlatformConfig, SelectionConfig, UniverseConfig
from statarb.contracts import validate_prices
from statarb.exceptions import LeakageViolationError
from statarb.fixtures import generate_synthetic_universe_fixture
from statarb.universe import filter_universe


def _make_cfg() -> PlatformConfig:
    return PlatformConfig.model_validate(
        {
            "run_name": "leakage_and_ca_test",
            "seed": 42,
            "data": {
                "prices_path": "synthetic_universe_prices.parquet",
                "borrow_path": "synthetic_universe_borrow.parquet",
                "use_log_prices": True,
            },
            "universe": {
                "min_price": 5.0,
                "min_adv_usd": 1_000_000.0,
                "min_history_days": 120,
                "require_active": True,
                "same_sector_only": True,
            },
            "selection": {
                "max_adf_pvalue": 0.05,
                "fdr_alpha": 0.10,
                "multiple_testing_method": "bh",
                "min_half_life_days": 2.0,
                "max_half_life_days": 40.0,
                "min_correlation": 0.50,
                "max_pairs_per_fold": 4,
                "include_baskets": True,
                "basket_size": 3,
                "max_baskets_per_fold": 1,
            },
            "walk_forward": {
                "train_window_days": 180,
                "test_window_days": 60,
                "step_days": 60,
                "embargo_days": 1,
                "window_type": "rolling",
            },
        }
    )


def test_point_in_time_universe_filters_penny_and_delisted() -> None:
    prices, _, _ = generate_synthetic_universe_fixture(seed=42, n_days=400)
    dates = prices["date"].unique().sort().to_list()
    uni_cfg = UniverseConfig(
        min_price=5.0, min_adv_usd=1_000_000.0, min_history_days=120, require_active=True
    )

    # Fold 1: before delisting day (220) -> TECH_DELIST eligible; IND_PENNY excluded
    res_early = filter_universe(prices, dates[0], dates[180], uni_cfg)
    assert "TECH_DELIST" in res_early.eligible_symbols
    assert "IND_PENNY" not in res_early.eligible_symbols
    assert "IND_PENNY" in res_early.excluded_reasons

    # Fold 2: after delisting day (day 220) -> TECH_DELIST excluded point-in-time
    res_late = filter_universe(prices, dates[100], dates[280], uni_cfg)
    assert "TECH_DELIST" not in res_late.eligible_symbols
    assert "TECH_DELIST" in res_late.excluded_reasons


def test_explicit_leakage_guards_raise() -> None:
    prices, _, _ = generate_synthetic_universe_fixture(seed=42, n_days=220)
    dates = prices["date"].unique().sort().to_list()

    with pytest.raises(LeakageViolationError):
        filter_universe(
            prices,
            formation_start=dates[0],
            formation_end=dates[150],
            cfg=UniverseConfig(),
            max_allowed_date=dates[149],
        )

    with pytest.raises(LeakageViolationError):
        select_spreads_on_training_window(
            prices,
            eligible_symbols=("TECH_A", "TECH_B"),
            sector_map={"TECH_A": "TECH", "TECH_B": "TECH"},
            train_start=dates[0],
            train_end=dates[150],
            cfg=SelectionConfig(),
            max_allowed_date=dates[149],
        )


def test_no_test_window_pair_selection_invariance() -> None:
    """Acceptance criterion: scrambling ALL test-window prices cannot change training selection."""
    prices, borrow, _ = generate_synthetic_universe_fixture(seed=42, n_days=320)
    cfg = _make_cfg()
    dates = prices["date"].unique().sort().to_list()
    folds = generate_walk_forward_folds(dates, cfg.walk_forward)
    first_fold = folds[0]
    assert first_fold.train_end < first_fold.test_start

    # Run backtest on original data
    res_orig = run_walk_forward_backtest(prices, borrow, cfg)

    # Severely distort ALL prices strictly after fold 1's train_end (during test window!)
    mutated_prices = prices.with_columns(
        pl.when(pl.col("date") > first_fold.train_end)
        .then(
            pl.when(pl.col("symbol") == "TECH_A")
            .then(pl.col("close_adj") * 3.5)
            .when(pl.col("symbol") == "IND_RW1")
            .then(pl.col("close_adj") * 0.25)
            .otherwise(pl.col("close_adj") * 1.3)
        )
        .otherwise(pl.col("close_adj"))
        .alias("close_adj")
    )
    res_mut = run_walk_forward_backtest(mutated_prices, borrow, cfg)

    # Fold 1 universe and selected models MUST be bit-for-bit identical
    f1_orig = res_orig.folds[0]
    f1_mut = res_mut.folds[0]
    assert f1_orig.universe.eligible_symbols == f1_mut.universe.eligible_symbols
    assert [m.to_dict() for m in f1_orig.selection.selected_models] == [
        m.to_dict() for m in f1_mut.selection.selected_models
    ]


def test_corporate_action_split_invariance_end_to_end() -> None:
    """Injecting a properly backward-adjusted 4-for-1 stock split leaves OOS backtest invariant."""
    prices, borrow, _ = generate_synthetic_universe_fixture(seed=42, n_days=420)
    cfg = _make_cfg()

    res_base = run_walk_forward_backtest(prices, borrow, cfg)

    # Inject an additional 4-for-1 stock split on TECH_B mid-test-window (day 270):
    # pre-split raw prices multiply by 4x, split_factor on day 270 becomes 4.0,
    # close_adj remains unchanged (properly backward-adjusted total-return series).
    dates = prices["date"].unique().sort().to_list()
    split_date: date = dates[270]
    split_prices = prices.with_columns(
        pl.when((pl.col("symbol") == "TECH_B") & (pl.col("date") < split_date))
        .then(pl.col("close_raw") * 4.0)
        .otherwise(pl.col("close_raw"))
        .alias("close_raw"),
        pl.when((pl.col("symbol") == "TECH_B") & (pl.col("date") == split_date))
        .then(4.0)
        .otherwise(pl.col("split_factor"))
        .alias("split_factor"),
    )

    # Must still pass corporate action consistency validation
    validated_split, _, ca_events = validate_prices(split_prices, check_corporate_actions=True)
    assert ca_events >= 2

    res_split = run_walk_forward_backtest(validated_split, borrow, cfg)

    # Equity curve, returns, and trades must match exactly
    assert len(res_base.daily_equity) == len(res_split.daily_equity)
    for eq_a, eq_b in zip(res_base.daily_equity, res_split.daily_equity, strict=True):
        assert eq_a.equity == pytest.approx(eq_b.equity, rel=1e-10, abs=1e-8)
        assert eq_a.daily_return_net == pytest.approx(eq_b.daily_return_net, abs=1e-12)
    assert len(res_base.trades) == len(res_split.trades)
    assert res_split.reconciliation.passed is True
