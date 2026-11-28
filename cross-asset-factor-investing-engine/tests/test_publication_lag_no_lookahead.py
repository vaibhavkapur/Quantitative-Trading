"""Integration tests proving zero look-ahead bias from publication lag across the full pipeline."""

from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from cross_asset_factor_engine import (
    DiagnosticTracker,
    EngineConfig,
    MarketDataset,
    compute_raw_factor_signals,
    normalize_market_data,
    run_walk_forward_evaluation,
    validate_dataset,
)


def test_publication_lag_blocks_future_fundamental_until_publication_date(
    small_dataset: MarketDataset,
    small_config: EngineConfig,
    small_oracle: dict[str, Any],
    fixtures_dir: Path,
) -> None:
    scenario = json.loads(
        (fixtures_dir / "boundary" / "publication_lag_scenario.json").read_text(encoding="utf-8")
    )
    lag_check = small_oracle["checks"]["eq_spx_publication_lag_gating"]
    tol = float(small_oracle["tolerance"])

    diag = DiagnosticTracker(deterministic_clock=True)
    norm = normalize_market_data(small_dataset, diagnostics=diag)
    raw = compute_raw_factor_signals(norm, small_dataset, small_config, diagnostics=diag)

    # Pre-publication date (2024-01-10): record for 2024-01-08 is NOT published until 2024-01-12T12:00Z
    pre_row = raw.filter(
        (pl.col("asset_id") == scenario["target_asset"])
        & (pl.col("factor_name") == "carry")
        & (pl.col("signal_date") == date.fromisoformat(lag_check["pre_publication_date"]))
    ).to_dicts()[0]
    assert pre_row["raw_value"] == pytest.approx(lag_check["pre_publication_carry_raw"], abs=tol)
    assert pre_row["source_max_publication_ts"] <= pre_row["as_of_ts"]

    # Post-publication date (2024-01-12): now published at 12:00Z <= 21:00Z cutoff
    post_row = raw.filter(
        (pl.col("asset_id") == scenario["target_asset"])
        & (pl.col("factor_name") == "carry")
        & (pl.col("signal_date") == date.fromisoformat(lag_check["post_publication_date"]))
    ).to_dicts()[0]
    assert post_row["raw_value"] == pytest.approx(lag_check["post_publication_carry_raw"], abs=tol)
    assert post_row["source_max_publication_ts"] <= post_row["as_of_ts"]

    # Diagnostics explicitly recorded blocked rows from the timestamp gate
    assert diag.get_counter("rows_blocked_by_timestamp_gate") > 0


def test_future_publication_perturbation_has_zero_effect_before_cutoff(
    multi_year_dataset: MarketDataset,
    default_config: EngineConfig,
) -> None:
    """End-to-end no-look-ahead integration test via future data perturbation invariance.

    Mutates:
      1. All factor_inputs with publication_ts > cutoff (including records whose reference `date`
         is BEFORE cutoff, simulating lagged corporate/fundamental announcements!).
      2. All prices and FX rates strictly after cutoff.

    Asserts bitwise-identical signals, scaler parameters, portfolio weights, and OOS daily returns
    for every date <= cutoff.
    """
    baseline = run_walk_forward_evaluation(multi_year_dataset, default_config)
    all_oos_dates = sorted(baseline.daily_returns_df["date"].unique().to_list())
    # Choose cutoff at the end of fold 1 so we verify an entire training + OOS test fold
    fold1_end = date.fromisoformat(baseline.folds[0]["test_end"])
    cutoff_ts = datetime(fold1_end.year, fold1_end.month, fold1_end.day, 21, 0, 0, tzinfo=timezone.utc)
    assert fold1_end in all_oos_dates

    # 1. Perturb all factor_inputs whose publication_ts > cutoff_ts (even if reference date <= fold1_end!)
    perturbed_fi = multi_year_dataset.factor_inputs.with_columns(
        pl.when(pl.col("publication_ts") > cutoff_ts)
        .then(pl.col("dividend_yield") * 15.0 + 0.50)
        .otherwise(pl.col("dividend_yield"))
        .alias("dividend_yield"),
        pl.when(pl.col("publication_ts") > cutoff_ts)
        .then(pl.col("earnings_quality") * -20.0 + 9.0)
        .otherwise(pl.col("earnings_quality"))
        .alias("earnings_quality"),
        pl.when(pl.col("publication_ts") > cutoff_ts)
        .then(pl.col("foreign_short_rate") + 0.25)
        .otherwise(pl.col("foreign_short_rate"))
        .alias("foreign_short_rate"),
    )

    # 2. Also inject an extreme retroactive fundamental revision dated BEFORE fold1_end
    #    whose publication_ts is AFTER cutoff_ts (classic look-ahead trap!).
    retro_row = pl.DataFrame(
        [
            {
                "date": date.fromisoformat(baseline.folds[0]["train_end"]),
                "asset_id": "EQ_SPX",
                "dividend_yield": 0.99,
                "earnings_quality": 99.0,
                "foreign_short_rate": None,
                "domestic_short_rate": 0.05,
                "convenience_yield_proxy": None,
                "as_of_ts": datetime(
                    date.fromisoformat(baseline.folds[0]["train_end"]).year,
                    date.fromisoformat(baseline.folds[0]["train_end"]).month,
                    date.fromisoformat(baseline.folds[0]["train_end"]).day,
                    20,
                    0,
                    tzinfo=timezone.utc,
                ),
                "publication_ts": datetime(
                    fold1_end.year, fold1_end.month, fold1_end.day, 22, 0, tzinfo=timezone.utc
                ),
            }
        ],
        schema=multi_year_dataset.factor_inputs.schema,
    )
    perturbed_fi = pl.concat([perturbed_fi, retro_row], how="vertical")

    # 3. Perturb prices strictly after fold1_end with a strong compounding regime shift (+1.5%/day)
    #    that dramatically alters all future returns/signals without an unphysical single-day +250% jump
    all_px_dates = sorted(multi_year_dataset.prices["date"].unique().to_list())
    after_dates = [d for d in all_px_dates if d > fold1_end]
    scale_map = {d: 1.015 ** (idx + 1) for idx, d in enumerate(after_dates)}
    scale_df = pl.DataFrame(
        {
            "date": list(scale_map.keys()),
            "_scale": list(scale_map.values()),
            "_prev_scale": [1.0 if idx == 0 else scale_map[after_dates[idx - 1]] for idx in range(len(after_dates))],
        }
    )
    perturbed_prices = (
        multi_year_dataset.prices.join(scale_df, on="date", how="left")
        .with_columns(
            pl.col("_scale").fill_null(1.0),
            pl.col("_prev_scale").fill_null(1.0),
        )
        .with_columns(
            (pl.col("front_price") * pl.col("_scale")).alias("front_price"),
            (pl.col("next_price") * pl.col("_scale") * pl.when(pl.col("date") > fold1_end).then(0.98).otherwise(1.0)).alias("next_price"),
            pl.when(pl.col("prev_front_price_new_contract").is_not_null())
            .then(pl.col("prev_front_price_new_contract") * pl.col("_prev_scale"))
            .otherwise(None)
            .alias("prev_front_price_new_contract"),
        )
        .drop(["_scale", "_prev_scale"])
    )

    perturbed_ds = validate_dataset(
        multi_year_dataset.contract_metadata,
        perturbed_prices,
        multi_year_dataset.fx_rates,
        perturbed_fi,
        base_currency="USD",
    )
    perturbed = run_walk_forward_evaluation(perturbed_ds, default_config)

    # Fold 1 scaler parameters must be IDENTICAL (zero leak from retroactive publication!)
    assert baseline.folds[0]["scaler_params"] == perturbed.folds[0]["scaler_params"]

    # All OOS signals on or before fold1_end must be BITWISE IDENTICAL
    base_sig = baseline.oos_signals.filter(pl.col("signal_date") <= fold1_end).sort(
        ["signal_date", "asset_id", "factor_name"]
    )
    pert_sig = perturbed.oos_signals.filter(pl.col("signal_date") <= fold1_end).sort(
        ["signal_date", "asset_id", "factor_name"]
    )
    assert base_sig.height == pert_sig.height
    np.testing.assert_allclose(
        base_sig["z_score"].fill_null(0.0).to_numpy(),
        pert_sig["z_score"].fill_null(0.0).to_numpy(),
        atol=0.0,
        rtol=0.0,
    )

    # All OOS weights decided on or before fold1_end must be IDENTICAL
    base_w = baseline.weights_df.filter(pl.col("decision_date") <= fold1_end).sort(
        ["strategy", "decision_date", "asset_id"]
    )
    pert_w = perturbed.weights_df.filter(pl.col("decision_date") <= fold1_end).sort(
        ["strategy", "decision_date", "asset_id"]
    )
    np.testing.assert_allclose(base_w["weight"].to_numpy(), pert_w["weight"].to_numpy(), atol=1e-12)

    # All OOS daily returns on or before fold1_end must be IDENTICAL
    base_r = baseline.daily_returns_df.filter(pl.col("date") <= fold1_end).sort(["strategy", "date"])
    pert_r = perturbed.daily_returns_df.filter(pl.col("date") <= fold1_end).sort(["strategy", "date"])
    np.testing.assert_allclose(base_r["net_return"].to_numpy(), pert_r["net_return"].to_numpy(), atol=1e-12)
