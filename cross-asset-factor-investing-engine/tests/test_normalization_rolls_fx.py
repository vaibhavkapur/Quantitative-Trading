"""Deterministic unit tests for futures rolls, FX conversions, and reference oracle agreement."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import polars as pl
import pytest

from cross_asset_factor_engine import (
    DiagnosticTracker,
    EngineConfig,
    MarketDataset,
    compute_raw_factor_signals,
    normalize_market_data,
    validate_dataset,
)


def test_normalization_matches_hand_calculated_oracle(
    small_dataset: MarketDataset,
    small_config: EngineConfig,
    small_oracle: dict[str, Any],
) -> None:
    tol = float(small_oracle["tolerance"])
    diag = DiagnosticTracker(deterministic_clock=True)
    norm = normalize_market_data(small_dataset, annualization_days=252, diagnostics=diag)
    raw = compute_raw_factor_signals(norm, small_dataset, small_config, diagnostics=diag)

    # 1. EUR-quoted equity futures FX conversion (EQ_STOXX day 1)
    c1 = small_oracle["checks"]["eq_stoxx_day1_fx_conversion"]
    row1 = norm.filter(
        (pl.col("asset_id") == c1["asset_id"]) & (pl.col("date") == date.fromisoformat(c1["date"]))
    ).to_dicts()[0]
    assert row1["local_return"] == pytest.approx(c1["local_return"], abs=tol)
    assert row1["fx_return_factor"] == pytest.approx(c1["fx_return_factor"], abs=tol)
    assert row1["gross_excess_return_base"] == pytest.approx(c1["gross_excess_return_base"], abs=tol)
    assert row1["financing_cost_drag"] == pytest.approx(c1["financing_cost_drag"], abs=tol)
    assert row1["net_return_before_tc"] == pytest.approx(c1["net_return_before_tc"], abs=tol)

    # 2. Commodity futures roll day (CMD_CL day 5) + roll cost + carry
    c2 = small_oracle["checks"]["cmd_cl_day5_roll_and_carry"]
    row2 = norm.filter(
        (pl.col("asset_id") == c2["asset_id"]) & (pl.col("date") == date.fromisoformat(c2["date"]))
    ).to_dicts()[0]
    assert row2["is_roll_date"] is True
    assert row2["local_return"] == pytest.approx(c2["local_return"], abs=tol)
    assert row2["roll_cost_drag"] == pytest.approx(c2["roll_cost_drag"], abs=tol)
    assert row2["financing_cost_drag"] == pytest.approx(c2["financing_cost_drag"], abs=tol)
    assert row2["net_return_before_tc"] == pytest.approx(c2["net_return_before_tc"], abs=tol)
    assert row2["implied_roll_yield"] == pytest.approx(c2["implied_roll_yield"], abs=tol)

    sig2 = raw.filter(
        (pl.col("asset_id") == c2["asset_id"])
        & (pl.col("signal_date") == date.fromisoformat(c2["date"]))
        & (pl.col("factor_name") == "carry")
    ).to_dicts()[0]
    assert sig2["raw_value"] == pytest.approx(c2["carry_raw"], abs=tol)

    # 3. Post-roll continuity (CMD_CL day 6: exact +1.0% return despite $4 contract splice jump!)
    c3 = small_oracle["checks"]["cmd_cl_day6_post_roll_continuity"]
    row3 = norm.filter(
        (pl.col("asset_id") == c3["asset_id"]) & (pl.col("date") == date.fromisoformat(c3["date"]))
    ).to_dicts()[0]
    assert row3["local_return"] == pytest.approx(c3["local_return"], abs=tol)
    assert row3["roll_cost_drag"] == pytest.approx(0.0, abs=tol)
    assert row3["net_return_before_tc"] == pytest.approx(c3["net_return_before_tc"], abs=tol)

    # 4. FX forward spot + lagged interest-rate differential excess return & carry
    c4 = small_oracle["checks"]["fx_eurusd_day1_excess_return_and_carry"]
    row4 = norm.filter(
        (pl.col("asset_id") == c4["asset_id"]) & (pl.col("date") == date.fromisoformat(c4["date"]))
    ).to_dicts()[0]
    assert row4["local_return"] == pytest.approx(c4["local_return"], abs=tol)
    assert row4["carry_return_component"] == pytest.approx(c4["carry_return_component"], abs=tol)
    assert row4["gross_excess_return_base"] == pytest.approx(c4["gross_excess_return_base"], abs=tol)
    assert row4["net_return_before_tc"] == pytest.approx(c4["net_return_before_tc"], abs=tol)

    sig4 = raw.filter(
        (pl.col("asset_id") == c4["asset_id"])
        & (pl.col("signal_date") == date.fromisoformat(c4["date"]))
        & (pl.col("factor_name") == "carry")
    ).to_dicts()[0]
    assert sig4["raw_value"] == pytest.approx(c4["carry_raw"], abs=tol)

    assert diag.get_counter("roll_events_processed") == 1
    assert diag.get_counter("fx_conversions_applied") == 12


def test_reverse_quoted_usdjpy_fx_conversion_identity() -> None:
    """Verify exact USD P&L identity for USDJPY (base=USD, quote=JPY): (S_t - S_{t-1})/S_t + carry."""
    d0, d1 = date(2024, 1, 2), date(2024, 1, 3)
    ts0 = datetime(2024, 1, 2, 20, 0, tzinfo=timezone.utc)
    ts1 = datetime(2024, 1, 3, 20, 0, tzinfo=timezone.utc)

    meta = pl.DataFrame(
        [
            {
                "asset_id": "FX_USDJPY",
                "asset_class": "fx",
                "quote_currency": "JPY",
                "base_currency": "USD",
                "instrument_type": "fx_forward",
                "carry_convention": "interest_rate_differential",
                "contract_multiplier": 100000.0,
                "tick_size": 0.01,
                "tc_bps": 1.0,
                "financing_spread_bps": 12.6,
                "roll_cost_bps": 0.0,
                "active_from": date(2020, 1, 1),
                "active_to": None,
            }
        ],
        schema_overrides={"active_to": pl.Date},
    )
    # USDJPY moves from 100.0 to 105.0 (USD strengthens by 5% vs JPY -> USD P&L = (105-100)/105)
    prices = pl.DataFrame(
        [
            {
                "date": d0,
                "asset_id": "FX_USDJPY",
                "contract_code": "USDJPY_1M",
                "front_price": 100.0,
                "next_price": 99.6,
                "front_expiry_days": 30,
                "next_expiry_days": 60,
                "is_roll_date": False,
                "prev_front_price_new_contract": None,
                "as_of_ts": ts0,
                "publication_ts": ts0,
            },
            {
                "date": d1,
                "asset_id": "FX_USDJPY",
                "contract_code": "USDJPY_1M",
                "front_price": 105.0,
                "next_price": 104.58,
                "front_expiry_days": 30,
                "next_expiry_days": 60,
                "is_roll_date": False,
                "prev_front_price_new_contract": None,
                "as_of_ts": ts1,
                "publication_ts": ts1,
            },
        ],
        schema_overrides={"prev_front_price_new_contract": pl.Float64},
    )
    fx = pl.DataFrame(
        [
            {"date": d0, "currency": "USD", "usd_per_unit": 1.0, "short_rate": 0.05, "as_of_ts": ts0, "publication_ts": ts0},
            {"date": d1, "currency": "USD", "usd_per_unit": 1.0, "short_rate": 0.05, "as_of_ts": ts1, "publication_ts": ts1},
            {"date": d0, "currency": "JPY", "usd_per_unit": 1.0 / 100.0, "short_rate": 0.00, "as_of_ts": ts0, "publication_ts": ts0},
            {"date": d1, "currency": "JPY", "usd_per_unit": 1.0 / 105.0, "short_rate": 0.00, "as_of_ts": ts1, "publication_ts": ts1},
        ]
    )
    fi = pl.DataFrame(
        schema={
            "date": pl.Date,
            "asset_id": pl.String,
            "dividend_yield": pl.Float64,
            "earnings_quality": pl.Float64,
            "foreign_short_rate": pl.Float64,
            "domestic_short_rate": pl.Float64,
            "convenience_yield_proxy": pl.Float64,
            "as_of_ts": pl.Datetime("us", "UTC"),
            "publication_ts": pl.Datetime("us", "UTC"),
        }
    )

    ds = validate_dataset(meta, prices, fx, fi, base_currency="USD")
    norm = normalize_market_data(ds, annualization_days=252)
    r1 = norm.filter(pl.col("date") == d1).to_dicts()[0]

    expected_spot_usd = (105.0 - 100.0) / 105.0
    expected_carry = (0.05 - 0.00) / 252.0
    expected_fin = (12.6 / 10_000.0) / 252.0

    assert r1["gross_excess_return_base"] == pytest.approx(expected_spot_usd + expected_carry, abs=1e-12)
    assert r1["net_return_before_tc"] == pytest.approx(
        expected_spot_usd + expected_carry - expected_fin, abs=1e-12
    )
