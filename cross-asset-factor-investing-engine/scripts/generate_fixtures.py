"""Deterministic fixture generator for valid, synthetic multi-year, invalid, and boundary scenarios.

All generated fixtures are synthetic research data (zero live/customer/proprietary records).
The small reference fixture also emits an independently hand-calculated analytical oracle
(`expected_reference.json`) computed using pure Python math without importing domain code.
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = PROJECT_ROOT / "data" / "fixtures"


def _business_days(start: date, n_days: int) -> list[date]:
    out: list[date] = []
    cur = start
    while len(out) < n_days:
        if cur.weekday() < 5:
            out.append(cur)
        cur += timedelta(days=1)
    return out


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})


def _ts(d: date, hhmmss: str = "20:00:00Z") -> str:
    return f"{d.isoformat()}T{hhmmss}"


def generate_valid_small_and_oracle() -> None:
    """Create a 12-day deterministic reference fixture with an independent pure-Python oracle."""
    out_dir = FIXTURES_DIR / "valid_small"
    out_dir.mkdir(parents=True, exist_ok=True)
    days = _business_days(date(2024, 1, 2), 12)

    metadata = [
        {
            "asset_id": "EQ_SPX",
            "asset_class": "equity",
            "quote_currency": "USD",
            "base_currency": None,
            "instrument_type": "equity_index_future",
            "carry_convention": "dividend_minus_financing",
            "contract_multiplier": 50.0,
            "tick_size": 0.25,
            "tc_bps": 2.0,
            "financing_spread_bps": 25.0,
            "roll_cost_bps": 1.5,
            "active_from": "2020-01-01",
            "active_to": None,
        },
        {
            "asset_id": "EQ_STOXX",
            "asset_class": "equity",
            "quote_currency": "EUR",
            "base_currency": None,
            "instrument_type": "equity_index_future",
            "carry_convention": "dividend_minus_financing",
            "contract_multiplier": 10.0,
            "tick_size": 1.0,
            "tc_bps": 2.5,
            "financing_spread_bps": 30.0,
            "roll_cost_bps": 2.0,
            "active_from": "2020-01-01",
            "active_to": None,
        },
        {
            "asset_id": "FX_EURUSD",
            "asset_class": "fx",
            "quote_currency": "USD",
            "base_currency": "EUR",
            "instrument_type": "fx_forward",
            "carry_convention": "interest_rate_differential",
            "contract_multiplier": 125000.0,
            "tick_size": 0.0001,
            "tc_bps": 1.0,
            "financing_spread_bps": 10.0,
            "roll_cost_bps": 0.5,
            "active_from": "2020-01-01",
            "active_to": None,
        },
        {
            "asset_id": "CMD_CL",
            "asset_class": "commodity",
            "quote_currency": "USD",
            "base_currency": None,
            "instrument_type": "commodity_future",
            "carry_convention": "futures_term_structure_roll",
            "contract_multiplier": 1000.0,
            "tick_size": 0.01,
            "tc_bps": 4.0,
            "financing_spread_bps": 20.0,
            "roll_cost_bps": 5.0,
            "active_from": "2020-01-01",
            "active_to": None,
        },
    ]

    # Exact deterministic FX rates (EUR moves cleanly so FX P&L conversion is verifiable by hand)
    eur_usd = [1.1000, 1.1055, 1.1000, 1.1110, 1.1165, 1.1110, 1.1220, 1.1275, 1.1220, 1.1330, 1.1385, 1.1440]
    usd_rate = 0.0500
    eur_rate = 0.0350

    fx_rows: list[dict[str, Any]] = []
    for i, d in enumerate(days):
        fx_rows.append(
            {
                "date": d.isoformat(),
                "currency": "USD",
                "usd_per_unit": 1.0,
                "short_rate": usd_rate,
                "as_of_ts": _ts(d, "19:00:00Z"),
                "publication_ts": _ts(d, "19:05:00Z"),
            }
        )
        fx_rows.append(
            {
                "date": d.isoformat(),
                "currency": "EUR",
                "usd_per_unit": eur_usd[i],
                "short_rate": eur_rate,
                "as_of_ts": _ts(d, "19:00:00Z"),
                "publication_ts": _ts(d, "19:05:00Z"),
            }
        )

    # Prices with an explicit commodity contract roll at day index 5 close (effective day index 6)
    spx_px = [100.0, 101.0, 102.01, 101.50, 102.515, 103.54, 104.0577, 105.0983, 104.5728, 105.6185, 106.6747, 107.7414]
    stoxx_px = [200.0, 202.0, 201.0, 203.01, 205.04, 204.015, 206.055, 208.115, 207.075, 209.145, 211.236, 213.348]
    # CMD_CL front contract CLG24 on days 0..5, rolls at day 5 close into CLH24 on days 6..11
    # Note deliberate $4.00 backwardation jump between CLG24 (80.0 -> 76.0 on CLH24 at day 5)
    cl_front = [80.0, 80.8, 81.6, 80.784, 81.592, 82.000, 78.780, 79.568, 79.170, 79.962, 80.761, 81.569]
    cl_next = [76.0, 76.76, 77.52, 76.745, 77.512, 78.000, 75.600, 76.356, 75.974, 76.734, 77.501, 78.276]
    # On day 6 (first day holding CLH24), prev_front_price_new_contract = CLH24's day 5 close = 78.000!
    # So day 6 return = 78.780 / 78.000 - 1 = +0.01 (+1.0%), NOT 78.780 / 82.000 - 1 = -3.93%!

    price_rows: list[dict[str, Any]] = []
    for i, d in enumerate(days):
        # EQ_SPX
        price_rows.append(
            {
                "date": d.isoformat(),
                "asset_id": "EQ_SPX",
                "contract_code": "ESH24",
                "front_price": spx_px[i],
                "next_price": round(spx_px[i] * 1.008, 6),
                "front_expiry_days": 75 - i,
                "next_expiry_days": 165 - i,
                "is_roll_date": "false",
                "prev_front_price_new_contract": None,
                "as_of_ts": _ts(d, "20:00:00Z"),
                "publication_ts": _ts(d, "20:05:00Z"),
            }
        )
        # EQ_STOXX (EUR-quoted)
        price_rows.append(
            {
                "date": d.isoformat(),
                "asset_id": "EQ_STOXX",
                "contract_code": "VGH24",
                "front_price": stoxx_px[i],
                "next_price": round(stoxx_px[i] * 1.004, 6),
                "front_expiry_days": 75 - i,
                "next_expiry_days": 165 - i,
                "is_roll_date": "false",
                "prev_front_price_new_contract": None,
                "as_of_ts": _ts(d, "19:30:00Z"),
                "publication_ts": _ts(d, "19:35:00Z"),
            }
        )
        # FX_EURUSD
        price_rows.append(
            {
                "date": d.isoformat(),
                "asset_id": "FX_EURUSD",
                "contract_code": "EURUSD_1M",
                "front_price": eur_usd[i],
                "next_price": round(eur_usd[i] * (1.0 + (usd_rate - eur_rate) * 30.0 / 365.0), 6),
                "front_expiry_days": 30,
                "next_expiry_days": 60,
                "is_roll_date": "false",
                "prev_front_price_new_contract": None,
                "as_of_ts": _ts(d, "20:00:00Z"),
                "publication_ts": _ts(d, "20:05:00Z"),
            }
        )
        # CMD_CL with explicit roll at day 5 close -> CLH24 starts day 6
        is_roll = i == 5
        contract_code = "CLG24" if i <= 5 else "CLH24"
        front_exp = (35 - i) if i <= 5 else (65 - i)
        next_exp = front_exp + 30
        prev_new = 78.000 if i == 6 else None
        price_rows.append(
            {
                "date": d.isoformat(),
                "asset_id": "CMD_CL",
                "contract_code": contract_code,
                "front_price": cl_front[i],
                "next_price": cl_next[i],
                "front_expiry_days": front_exp,
                "next_expiry_days": next_exp,
                "is_roll_date": "true" if is_roll else "false",
                "prev_front_price_new_contract": prev_new,
                "as_of_ts": _ts(d, "20:00:00Z"),
                "publication_ts": _ts(d, "20:05:00Z"),
            }
        )

    # Point-in-time factor inputs with an intentional publication lag example!
    # Record 1 published before day 0; Record 2 for day 4 published AFTER day 7 (so days 4..6 must NOT see it!)
    factor_rows: list[dict[str, Any]] = [
        {
            "date": "2023-12-29",
            "asset_id": "EQ_SPX",
            "dividend_yield": 0.018,
            "earnings_quality": 0.85,
            "foreign_short_rate": None,
            "domestic_short_rate": usd_rate,
            "convenience_yield_proxy": None,
            "as_of_ts": "2023-12-29T20:00:00Z",
            "publication_ts": "2024-01-01T12:00:00Z",
        },
        {
            "date": days[4].isoformat(),
            "asset_id": "EQ_SPX",
            "dividend_yield": 0.025,
            "earnings_quality": 1.25,
            "foreign_short_rate": None,
            "domestic_short_rate": usd_rate,
            "convenience_yield_proxy": None,
            "as_of_ts": _ts(days[4], "20:00:00Z"),
            "publication_ts": _ts(days[8], "12:00:00Z"),  # Lagged! Only visible starting days[8]
        },
        {
            "date": "2023-12-29",
            "asset_id": "EQ_STOXX",
            "dividend_yield": 0.032,
            "earnings_quality": 0.45,
            "foreign_short_rate": None,
            "domestic_short_rate": eur_rate,
            "convenience_yield_proxy": None,
            "as_of_ts": "2023-12-29T20:00:00Z",
            "publication_ts": "2024-01-01T12:00:00Z",
        },
        {
            "date": "2023-12-29",
            "asset_id": "FX_EURUSD",
            "dividend_yield": None,
            "earnings_quality": None,
            "foreign_short_rate": eur_rate,
            "domestic_short_rate": usd_rate,
            "convenience_yield_proxy": None,
            "as_of_ts": "2023-12-29T20:00:00Z",
            "publication_ts": "2024-01-01T12:00:00Z",
        },
        {
            "date": "2023-12-29",
            "asset_id": "CMD_CL",
            "dividend_yield": None,
            "earnings_quality": None,
            "foreign_short_rate": None,
            "domestic_short_rate": usd_rate,
            "convenience_yield_proxy": 0.04,
            "as_of_ts": "2023-12-29T20:00:00Z",
            "publication_ts": "2024-01-01T12:00:00Z",
        },
    ]

    _write_csv(out_dir / "contract_metadata.csv", list(metadata[0].keys()), metadata)
    _write_csv(out_dir / "prices.csv", list(price_rows[0].keys()), price_rows)
    _write_csv(out_dir / "fx_rates.csv", list(fx_rows[0].keys()), fx_rows)
    _write_csv(out_dir / "factor_inputs.csv", list(factor_rows[0].keys()), factor_rows)

    # Independent analytical hand-calculated oracle (pure Python math!)
    # 1. EQ_STOXX day 1 (2024-01-03): local_ret = 202/200 - 1 = 0.01; fx_factor = 1.1055/1.1000 = 1.005
    #    gross_excess_return_base = 0.01 * 1.005 = 0.01005
    #    financing_cost_drag = (30 / 10000) / 252
    stoxx_d1_local = 202.0 / 200.0 - 1.0
    stoxx_d1_fx_factor = 1.1055 / 1.1000
    stoxx_d1_gross_usd = stoxx_d1_local * stoxx_d1_fx_factor
    stoxx_fin_drag = (30.0 / 10000.0) / 252.0

    # 2. CMD_CL roll day (day 5 = 2024-01-09) and post-roll day (day 6 = 2024-01-10):
    #    day 5: local_ret = 82.000 / 81.592 - 1; roll_cost_drag = 5.0 / 10000 = 0.0005
    #    day 6: local_ret = 78.780 / 78.000 - 1 = 0.01 EXACTLY (no -3.93% jump!)
    cl_d5_local = 82.000 / 81.592 - 1.0
    cl_d5_roll_cost = 5.0 / 10000.0
    cl_d6_local = 78.780 / 78.000 - 1.0
    cl_fin_drag = (20.0 / 10000.0) / 252.0
    # Implied roll yield on day 5: ln(82.0 / 78.0) * (365 / 30)
    cl_d5_roll_yield = math.log(82.0 / 78.0) * (365.0 / 30.0)
    # Commodity carry raw on day 5: implied_roll_yield - (roll_cost_bps/10000)*(365/30)
    cl_d5_carry_raw = cl_d5_roll_yield - (5.0 / 10000.0) * (365.0 / 30.0)

    # 3. FX_EURUSD day 1: spot_ret = 1.1055/1.1000 - 1 = 0.005; carry_ret = (0.035 - 0.050)/252
    fx_d1_spot = 1.1055 / 1.1000 - 1.0
    fx_d1_carry_ret = (eur_rate - usd_rate) / 252.0
    fx_d1_gross = fx_d1_spot + fx_d1_carry_ret
    fx_fin_drag = (10.0 / 10000.0) / 252.0
    fx_carry_raw = (eur_rate - usd_rate) - (10.0 / 10000.0)

    # 4. EQ_SPX carry on day 6 (2024-01-10) vs day 8 (2024-01-12):
    #    On day 6, record for day 4 has publication_ts = day 8 12:00Z -> MUST NOT be visible!
    #    So day 6 uses dividend_yield = 0.018: carry = 0.018 - 0.050 - 0.0025 = -0.0345
    #    On day 8 (cutoff 21:00Z >= 12:00Z), record for day 4 IS visible!
    #    So day 8 uses dividend_yield = 0.025: carry = 0.025 - 0.050 - 0.0025 = -0.0275
    spx_d6_carry_raw = 0.018 - usd_rate - (25.0 / 10000.0)
    spx_d8_carry_raw = 0.025 - usd_rate - (25.0 / 10000.0)

    oracle = {
        "schema_version": "1.0.0",
        "description": "Independent analytical hand-calculated oracle for valid_small fixture.",
        "tolerance": 1e-10,
        "checks": {
            "eq_stoxx_day1_fx_conversion": {
                "date": days[1].isoformat(),
                "asset_id": "EQ_STOXX",
                "local_return": stoxx_d1_local,
                "fx_return_factor": stoxx_d1_fx_factor,
                "gross_excess_return_base": stoxx_d1_gross_usd,
                "financing_cost_drag": stoxx_fin_drag,
                "net_return_before_tc": stoxx_d1_gross_usd - stoxx_fin_drag,
            },
            "cmd_cl_day5_roll_and_carry": {
                "date": days[5].isoformat(),
                "asset_id": "CMD_CL",
                "local_return": cl_d5_local,
                "roll_cost_drag": cl_d5_roll_cost,
                "financing_cost_drag": cl_fin_drag,
                "net_return_before_tc": cl_d5_local - cl_d5_roll_cost - cl_fin_drag,
                "implied_roll_yield": cl_d5_roll_yield,
                "carry_raw": cl_d5_carry_raw,
            },
            "cmd_cl_day6_post_roll_continuity": {
                "date": days[6].isoformat(),
                "asset_id": "CMD_CL",
                "local_return": cl_d6_local,
                "roll_cost_drag": 0.0,
                "net_return_before_tc": cl_d6_local - cl_fin_drag,
            },
            "fx_eurusd_day1_excess_return_and_carry": {
                "date": days[1].isoformat(),
                "asset_id": "FX_EURUSD",
                "local_return": fx_d1_spot,
                "carry_return_component": fx_d1_carry_ret,
                "gross_excess_return_base": fx_d1_gross,
                "net_return_before_tc": fx_d1_gross - fx_fin_drag,
                "carry_raw": fx_carry_raw,
            },
            "eq_spx_publication_lag_gating": {
                "pre_publication_date": days[6].isoformat(),
                "pre_publication_carry_raw": spx_d6_carry_raw,
                "post_publication_date": days[8].isoformat(),
                "post_publication_carry_raw": spx_d8_carry_raw,
            },
        },
    }
    (out_dir / "expected_reference.json").write_text(json.dumps(oracle, indent=2) + "\n", encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "fixture_name": "valid_small",
                "provenance": "synthetic_hand_calculated_reference",
                "n_days": len(days),
                "n_assets": len(metadata),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def generate_synthetic_multi_year(seed: int = 42, n_days: int = 1008) -> None:
    """Create a deterministic 4-year synthetic cross-asset dataset (9 assets, 3 classes)."""
    out_dir = FIXTURES_DIR / "synthetic_multi_year"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    days = _business_days(date(2021, 1, 4), n_days)

    metadata = [
        # Equities (3)
        {
            "asset_id": "EQ_SPX",
            "asset_class": "equity",
            "quote_currency": "USD",
            "base_currency": None,
            "instrument_type": "equity_index_future",
            "carry_convention": "dividend_minus_financing",
            "contract_multiplier": 50.0,
            "tick_size": 0.25,
            "tc_bps": 2.0,
            "financing_spread_bps": 25.0,
            "roll_cost_bps": 1.5,
            "active_from": "2020-01-01",
            "active_to": None,
        },
        {
            "asset_id": "EQ_STOXX",
            "asset_class": "equity",
            "quote_currency": "EUR",
            "base_currency": None,
            "instrument_type": "equity_index_future",
            "carry_convention": "dividend_minus_financing",
            "contract_multiplier": 10.0,
            "tick_size": 1.0,
            "tc_bps": 2.5,
            "financing_spread_bps": 30.0,
            "roll_cost_bps": 2.0,
            "active_from": "2020-01-01",
            "active_to": None,
        },
        {
            "asset_id": "EQ_NKY",
            "asset_class": "equity",
            "quote_currency": "JPY",
            "base_currency": None,
            "instrument_type": "equity_index_future",
            "carry_convention": "dividend_minus_financing",
            "contract_multiplier": 500.0,
            "tick_size": 5.0,
            "tc_bps": 3.0,
            "financing_spread_bps": 30.0,
            "roll_cost_bps": 2.5,
            "active_from": "2020-01-01",
            "active_to": None,
        },
        # FX forwards (3)
        {
            "asset_id": "FX_EURUSD",
            "asset_class": "fx",
            "quote_currency": "USD",
            "base_currency": "EUR",
            "instrument_type": "fx_forward",
            "carry_convention": "interest_rate_differential",
            "contract_multiplier": 125000.0,
            "tick_size": 0.0001,
            "tc_bps": 1.0,
            "financing_spread_bps": 10.0,
            "roll_cost_bps": 0.5,
            "active_from": "2020-01-01",
            "active_to": None,
        },
        {
            "asset_id": "FX_USDJPY",
            "asset_class": "fx",
            "quote_currency": "JPY",
            "base_currency": "USD",
            "instrument_type": "fx_forward",
            "carry_convention": "interest_rate_differential",
            "contract_multiplier": 100000.0,
            "tick_size": 0.01,
            "tc_bps": 1.2,
            "financing_spread_bps": 10.0,
            "roll_cost_bps": 0.5,
            "active_from": "2020-01-01",
            "active_to": None,
        },
        {
            "asset_id": "FX_GBPUSD",
            "asset_class": "fx",
            "quote_currency": "USD",
            "base_currency": "GBP",
            "instrument_type": "fx_forward",
            "carry_convention": "interest_rate_differential",
            "contract_multiplier": 62500.0,
            "tick_size": 0.0001,
            "tc_bps": 1.5,
            "financing_spread_bps": 12.0,
            "roll_cost_bps": 0.5,
            "active_from": "2020-01-01",
            "active_to": None,
        },
        # Commodities (3)
        {
            "asset_id": "CMD_CL",
            "asset_class": "commodity",
            "quote_currency": "USD",
            "base_currency": None,
            "instrument_type": "commodity_future",
            "carry_convention": "futures_term_structure_roll",
            "contract_multiplier": 1000.0,
            "tick_size": 0.01,
            "tc_bps": 4.0,
            "financing_spread_bps": 20.0,
            "roll_cost_bps": 5.0,
            "active_from": "2020-01-01",
            "active_to": None,
        },
        {
            "asset_id": "CMD_GC",
            "asset_class": "commodity",
            "quote_currency": "USD",
            "base_currency": None,
            "instrument_type": "commodity_future",
            "carry_convention": "futures_term_structure_roll",
            "contract_multiplier": 100.0,
            "tick_size": 0.10,
            "tc_bps": 3.0,
            "financing_spread_bps": 18.0,
            "roll_cost_bps": 3.0,
            "active_from": "2020-01-01",
            "active_to": None,
        },
        {
            "asset_id": "CMD_HG",
            "asset_class": "commodity",
            "quote_currency": "USD",
            "base_currency": None,
            "instrument_type": "commodity_future",
            "carry_convention": "futures_term_structure_roll",
            "contract_multiplier": 25000.0,
            "tick_size": 0.0005,
            "tc_bps": 4.5,
            "financing_spread_bps": 22.0,
            "roll_cost_bps": 4.5,
            "active_from": "2020-01-01",
            "active_to": None,
        },
    ]

    # Simulate macro short rates (smooth cycles) for USD, EUR, JPY, GBP
    t_grid = np.linspace(0.0, 4.0, n_days)
    rates = {
        "USD": 0.010 + 0.042 * (0.5 - 0.5 * np.cos(np.pi * np.clip(t_grid / 2.2, 0.0, 1.0))),
        "EUR": -0.004 + 0.040 * (0.5 - 0.5 * np.cos(np.pi * np.clip((t_grid - 0.4) / 2.3, 0.0, 1.0))),
        "JPY": -0.001 + 0.004 * np.clip((t_grid - 2.8) / 1.2, 0.0, 1.0),
        "GBP": 0.005 + 0.048 * (0.5 - 0.5 * np.cos(np.pi * np.clip((t_grid - 0.15) / 2.1, 0.0, 1.0))),
    }

    # Correlated global risk-on factor + idiosyncratic AR(1) trend regimes for realistic cross-asset dynamics
    global_shock = rng.normal(0.0, 1.0, size=n_days)
    fx_vol = {"EUR": 0.085, "JPY": 0.095, "GBP": 0.095}
    fx_spot = {"USD": np.ones(n_days), "EUR": np.zeros(n_days), "JPY": np.zeros(n_days), "GBP": np.zeros(n_days)}
    fx_spot["EUR"][0] = 1.18
    fx_spot["JPY"][0] = 1.0 / 104.0  # USD per 1 JPY
    fx_spot["GBP"][0] = 1.35

    for ccy in ("EUR", "JPY", "GBP"):
        trend = 0.0
        for i in range(1, n_days):
            carry_diff = rates[ccy][i - 1] - rates["USD"][i - 1]
            trend = 0.96 * trend + 0.00018 * rng.normal()
            beta = 0.25 if ccy in ("EUR", "GBP") else -0.20
            eps = rng.normal(0.0, 1.0)
            daily_vol = fx_vol[ccy] / math.sqrt(252.0)
            ret = 0.15 * carry_diff / 252.0 + trend + daily_vol * (beta * global_shock[i] + math.sqrt(1 - beta**2) * eps)
            fx_spot[ccy][i] = fx_spot[ccy][i - 1] * math.exp(ret)

    fx_rows: list[dict[str, Any]] = []
    for i, d in enumerate(days):
        for ccy in ("USD", "EUR", "JPY", "GBP"):
            fx_rows.append(
                {
                    "date": d.isoformat(),
                    "currency": ccy,
                    "usd_per_unit": round(float(fx_spot[ccy][i]), 8),
                    "short_rate": round(float(rates[ccy][i]), 6),
                    "as_of_ts": _ts(d, "19:00:00Z"),
                    "publication_ts": _ts(d, "19:05:00Z"),
                }
            )

    # Asset price series + term structure + quarterly/bi-monthly rolls
    asset_params = {
        "EQ_SPX": {"init": 3700.0, "vol": 0.16, "beta": 0.85, "drift": 0.075, "roll_period": 63, "slope_mu": -0.015},
        "EQ_STOXX": {"init": 3550.0, "vol": 0.19, "beta": 0.78, "drift": 0.060, "roll_period": 63, "slope_mu": -0.008},
        "EQ_NKY": {"init": 27200.0, "vol": 0.21, "beta": 0.65, "drift": 0.065, "roll_period": 63, "slope_mu": 0.005},
        "CMD_CL": {"init": 49.0, "vol": 0.34, "beta": 0.35, "drift": 0.030, "roll_period": 42, "slope_mu": 0.065},
        "CMD_GC": {"init": 1910.0, "vol": 0.14, "beta": 0.05, "drift": 0.050, "roll_period": 63, "slope_mu": -0.030},
        "CMD_HG": {"init": 3.55, "vol": 0.24, "beta": 0.45, "drift": 0.035, "roll_period": 63, "slope_mu": 0.020},
    }

    price_rows: list[dict[str, Any]] = []
    factor_rows: list[dict[str, Any]] = []

    # Equities & Commodities
    for aid, params in asset_params.items():
        roll_period = params["roll_period"]
        # Continuous economic return + regime-varying slope (backwardation > 0 means front > next)
        slope = np.zeros(n_days)
        slope[0] = params["slope_mu"]
        trend = 0.0
        vol_state = params["vol"]
        econ_rets = np.zeros(n_days)
        for i in range(1, n_days):
            slope[i] = 0.985 * slope[i - 1] + 0.015 * params["slope_mu"] + 0.006 * rng.normal()
            trend = 0.965 * trend + 0.00024 * rng.normal()
            # Mild vol clustering so defensive low-vol has realistic persistence
            vol_state = float(np.clip(0.97 * vol_state + 0.03 * params["vol"] + 0.008 * rng.normal(), 0.08, 0.55))
            eps = rng.normal(0.0, 1.0)
            beta = params["beta"]
            # Mild carry + momentum + defensive premia in synthetic DGP
            carry_alpha = 0.25 * slope[i - 1] / 252.0 if aid.startswith("CMD_") else 0.0
            def_alpha = -0.08 * (vol_state - params["vol"]) / 252.0
            dvol = vol_state / math.sqrt(252.0)
            econ_rets[i] = (
                params["drift"] / 252.0
                + trend
                + carry_alpha
                + def_alpha
                + dvol * (beta * global_shock[i] + math.sqrt(max(1.0 - beta**2, 0.05)) * eps)
            )

        # Convert economic returns into realistic spliced contract prices with contract rolls
        contract_idx = 1
        front_px = params["init"]
        prev_day_next_px = None
        for i, d in enumerate(days):
            days_in_cycle = i % roll_period
            front_exp = (roll_period - days_in_cycle) + 10
            next_exp = front_exp + roll_period
            is_roll = (days_in_cycle == roll_period - 1) and (i < n_days - 1)
            prev_new_contract_px = None

            if i > 0:
                if days_in_cycle == 0:
                    # We rolled at close of i-1 from old contract to next contract!
                    contract_idx += 1
                    assert prev_day_next_px is not None
                    prev_new_contract_px = prev_day_next_px
                    front_px = prev_new_contract_px * (1.0 + econ_rets[i])
                else:
                    front_px = front_px * (1.0 + econ_rets[i])

            # Next contract price reflects annualized slope (positive slope = backwardation -> next < front)
            dt_years = (next_exp - front_exp) / 365.0
            next_px = front_px * math.exp(-slope[i] * dt_years)
            code = f"{aid.split('_')[1]}_C{contract_idx:02d}"

            price_rows.append(
                {
                    "date": d.isoformat(),
                    "asset_id": aid,
                    "contract_code": code,
                    "front_price": round(float(front_px), 6),
                    "next_price": round(float(next_px), 6),
                    "front_expiry_days": int(front_exp),
                    "next_expiry_days": int(next_exp),
                    "is_roll_date": "true" if is_roll else "false",
                    "prev_front_price_new_contract": (
                        round(float(prev_new_contract_px), 6) if prev_new_contract_px is not None else None
                    ),
                    "as_of_ts": _ts(d, "20:00:00Z"),
                    "publication_ts": _ts(d, "20:05:00Z"),
                }
            )
            prev_day_next_px = float(next_px)

    # FX Forward prices (quoted as pair spot price; USDJPY = 1 / usd_per_jpy)
    fx_pair_specs = [
        ("FX_EURUSD", "EUR", "USD"),
        ("FX_USDJPY", "USD", "JPY"),
        ("FX_GBPUSD", "GBP", "USD"),
    ]
    for aid, bccy, qccy in fx_pair_specs:
        for i, d in enumerate(days):
            spot = fx_spot[bccy][i] / fx_spot[qccy][i]
            r_base = rates[bccy][i]
            r_quote = rates[qccy][i]
            fwd_next = spot * math.exp((r_quote - r_base) * 30.0 / 365.0)
            price_rows.append(
                {
                    "date": d.isoformat(),
                    "asset_id": aid,
                    "contract_code": f"{bccy}{qccy}_1M",
                    "front_price": round(float(spot), 6),
                    "next_price": round(float(fwd_next), 6),
                    "front_expiry_days": 30,
                    "next_expiry_days": 60,
                    "is_roll_date": "true" if (i % 21 == 20 and i < n_days - 1) else "false",
                    "prev_front_price_new_contract": None,
                    "as_of_ts": _ts(d, "20:00:00Z"),
                    "publication_ts": _ts(d, "20:05:00Z"),
                }
            )

    # Point-in-time factor_inputs
    # 1. Initial pre-start equity fundamentals published before day 0
    eq_base_div = {"EQ_SPX": 0.016, "EQ_STOXX": 0.032, "EQ_NKY": 0.021}
    eq_base_qual = {"EQ_SPX": 0.85, "EQ_STOXX": 0.35, "EQ_NKY": 0.55}
    eq_ccy = {"EQ_SPX": "USD", "EQ_STOXX": "EUR", "EQ_NKY": "JPY"}
    for aid in ("EQ_SPX", "EQ_STOXX", "EQ_NKY"):
        factor_rows.append(
            {
                "date": "2020-12-31",
                "asset_id": aid,
                "dividend_yield": eq_base_div[aid],
                "earnings_quality": eq_base_qual[aid],
                "foreign_short_rate": None,
                "domestic_short_rate": round(float(rates[eq_ccy[aid]][0]), 6),
                "convenience_yield_proxy": None,
                "as_of_ts": "2020-12-31T20:00:00Z",
                "publication_ts": "2021-01-03T12:00:00Z",
            }
        )

    # Monthly equity fundamental updates with realistic 15-business-day publication lag!
    for i in range(21, n_days, 21):
        ref_d = days[i]
        pub_d = days[min(i + 15, n_days - 1)] if (i + 15) < n_days else (days[-1] + timedelta(days=21))
        for aid in ("EQ_SPX", "EQ_STOXX", "EQ_NKY"):
            eq_base_div[aid] = float(np.clip(eq_base_div[aid] + 0.0012 * rng.normal(), 0.008, 0.060))
            eq_base_qual[aid] = float(np.clip(eq_base_qual[aid] + 0.08 * rng.normal(), -1.5, 2.0))
            factor_rows.append(
                {
                    "date": ref_d.isoformat(),
                    "asset_id": aid,
                    "dividend_yield": round(eq_base_div[aid], 6),
                    "earnings_quality": round(eq_base_qual[aid], 6),
                    "foreign_short_rate": None,
                    "domestic_short_rate": round(float(rates[eq_ccy[aid]][i]), 6),
                    "convenience_yield_proxy": None,
                    "as_of_ts": _ts(ref_d, "20:00:00Z"),
                    "publication_ts": _ts(pub_d, "12:00:00Z"),
                }
            )

    # Daily FX short rate inputs (published same day at 19:05Z before 21:00Z cutoff)
    for i, d in enumerate(days):
        for aid, bccy, qccy in fx_pair_specs:
            factor_rows.append(
                {
                    "date": d.isoformat(),
                    "asset_id": aid,
                    "dividend_yield": None,
                    "earnings_quality": None,
                    "foreign_short_rate": round(float(rates[bccy][i]), 6),
                    "domestic_short_rate": round(float(rates[qccy][i]), 6),
                    "convenience_yield_proxy": None,
                    "as_of_ts": _ts(d, "19:00:00Z"),
                    "publication_ts": _ts(d, "19:05:00Z"),
                }
            )

    _write_csv(out_dir / "contract_metadata.csv", list(metadata[0].keys()), metadata)
    _write_csv(out_dir / "prices.csv", list(price_rows[0].keys()), price_rows)
    _write_csv(out_dir / "fx_rates.csv", list(fx_rows[0].keys()), fx_rows)
    _write_csv(out_dir / "factor_inputs.csv", list(factor_rows[0].keys()), factor_rows)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "fixture_name": "synthetic_multi_year",
                "provenance": "synthetic_seeded_dgp",
                "seed": seed,
                "n_days": n_days,
                "date_start": days[0].isoformat(),
                "date_end": days[-1].isoformat(),
                "n_assets": len(metadata),
                "asset_classes": ["equity", "fx", "commodity"],
                "notes": "Synthetic multi-asset dataset generated deterministically for offline research evaluation. Contains no live market, customer, or proprietary data.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def generate_invalid_and_boundary_fixtures() -> None:
    """Generate explicit invalid and boundary fixture artifacts for automated testing."""
    inv_dir = FIXTURES_DIR / "invalid"
    bnd_dir = FIXTURES_DIR / "boundary"
    inv_dir.mkdir(parents=True, exist_ok=True)
    bnd_dir.mkdir(parents=True, exist_ok=True)

    # 1. Missing column in prices
    _write_csv(
        inv_dir / "missing_columns_prices.csv",
        ["date", "asset_id", "contract_code", "front_price", "front_expiry_days", "is_roll_date", "as_of_ts"],
        [
            {
                "date": "2024-01-02",
                "asset_id": "EQ_SPX",
                "contract_code": "ESH24",
                "front_price": 100.0,
                "front_expiry_days": 60,
                "is_roll_date": "false",
                "as_of_ts": "2024-01-02T20:00:00Z",
            }
        ],
    )

    # 2. Negative price
    _write_csv(
        inv_dir / "negative_price.csv",
        [
            "date",
            "asset_id",
            "contract_code",
            "front_price",
            "next_price",
            "front_expiry_days",
            "next_expiry_days",
            "is_roll_date",
            "prev_front_price_new_contract",
            "as_of_ts",
            "publication_ts",
        ],
        [
            {
                "date": "2024-01-02",
                "asset_id": "EQ_SPX",
                "contract_code": "ESH24",
                "front_price": -5.0,
                "next_price": 100.0,
                "front_expiry_days": 60,
                "next_expiry_days": 150,
                "is_roll_date": "false",
                "prev_front_price_new_contract": None,
                "as_of_ts": "2024-01-02T20:00:00Z",
                "publication_ts": "2024-01-02T20:05:00Z",
            }
        ],
    )

    # 3. Invalid roll transition (contract changes with missing prev_front_price_new_contract)
    _write_csv(
        inv_dir / "invalid_roll_transition.csv",
        [
            "date",
            "asset_id",
            "contract_code",
            "front_price",
            "next_price",
            "front_expiry_days",
            "next_expiry_days",
            "is_roll_date",
            "prev_front_price_new_contract",
            "as_of_ts",
            "publication_ts",
        ],
        [
            {
                "date": "2024-01-02",
                "asset_id": "CMD_CL",
                "contract_code": "CLG24",
                "front_price": 80.0,
                "next_price": 78.0,
                "front_expiry_days": 10,
                "next_expiry_days": 40,
                "is_roll_date": "true",
                "prev_front_price_new_contract": None,
                "as_of_ts": "2024-01-02T20:00:00Z",
                "publication_ts": "2024-01-02T20:05:00Z",
            },
            {
                "date": "2024-01-03",
                "asset_id": "CMD_CL",
                "contract_code": "CLH24",
                "front_price": 78.5,
                "next_price": 76.5,
                "front_expiry_days": 39,
                "next_expiry_days": 69,
                "is_roll_date": "false",
                "prev_front_price_new_contract": None,
                "as_of_ts": "2024-01-03T20:00:00Z",
                "publication_ts": "2024-01-03T20:05:00Z",
            },
        ],
    )

    # 4. Boundary fixture: missing factor inputs for EQ_SPX
    _write_csv(
        bnd_dir / "missing_factor_inputs.csv",
        [
            "date",
            "asset_id",
            "dividend_yield",
            "earnings_quality",
            "foreign_short_rate",
            "domestic_short_rate",
            "convenience_yield_proxy",
            "as_of_ts",
            "publication_ts",
        ],
        [
            {
                "date": "2023-12-29",
                "asset_id": "EQ_STOXX",
                "dividend_yield": 0.032,
                "earnings_quality": 0.45,
                "foreign_short_rate": None,
                "domestic_short_rate": 0.035,
                "convenience_yield_proxy": None,
                "as_of_ts": "2023-12-29T20:00:00Z",
                "publication_ts": "2024-01-01T12:00:00Z",
            },
            {
                "date": "2023-12-29",
                "asset_id": "FX_EURUSD",
                "dividend_yield": None,
                "earnings_quality": None,
                "foreign_short_rate": 0.035,
                "domestic_short_rate": 0.050,
                "convenience_yield_proxy": None,
                "as_of_ts": "2023-12-29T20:00:00Z",
                "publication_ts": "2024-01-01T12:00:00Z",
            },
        ],
    )

    (bnd_dir / "publication_lag_scenario.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "scenario": "no_look_ahead_from_publication_lag",
                "description": "Verifies that mutating or injecting future-published factor_inputs or prices with publication_ts > as_of_ts produces bitwise-identical signals and weights prior to publication.",
                "target_asset": "EQ_SPX",
                "reference_date": "2024-01-08",
                "publication_date": "2024-01-12",
                "expected_pre_pub_dividend_yield": 0.018,
                "expected_post_pub_dividend_yield": 0.025,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    generate_valid_small_and_oracle()
    generate_synthetic_multi_year(seed=42, n_days=1008)
    generate_invalid_and_boundary_fixtures()
    print(f"Generated deterministic fixtures in {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
