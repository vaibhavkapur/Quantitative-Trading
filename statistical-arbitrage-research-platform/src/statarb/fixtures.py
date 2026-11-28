"""Deterministic synthetic fixture generators for unit, reference, and walk-forward testing.

All fixtures are 100% synthetic with fixed RNG seeds; no live market feeds, real transactions,
or proprietary data are used.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from statarb.contracts import compute_frame_hash


def _trading_dates(start: date, n_days: int) -> list[date]:
    out: list[date] = []
    cur = start
    while len(out) < n_days:
        if cur.weekday() < 5:
            out.append(cur)
        cur += timedelta(days=1)
    return out


def _simulate_ou_spread(
    rng: np.random.Generator,
    n_days: int,
    phi: float,
    sigma: float,
) -> np.ndarray:
    s = np.zeros(n_days, dtype=np.float64)
    eps = rng.normal(0.0, sigma, size=n_days)
    for t in range(1, n_days):
        s[t] = phi * s[t - 1] + eps[t]
    return s


@dataclass(frozen=True)
class ReferenceFixtureBundle:
    prices: pl.DataFrame
    borrow: pl.DataFrame
    ground_truth: dict[str, Any]


def generate_reference_pair_fixture(seed: int = 7, n_days: int = 260) -> ReferenceFixtureBundle:
    """Generate a small deterministic reference fixture with known analytical DGP.

    Contains:
    - REF_Y, REF_X: cointegrated pair with y_t = alpha + beta * x_t + s_t,
      alpha=5.0, beta=1.5, s_t = 0.85 * s_{t-1} + N(0, 0.35^2) (theoretical HL ~ 4.265 days).
    - REF_IND_A, REF_IND_B: independent random walks (non-cointegrated control).
    """
    rng = np.random.default_rng(seed)
    dates = _trading_dates(date(2023, 1, 2), n_days)

    alpha_true = 5.0
    beta_true = 1.5
    phi_true = 0.85
    sigma_s = 0.35
    true_half_life = -math.log(2.0) / math.log(phi_true)

    # Positive random walk for REF_X
    x_steps = rng.normal(0.0, 0.75, size=n_days)
    x_price = 50.0 + np.cumsum(x_steps)
    x_price = np.maximum(x_price, 10.0)

    spread = _simulate_ou_spread(rng, n_days, phi=phi_true, sigma=sigma_s)
    y_price = alpha_true + beta_true * x_price + spread
    y_price = np.maximum(y_price, 10.0)

    # Two independent random walks
    ind_a = np.maximum(60.0 + np.cumsum(rng.normal(0.02, 1.1, size=n_days)), 10.0)
    ind_b = np.maximum(45.0 + np.cumsum(rng.normal(-0.01, 0.9, size=n_days)), 10.0)

    series_map = {
        "REF_X": (x_price, "TECH"),
        "REF_Y": (y_price, "TECH"),
        "REF_IND_A": (ind_a, "TECH"),
        "REF_IND_B": (ind_b, "TECH"),
    }

    price_rows: list[dict[str, Any]] = []
    borrow_rows: list[dict[str, Any]] = []
    for sym, (arr, sector) in series_map.items():
        for i, d in enumerate(dates):
            px = round(float(arr[i]), 6)
            price_rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "close_adj": px,
                    "close_raw": px,
                    "volume": 250_000.0,
                    "adv_usd": round(px * 250_000.0, 2),
                    "market_cap_usd": round(px * 25_000_000.0, 2),
                    "sector": sector,
                    "is_active": True,
                    "split_factor": 1.0,
                    "cash_dividend": 0.0,
                }
            )
            borrow_rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "borrow_available": True,
                    "borrow_fee_bps_annual": 40.0,
                    "max_short_notional": 500_000.0,
                }
            )

    prices_df = pl.DataFrame(price_rows).sort(["date", "symbol"])
    borrow_df = pl.DataFrame(borrow_rows).sort(["date", "symbol"])

    ground_truth = {
        "fixture_name": "reference_pair_fixture",
        "seed": seed,
        "n_days": n_days,
        "cointegrated_pair": {
            "y_symbol": "REF_Y",
            "x_symbol": "REF_X",
            "true_alpha": alpha_true,
            "true_beta": beta_true,
            "true_phi": phi_true,
            "true_half_life_days": round(true_half_life, 6),
            "spread_innovation_std": sigma_s,
        },
        "independent_pair": {
            "y_symbol": "REF_IND_A",
            "x_symbol": "REF_IND_B",
        },
    }
    return ReferenceFixtureBundle(prices=prices_df, borrow=borrow_df, ground_truth=ground_truth)


def generate_synthetic_universe_fixture(
    seed: int = 42,
    n_days: int = 600,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Generate a rich multi-sector synthetic universe for walk-forward research.

    Includes:
    - TECH_A / TECH_B: log-cointegrated pair with 2-for-1 split on TECH_A (day 320)
      and $0.50 cash dividend (day 420)
    - TECH_C / TECH_D: log-cointegrated pair with borrow unavailability on TECH_D (days 380-395)
    - TECH_DELIST: active through day 220 then delisted (point-in-time survivorship test)
    - FIN_A / FIN_B / FIN_C: 3-asset log-cointegrated basket (Johansen test)
    - FIN_BREAK_A / FIN_BREAK_B: cointegrated for first 300 days, then breaks down
    - IND_RW1, IND_RW2, IND_RW3: independent random walks (multiple-testing negative controls)
    - IND_PENNY: low price & low ADV (universe filter exclusion)
    """
    rng = np.random.default_rng(seed)
    dates = _trading_dates(date(2022, 1, 3), n_days)

    # Sector common factors (log-space random walks)
    tech_factor = np.cumsum(rng.normal(0.0002, 0.012, size=n_days))
    fin_factor = np.cumsum(rng.normal(0.00015, 0.011, size=n_days))
    ind_factor = np.cumsum(rng.normal(0.0001, 0.010, size=n_days))

    # 1. TECH_A, TECH_B: log_A = 0.3 + 1.15 * log_B + OU(hl~7d)
    log_tech_b = math.log(80.0) + tech_factor + np.cumsum(rng.normal(0.0, 0.006, size=n_days))
    ou_ab = _simulate_ou_spread(rng, n_days, phi=math.exp(-math.log(2.0) / 7.0), sigma=0.014)
    log_tech_a = 0.30 + 1.15 * log_tech_b + ou_ab

    # 2. TECH_C, TECH_D: log_C = -0.1 + 0.90 * log_D + OU(hl~9d)
    log_tech_d = math.log(65.0) + tech_factor + np.cumsum(rng.normal(0.0, 0.007, size=n_days))
    ou_cd = _simulate_ou_spread(rng, n_days, phi=math.exp(-math.log(2.0) / 9.0), sigma=0.015)
    log_tech_c = -0.10 + 0.90 * log_tech_d + ou_cd

    # 3. TECH_DELIST: active days 0..220, then delisted
    log_tech_delist = (
        math.log(30.0) + tech_factor + np.cumsum(rng.normal(-0.001, 0.015, size=n_days))
    )

    # 4. FIN_A, FIN_B, FIN_C: 3-asset cointegrated basket (2 independent I(1) trends -> rank 1)
    #    log_FIN_A = 0.2 + 0.65 * log_FIN_B + 0.45 * log_FIN_C + OU(hl~5d)
    log_fin_b = math.log(55.0) + fin_factor + np.cumsum(rng.normal(0.0, 0.012, size=n_days))
    log_fin_c = math.log(70.0) + np.cumsum(rng.normal(0.0001, 0.014, size=n_days))
    ou_basket = _simulate_ou_spread(rng, n_days, phi=math.exp(-math.log(2.0) / 5.0), sigma=0.012)
    log_fin_a = 0.20 + 0.65 * log_fin_b + 0.45 * log_fin_c + ou_basket

    # 5. FIN_BREAK_A, FIN_BREAK_B: cointegrated days 0..299, breaks down days 300..n_days
    log_break_b = math.log(50.0) + fin_factor + np.cumsum(rng.normal(0.0, 0.008, size=n_days))
    ou_break = _simulate_ou_spread(rng, n_days, phi=math.exp(-math.log(2.0) / 6.0), sigma=0.013)
    drift_break = np.zeros(n_days, dtype=np.float64)
    break_day_idx = 300
    if n_days > break_day_idx:
        drift_break[break_day_idx:] = np.cumsum(
            rng.normal(0.0025, 0.016, size=n_days - break_day_idx)
        )
    log_break_a = 0.15 + 1.05 * log_break_b + ou_break + drift_break

    # 6. IND_RW1, IND_RW2, IND_RW3: independent random walks
    log_ind_1 = (
        math.log(40.0) + 0.3 * ind_factor + np.cumsum(rng.normal(0.0002, 0.014, size=n_days))
    )
    log_ind_2 = (
        math.log(90.0) + 0.2 * ind_factor + np.cumsum(rng.normal(-0.0001, 0.015, size=n_days))
    )
    log_ind_3 = (
        math.log(62.0) + 0.1 * ind_factor + np.cumsum(rng.normal(0.0001, 0.013, size=n_days))
    )

    # 7. IND_PENNY: low price & low liquidity
    log_penny = math.log(2.5) + np.cumsum(rng.normal(0.0, 0.012, size=n_days))

    raw_specs: dict[str, tuple[np.ndarray, str, float]] = {
        "TECH_A": (np.exp(log_tech_a), "TECH", 450_000.0),
        "TECH_B": (np.exp(log_tech_b), "TECH", 500_000.0),
        "TECH_C": (np.exp(log_tech_c), "TECH", 350_000.0),
        "TECH_D": (np.exp(log_tech_d), "TECH", 380_000.0),
        "TECH_DELIST": (np.exp(log_tech_delist), "TECH", 200_000.0),
        "FIN_A": (np.exp(log_fin_a), "FIN", 420_000.0),
        "FIN_B": (np.exp(log_fin_b), "FIN", 460_000.0),
        "FIN_C": (np.exp(log_fin_c), "FIN", 410_000.0),
        "FIN_BREAK_A": (np.exp(log_break_a), "FIN", 300_000.0),
        "FIN_BREAK_B": (np.exp(log_break_b), "FIN", 320_000.0),
        "IND_RW1": (np.exp(log_ind_1), "IND", 300_000.0),
        "IND_RW2": (np.exp(log_ind_2), "IND", 280_000.0),
        "IND_RW3": (np.exp(log_ind_3), "IND", 310_000.0),
        "IND_PENNY": (np.exp(log_penny), "IND", 15_000.0),
    }

    split_day_idx = 320
    div_day_idx = 420
    delist_day_idx = 220
    borrow_outage_start = 380
    borrow_outage_end = 395

    price_rows: list[dict[str, Any]] = []
    borrow_rows: list[dict[str, Any]] = []

    for sym, (adj_arr, sector, base_vol) in raw_specs.items():
        # Construct close_raw from close_adj + corporate actions so that
        # close_adj is exact backward-adjusted total-return price
        raw_arr = adj_arr.copy()
        split_factors = np.ones(n_days, dtype=np.float64)
        dividends = np.zeros(n_days, dtype=np.float64)

        if sym == "TECH_A":
            # 2-for-1 split on split_day_idx: raw price before split is 2x post-split level
            if split_day_idx < n_days:
                split_factors[split_day_idx] = 2.0
                raw_arr[:split_day_idx] = raw_arr[:split_day_idx] * 2.0
            # Cash dividend of $0.50 on div_day_idx: pre-div adj prices slightly below raw
            # Keep consistency: (close_raw[t] + div) / close_raw[t-1] == close_adj[t]/close_adj[t-1]
            if div_day_idx < n_days:
                div_amt = 0.50
                dividends[div_day_idx] = div_amt
                prev_raw = raw_arr[div_day_idx - 1]
                div_scale = (raw_arr[div_day_idx] - div_amt) / raw_arr[div_day_idx]
                raw_arr[div_day_idx:] = raw_arr[div_day_idx:] * div_scale
                target_ratio = adj_arr[div_day_idx] / adj_arr[div_day_idx - 1]
                raw_arr[div_day_idx] = max(target_ratio * prev_raw - div_amt, 1.0)
                for t in range(div_day_idx + 1, n_days):
                    raw_arr[t] = raw_arr[t - 1] * (adj_arr[t] / adj_arr[t - 1])

        for i, d in enumerate(dates):
            if sym == "TECH_DELIST" and i > delist_day_idx:
                continue
            is_active = not (sym == "TECH_DELIST" and i == delist_day_idx)
            px_adj = round(float(adj_arr[i]), 6)
            px_raw = round(float(raw_arr[i]), 6)
            vol = round(float(base_vol * rng.uniform(0.85, 1.15)), 1)
            adv = round(px_raw * base_vol, 2)
            price_rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "close_adj": px_adj,
                    "close_raw": px_raw,
                    "volume": vol,
                    "adv_usd": adv,
                    "market_cap_usd": round(px_raw * base_vol * 100.0, 2),
                    "sector": sector,
                    "is_active": is_active,
                    "split_factor": float(split_factors[i]),
                    "cash_dividend": float(dividends[i]),
                }
            )

            # Borrow schedule
            b_avail = True
            b_fee = 45.0 if sector == "TECH" else (55.0 if sector == "FIN" else 35.0)
            max_short = 2_000_000.0
            if sym == "TECH_D" and borrow_outage_start <= i <= borrow_outage_end:
                b_avail = False
                b_fee = 350.0
                max_short = 0.0
            elif sym == "TECH_C" and 250 <= i <= 350:
                b_fee = 180.0
            elif sym == "IND_PENNY":
                b_avail = False
                b_fee = 500.0
                max_short = 0.0

            borrow_rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "borrow_available": b_avail,
                    "borrow_fee_bps_annual": b_fee,
                    "max_short_notional": max_short,
                }
            )

    prices_df = pl.DataFrame(price_rows).sort(["date", "symbol"])
    borrow_df = pl.DataFrame(borrow_rows).sort(["date", "symbol"])

    metadata = {
        "fixture_name": "synthetic_universe_fixture",
        "seed": seed,
        "n_days": n_days,
        "date_start": str(dates[0]),
        "date_end": str(dates[-1]),
        "symbols": sorted(raw_specs.keys()),
        "planted_structures": {
            "cointegrated_pairs": [["TECH_A", "TECH_B"], ["TECH_C", "TECH_D"]],
            "cointegrated_basket": ["FIN_A", "FIN_B", "FIN_C"],
            "regime_break_pair": {
                "symbols": ["FIN_BREAK_A", "FIN_BREAK_B"],
                "break_date": str(dates[break_day_idx]) if break_day_idx < n_days else None,
            },
            "independent_controls": ["IND_RW1", "IND_RW2", "IND_RW3"],
            "filtered_out_symbols": {
                "IND_PENNY": "below_min_price_and_adv",
                "TECH_DELIST": (
                    f"delisted_after_{dates[delist_day_idx]}"
                    if delist_day_idx < n_days
                    else "active"
                ),
            },
            "corporate_actions": {
                "TECH_A_split_2_for_1": str(dates[split_day_idx])
                if split_day_idx < n_days
                else None,
                "TECH_A_cash_dividend_0.50": str(dates[div_day_idx])
                if div_day_idx < n_days
                else None,
            },
            "borrow_outage": {
                "symbol": "TECH_D",
                "start_date": str(dates[borrow_outage_start])
                if borrow_outage_start < n_days
                else None,
                "end_date": str(dates[borrow_outage_end]) if borrow_outage_end < n_days else None,
            },
        },
    }
    return prices_df, borrow_df, metadata


def write_all_fixtures(fixtures_dir: str | Path) -> dict[str, Any]:
    """Generate and persist all valid, reference, and invalid/boundary fixtures."""
    root = Path(fixtures_dir)
    invalid_dir = root / "invalid"
    root.mkdir(parents=True, exist_ok=True)
    invalid_dir.mkdir(parents=True, exist_ok=True)

    # 1. Reference fixture (CSV for easy human inspection)
    ref = generate_reference_pair_fixture(seed=7, n_days=260)
    ref_prices_path = root / "reference_pair_prices.csv"
    ref_borrow_path = root / "reference_pair_borrow.csv"
    ref_truth_path = root / "reference_pair_ground_truth.json"
    ref.prices.write_csv(ref_prices_path)
    ref.borrow.write_csv(ref_borrow_path)
    ref_truth_path.write_text(json.dumps(ref.ground_truth, indent=2) + "\n", encoding="utf-8")

    # 2. Full synthetic universe fixture (Parquet for speed + Polars fidelity)
    uni_prices, uni_borrow, uni_meta = generate_synthetic_universe_fixture(seed=42, n_days=600)
    uni_prices_path = root / "synthetic_universe_prices.parquet"
    uni_borrow_path = root / "synthetic_universe_borrow.parquet"
    uni_meta_path = root / "synthetic_universe_metadata.json"
    uni_prices.write_parquet(uni_prices_path)
    uni_borrow.write_parquet(uni_borrow_path)
    uni_meta_path.write_text(json.dumps(uni_meta, indent=2) + "\n", encoding="utf-8")

    # 3. Invalid & boundary fixtures (small CSVs)
    small = ref.prices.head(40)
    # 3a. Missing price
    missing_df = small.with_columns(
        pl.when((pl.col("symbol") == "REF_Y") & (pl.col("date") == small["date"][10]))
        .then(None)
        .otherwise(pl.col("close_adj"))
        .alias("close_adj")
    )
    missing_df.write_csv(invalid_dir / "missing_prices.csv")

    # 3b. Duplicate keys
    dup_df = pl.concat([small, small.head(1)])
    dup_df.write_csv(invalid_dir / "duplicate_keys.csv")

    # 3c. Negative price
    neg_df = small.with_columns(
        pl.when((pl.col("symbol") == "REF_X") & (pl.col("date") == small["date"][5]))
        .then(-5.0)
        .otherwise(pl.col("close_adj"))
        .alias("close_adj")
    )
    neg_df.write_csv(invalid_dir / "negative_price.csv")

    # 3d. Unavailable borrow everywhere
    unavail_borrow = ref.borrow.with_columns(
        pl.lit(False).alias("borrow_available"),
        pl.lit(0.0).alias("max_short_notional"),
    )
    unavail_borrow.write_csv(invalid_dir / "unavailable_borrow.csv")

    # 3e. Unadjusted split passed as close_adj (corporate action error fixture)
    tech_a_slice = (
        uni_prices.filter(pl.col("symbol") == "TECH_A")
        .sort("date")
        .slice(315, 15)
        .with_columns(pl.col("close_raw").alias("close_adj"))
    )
    tech_a_slice.write_csv(invalid_dir / "unadjusted_split_prices.csv")

    manifest = {
        "provenance": "100% deterministic synthetic data generated by statarb.fixtures",
        "files": {
            "reference_pair_prices.csv": {
                "rows": ref.prices.height,
                "hash": compute_frame_hash(ref.prices, ["date", "symbol"]),
            },
            "reference_pair_borrow.csv": {
                "rows": ref.borrow.height,
                "hash": compute_frame_hash(ref.borrow, ["date", "symbol"]),
            },
            "synthetic_universe_prices.parquet": {
                "rows": uni_prices.height,
                "hash": compute_frame_hash(uni_prices, ["date", "symbol"]),
            },
            "synthetic_universe_borrow.parquet": {
                "rows": uni_borrow.height,
                "hash": compute_frame_hash(uni_borrow, ["date", "symbol"]),
            },
        },
        "invalid_fixtures": [
            "invalid/missing_prices.csv",
            "invalid/duplicate_keys.csv",
            "invalid/negative_price.csv",
            "invalid/unavailable_borrow.csv",
            "invalid/unadjusted_split_prices.csv",
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
