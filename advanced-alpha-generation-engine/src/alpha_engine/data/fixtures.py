"""Deterministic synthetic point-in-time market, fundamental, and corporate action fixtures.

All data generated here is strictly synthetic (non-live, zero licensing restrictions) and
includes realistic point-in-time features:
- UTC event and availability timestamps (with explicit fundamental filing lags)
- Split / reverse-split / cash dividend corporate actions (raw close vs adjusted_close)
- Mid-sample delistings to test survivorship bias handling
- Multi-sector factor structure with realistic noise so both passing and failing alpha candidates exist
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from alpha_engine.contracts.models import DatasetManifest

SECTORS = ("TECH", "FIN", "HEALTH", "IND", "CONS")


@dataclass(frozen=True)
class SyntheticDatasetBundle:
    bars: pl.DataFrame
    fundamentals: pl.DataFrame
    corporate_actions: pl.DataFrame
    membership: pl.DataFrame
    manifest: DatasetManifest


def _trading_dates(start: datetime, n_days: int) -> list[datetime]:
    dates: list[datetime] = []
    current = start
    while len(dates) < n_days:
        if current.weekday() < 5:  # Mon-Fri
            dates.append(current)
        current += timedelta(days=1)
    return dates


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_synthetic_pit_dataset(
    *,
    seed: int = 42,
    n_symbols: int = 30,
    n_trading_days: int = 440,
    n_delisted: int = 4,
    filing_lag_days: int = 45,
) -> SyntheticDatasetBundle:
    """Generate deterministic PIT equities panel with realistic corporate actions and delistings."""
    rng = np.random.default_rng(seed)

    # Market close at 21:00 UTC; bar available 5 minutes after close (21:05 UTC)
    start_dt = datetime(2023, 1, 3, 21, 0, tzinfo=UTC)
    dates = _trading_dates(start_dt, n_trading_days)
    bar_avail_delay = timedelta(minutes=5)

    symbols = [f"SYN{idx:03d}" for idx in range(1, n_symbols + 1)]
    symbol_sectors = {sym: SECTORS[i % len(SECTORS)] for i, sym in enumerate(symbols)}

    # Assign 4 symbols to delist mid-sample after distress drift
    delist_indices = list(range(n_symbols - n_delisted, n_symbols))
    delist_day_map: dict[str, int] = {}
    for k, sym_idx in enumerate(delist_indices):
        exit_day = int(190 + k * 55)
        delist_day_map[symbols[sym_idx]] = min(exit_day, n_trading_days - 20)

    # Assign realistic splits on 3 active symbols and dividends across several symbols
    split_schedule: dict[str, tuple[int, float, str]] = {
        symbols[0]: (160, 2.0, "split"),         # 2-for-1 split
        symbols[5]: (260, 3.0, "split"),         # 3-for-1 split
        symbols[10]: (310, 0.5, "reverse_split"),  # 1-for-2 reverse split
    }

    # Market & sector latent daily returns
    market_ret = rng.normal(0.00035, 0.0095, size=n_trading_days)
    sector_rets = {
        sec: rng.normal(0.0, 0.0045, size=n_trading_days) for sec in SECTORS
    }

    # Persistent stock-level characteristics (slowly drifting alpha drivers + idiosyncratic vol)
    betas = rng.uniform(0.7, 1.35, size=n_symbols)
    base_vols = rng.uniform(0.009, 0.024, size=n_symbols)
    # Latent slow trend / quality state drives realistic modest momentum + value + low-vol predictability
    latent_quality = rng.normal(0.0, 1.0, size=n_symbols)
    latent_trend = rng.normal(0.0, 1.0, size=n_symbols)
    base_shares = rng.uniform(5e7, 5e8, size=n_symbols)

    bar_rows: list[dict] = []
    action_rows: list[dict] = []
    fundamental_rows: list[dict] = []
    membership_rows: list[dict] = []

    action_counter = 1

    # Pre-generate daily adjusted returns with realistic modest cross-sectional alpha structure
    # - Slow-moving momentum trend (AR(1) rho=0.96)
    # - Quality/value drift tied to earnings yield & ROE
    # - Low-volatility anomaly (lower idiosyncratic vol earns slightly higher risk-adjusted drift)
    # - Short-term microstructure noise (1-day noise that reverts slightly, not enough to beat costs)
    adj_closes = np.zeros((n_symbols, n_trading_days), dtype=np.float64)
    adj_closes[:, 0] = rng.uniform(25.0, 180.0, size=n_symbols)

    trend_state = latent_trend.copy()
    quality_state = latent_quality.copy()
    prev_idio = np.zeros(n_symbols, dtype=np.float64)

    for t in range(1, n_trading_days):
        trend_state = 0.965 * trend_state + rng.normal(0.0, 0.26, size=n_symbols)
        quality_state = 0.992 * quality_state + rng.normal(0.0, 0.08, size=n_symbols)
        low_vol_score = -(base_vols - base_vols.mean()) / (base_vols.std() + 1e-12)

        for i, sym in enumerate(symbols):
            sec = symbol_sectors[sym]
            distress = 0.0
            if sym in delist_day_map and t > delist_day_map[sym] - 40:
                distress = -0.0085  # Realistic pre-delisting downward drift

            # Modest predictable drift + realistic daily idiosyncratic noise
            alpha_drift = (
                0.00055 * trend_state[i]
                + 0.00045 * quality_state[i]
                + 0.00030 * low_vol_score[i]
                - 0.04 * prev_idio[i]
                + distress
            )
            idio = rng.normal(0.0, base_vols[i])
            prev_idio[i] = idio
            daily_ret = betas[i] * market_ret[t] + sector_rets[sec][t] + alpha_drift + idio
            daily_ret = float(np.clip(daily_ret, -0.25, 0.25))
            adj_closes[i, t] = max(1.0, adj_closes[i, t - 1] * (1.0 + daily_ret))

    # Build membership, corporate actions, raw vs adjusted bars, and quarterly fundamentals
    for i, sym in enumerate(symbols):
        exit_day = delist_day_map.get(sym)
        last_day = exit_day if exit_day is not None else n_trading_days - 1

        membership_rows.append(
            {
                "symbol": sym,
                "sector": symbol_sectors[sym],
                "entry_ts_utc": dates[0],
                "exit_ts_utc": dates[exit_day] if exit_day is not None else None,
                "exit_reason": "delisted_distress" if exit_day is not None else None,
            }
        )

        if exit_day is not None:
            action_rows.append(
                {
                    "action_id": f"CA-{action_counter:04d}",
                    "symbol": sym,
                    "action_type": "delisting",
                    "ex_date_utc": dates[exit_day],
                    "available_at_utc": dates[exit_day] + bar_avail_delay,
                    "ratio": 1.0,
                    "cash_amount_usd": 0.0,
                    "currency": "USD",
                }
            )
            action_counter += 1

        split_info = split_schedule.get(sym)
        split_day = split_info[0] if split_info is not None else None
        split_ratio = split_info[1] if split_info is not None else 1.0
        split_type = split_info[2] if split_info is not None else None

        if split_info is not None and split_day is not None and split_day <= last_day:
            action_rows.append(
                {
                    "action_id": f"CA-{action_counter:04d}",
                    "symbol": sym,
                    "action_type": split_type,
                    "ex_date_utc": dates[split_day],
                    "available_at_utc": dates[split_day] - timedelta(days=5),  # announced ahead of ex-date
                    "ratio": split_ratio,
                    "cash_amount_usd": 0.0,
                    "currency": "USD",
                }
            )
            action_counter += 1

        # Compute raw prices from adjusted prices:
        # Before a 2-for-1 split (ratio=2.0), raw_close = adj_close * 2.0; on/after split_day, raw_close = adj_close.
        # Thus on split_day, raw_close drops by ~50% while adjusted_close stays smooth!
        for t in range(0, last_day + 1):
            adj_c = float(adj_closes[i, t])
            cum_split_mult = split_ratio if (split_day is not None and t < split_day) else 1.0
            day_split_factor = split_ratio if (split_day is not None and t == split_day) else 1.0

            div_cash = 0.0
            if (i % 4 == 0) and (t > 0) and (t % 63 == 0):
                div_cash = round(adj_c * 0.005, 4)
                action_rows.append(
                    {
                        "action_id": f"CA-{action_counter:04d}",
                        "symbol": sym,
                        "action_type": "dividend",
                        "ex_date_utc": dates[t],
                        "available_at_utc": dates[t] - timedelta(days=10),
                        "ratio": 1.0,
                        "cash_amount_usd": div_cash,
                        "currency": "USD",
                    }
                )
                action_counter += 1

            raw_c = adj_c * cum_split_mult
            intraday_spread = abs(float(rng.normal(0.0, base_vols[i] * 0.45)))
            raw_o = raw_c * (1.0 + float(rng.normal(0.0, base_vols[i] * 0.25)))
            raw_h = max(raw_o, raw_c) * (1.0 + intraday_spread)
            raw_l = max(0.01, min(raw_o, raw_c) * (1.0 - intraday_spread))
            vol = float(rng.lognormal(mean=13.5, sigma=0.35))

            bar_rows.append(
                {
                    "symbol": sym,
                    "event_ts_utc": dates[t],
                    "available_at_utc": dates[t] + bar_avail_delay,
                    "open": round(raw_o, 6),
                    "high": round(raw_h, 6),
                    "low": round(raw_l, 6),
                    "close": round(raw_c, 6),
                    "adjusted_close": round(adj_c, 6),
                    "volume": round(vol, 2),
                    "currency": "USD",
                    "split_factor": day_split_factor,
                    "dividend_cash": div_cash,
                }
            )

        # Generate quarterly fundamental records starting 1 quarter BEFORE dataset start so
        # early trading dates have valid PIT fundamentals without look-ahead!
        quarter_dates = [
            dates[0] - timedelta(days=90),
            *[dates[q] for q in range(0, last_day + 1, 63)],
        ]
        for q_idx, period_end in enumerate(quarter_dates):
            # Reference price at period_end (or day 0 for pre-history quarter)
            ref_t = 0 if q_idx == 0 else min((q_idx - 1) * 63, last_day)
            ref_price = float(adj_closes[i, ref_t])
            mkt_cap = ref_price * float(base_shares[i])

            # Quality state positively links to ROE and earnings yield
            q_score = float(latent_quality[i] + rng.normal(0.0, 0.25))
            earnings_yield = float(np.clip(0.055 + 0.018 * q_score, 0.005, 0.16))
            book_to_mkt = float(np.clip(0.45 + 0.14 * q_score + rng.normal(0.0, 0.05), 0.10, 1.40))
            earnings_ttm = mkt_cap * earnings_yield
            book_val = mkt_cap * book_to_mkt
            roe = float(np.clip(earnings_ttm / book_val, -0.3, 0.6))

            # Explicit filing lag: available_at_utc = period_end + filing_lag_days
            avail_ts = period_end + timedelta(days=filing_lag_days)
            fundamental_rows.append(
                {
                    "symbol": sym,
                    "period_end_utc": period_end,
                    "event_ts_utc": period_end,
                    "available_at_utc": avail_ts,
                    "earnings_ttm_usd": round(earnings_ttm, 2),
                    "book_value_usd": round(book_val, 2),
                    "market_cap_usd": round(mkt_cap, 2),
                    "roe": round(roe, 6),
                    "currency": "USD",
                }
            )

    bars_df = pl.DataFrame(
        bar_rows,
        schema={
            "symbol": pl.String,
            "event_ts_utc": pl.Datetime("us", "UTC"),
            "available_at_utc": pl.Datetime("us", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "adjusted_close": pl.Float64,
            "volume": pl.Float64,
            "currency": pl.String,
            "split_factor": pl.Float64,
            "dividend_cash": pl.Float64,
        },
    ).sort(["event_ts_utc", "symbol"])

    fundamentals_df = pl.DataFrame(
        fundamental_rows,
        schema={
            "symbol": pl.String,
            "period_end_utc": pl.Datetime("us", "UTC"),
            "event_ts_utc": pl.Datetime("us", "UTC"),
            "available_at_utc": pl.Datetime("us", "UTC"),
            "earnings_ttm_usd": pl.Float64,
            "book_value_usd": pl.Float64,
            "market_cap_usd": pl.Float64,
            "roe": pl.Float64,
            "currency": pl.String,
        },
    ).sort(["available_at_utc", "symbol"])

    actions_df = pl.DataFrame(
        action_rows,
        schema={
            "action_id": pl.String,
            "symbol": pl.String,
            "action_type": pl.String,
            "ex_date_utc": pl.Datetime("us", "UTC"),
            "available_at_utc": pl.Datetime("us", "UTC"),
            "ratio": pl.Float64,
            "cash_amount_usd": pl.Float64,
            "currency": pl.String,
        },
    ).sort(["ex_date_utc", "symbol"])

    membership_df = pl.DataFrame(
        membership_rows,
        schema={
            "symbol": pl.String,
            "sector": pl.String,
            "entry_ts_utc": pl.Datetime("us", "UTC"),
            "exit_ts_utc": pl.Datetime("us", "UTC"),
            "exit_reason": pl.String,
        },
    ).sort("symbol")

    manifest = DatasetManifest(
        dataset_id=f"synth_us_equities_pit_seed{seed}",
        generated_at_utc=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        source_type="synthetic",
        entitlements="internal_synthetic_research_only",
        license="proprietary_synthetic_fixture",
        seed=seed,
        n_symbols=n_symbols,
        n_delisted_symbols=n_delisted,
        n_trading_days=n_trading_days,
        start_date_utc=dates[0],
        end_date_utc=dates[-1],
        file_checksums={},
    )
    manifest = manifest.model_copy(update={"manifest_digest": manifest.compute_digest()})

    return SyntheticDatasetBundle(
        bars=bars_df,
        fundamentals=fundamentals_df,
        corporate_actions=actions_df,
        membership=membership_df,
        manifest=manifest,
    )


def save_dataset_fixtures(bundle: SyntheticDatasetBundle, output_dir: Path) -> DatasetManifest:
    """Persist synthetic PIT fixtures as Parquet files + checksummed manifest.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "market_bars.parquet": bundle.bars,
        "fundamentals.parquet": bundle.fundamentals,
        "corporate_actions.parquet": bundle.corporate_actions,
        "universe_membership.parquet": bundle.membership,
    }
    checksums: dict[str, str] = {}
    for fname, df in files.items():
        fpath = output_dir / fname
        df.write_parquet(fpath)
        checksums[fname] = _sha256_file(fpath)

    updated = bundle.manifest.model_copy(update={"file_checksums": checksums})
    updated = updated.model_copy(update={"manifest_digest": updated.compute_digest()})

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(updated.model_dump(mode="json"), fh, indent=2, sort_keys=True)
    return updated
