"""Deterministic synthetic and hand-calculated reference fixture generators.

All generated fixtures are synthetic educational/research data with fixed seeds.
No real market feeds, customer records, credentials, or proprietary artifacts.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
import math
from pathlib import Path

import numpy as np
import polars as pl


def _trading_dates(start: date, n_days: int) -> list[date]:
    days: list[date] = []
    cur = start
    while len(days) < n_days:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days


def generate_multi_regime_fixture(
    *,
    n_days: int = 420,
    bars_per_day: int = 13,  # 30-min bars across 6.5h session (compact & realistic)
    seed: int = 42,
    asset_id: str = "SYN_EQ_01",
    include_missing_bars: bool = True,
    include_overnight_gaps: bool = True,
    shock_day_index: int | None = 310,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Generate a deterministic multi-regime GARCH + volatility-shock dataset with intraday bars.

    Includes:
      - Persistent GARCH-like variance dynamics with leverage effect
      - Calm, elevated, and explicit volatility shock regimes
      - Controlled overnight gap returns
      - Controlled missing intraday bars (~2% of sessions with partial bars, 1 session fully missing)
    """
    rng = np.random.default_rng(seed)
    dates = _trading_dates(date(2023, 1, 3), n_days)

    # Underlying latent daily variance process with regime shifts & shock
    omega = 4.5e-6
    alpha = 0.11
    beta = 0.84
    gamma_lev = 0.05  # leverage effect for negative shocks

    sig2 = np.empty(n_days, dtype=np.float64)
    sig2[0] = 1.0e-4  # ~15.8% annualized vol

    returns_rows: list[dict[str, object]] = []
    intraday_rows: list[dict[str, object]] = []
    price = 100.0

    for d_idx, sess_date in enumerate(dates):
        # Deterministic regime multiplier
        regime_mult = 1.0
        if 120 <= d_idx < 175:
            regime_mult = 1.8  # elevated volatility regime
        elif shock_day_index is not None and shock_day_index <= d_idx < shock_day_index + 15:
            # Sharp 5x variance shock followed by decay
            decay_k = d_idx - shock_day_index
            regime_mult = 1.0 + 4.5 * math.exp(-decay_k / 5.0)

        daily_var_target = sig2[d_idx] * regime_mult
        # Split variance: 20% overnight, 80% intraday
        overnight_std = math.sqrt(0.20 * daily_var_target) if include_overnight_gaps else 0.0
        intraday_bar_std = math.sqrt((0.80 if include_overnight_gaps else 1.0) * daily_var_target / bars_per_day)

        overnight_ret = float(rng.normal(0.0, overnight_std)) if overnight_std > 0.0 else 0.0
        # Inject an explicit large overnight earnings-like gap on day 65
        if include_overnight_gaps and d_idx == 65:
            overnight_ret = -0.032

        open_price = price * math.exp(overnight_ret)
        bar_cur_price = open_price
        session_intraday_ret = 0.0

        # Deterministic missing bar simulation
        skip_whole_session = include_missing_bars and (d_idx == 90)
        partial_missing_session = include_missing_bars and (d_idx in {45, 140, 235, 340})

        if not skip_whole_session:
            for b in range(bars_per_day):
                bar_ts = datetime.combine(sess_date, time(9, 30)) + timedelta(minutes=30 * (b + 1))
                bar_ret = float(rng.normal(0.0, intraday_bar_std))
                bar_open = bar_cur_price
                bar_close = bar_open * math.exp(bar_ret)
                # Realistic intrabar range around [min(open, close), max(open, close)]
                spread = abs(float(rng.normal(0.0, 0.35 * intraday_bar_std)))
                bar_high = max(bar_open, bar_close) * math.exp(spread)
                bar_low = min(bar_open, bar_close) * math.exp(-spread)
                bar_vol = float(round(10000.0 * (1.0 + 2.0 * abs(bar_ret) / max(intraday_bar_std, 1e-6)), 1))

                bar_cur_price = bar_close
                session_intraday_ret += bar_ret

                # Omit 3 middle bars on partial-missing sessions
                if partial_missing_session and b in {4, 5, 6}:
                    continue

                intraday_rows.append(
                    {
                        "asset_id": asset_id,
                        "timestamp": bar_ts.isoformat(),
                        "session_date": sess_date.isoformat(),
                        "open": round(bar_open, 6),
                        "high": round(bar_high, 6),
                        "low": round(bar_low, 6),
                        "close": round(bar_close, 6),
                        "volume": bar_vol,
                        "is_missing": False,
                    }
                )
        else:
            # Session with no intraday bars: still evolve close price causally
            session_intraday_ret = float(rng.normal(0.0, math.sqrt(0.80 * daily_var_target)))
            bar_cur_price = open_price * math.exp(session_intraday_ret)

        total_log_ret = overnight_ret + session_intraday_ret
        close_price = price * math.exp(total_log_ret)
        price = close_price

        returns_rows.append(
            {
                "asset_id": asset_id,
                "timestamp": datetime.combine(sess_date, time(16, 0)).isoformat(),
                "session_date": sess_date.isoformat(),
                "log_return": round(total_log_ret, 8),
                "overnight_return": round(overnight_ret, 8),
                "close_price": round(close_price, 6),
            }
        )

        if d_idx + 1 < n_days:
            lev_term = gamma_lev * (total_log_ret**2) if total_log_ret < 0.0 else 0.0
            sig2[d_idx + 1] = max(
                1e-6,
                omega + alpha * (total_log_ret**2) + lev_term + beta * sig2[d_idx],
            )

    return pl.DataFrame(returns_rows), pl.DataFrame(intraday_rows)


def generate_hand_calculated_reference_fixture() -> tuple[pl.DataFrame, pl.DataFrame, dict[str, object]]:
    """Build a small deterministic fixture with exact hand-calculable expected outputs.

    Design (6 trading days, 2 intraday bars per day, asset='REF_01'):
      Day 0: overnight=0.0, bar returns [+0.01, -0.01] -> intraday_rv = 2e-4, daily_rv = 2e-4
      Day 1: overnight=+0.02, bar returns [+0.01, +0.01] -> intraday_rv = 2e-4, gap_sq = 4e-4, daily_rv = 6e-4
      Day 2: overnight=0.0, bar returns [0.0, 0.0] -> intraday_rv = 0.0, daily_rv = 0.0
      Day 3: overnight=0.0, bar returns [+0.02, -0.02] -> intraday_rv = 8e-4, daily_rv = 8e-4
      Day 4: overnight=-0.01, bar returns [+0.01, -0.01] -> intraday_rv = 2e-4, gap_sq = 1e-4, daily_rv = 3e-4
      Day 5: overnight=0.0, bar returns [+0.03, 0.0] -> intraday_rv = 9e-4, daily_rv = 9e-4

    Expected hand-calculated values:
      - daily_rv = [2e-4, 6e-4, 0.0, 8e-4, 3e-4, 9e-4]
      - target_rv_h1 (days 0..4) = [6e-4, 0.0, 8e-4, 3e-4, 9e-4], day 5 = None
      - target_rv_h2 (days 0..3) = [3e-4, 4e-4, 5.5e-4, 6e-4], days 4..5 = None
      - rv_5d at day 4 = mean(2e-4, 6e-4, 0.0, 8e-4, 3e-4) = 3.8e-4
    """
    spec = [
        (0.0, [0.01, -0.01]),
        (0.02, [0.01, 0.01]),
        (0.0, [0.0, 0.0]),
        (0.0, [0.02, -0.02]),
        (-0.01, [0.01, -0.01]),
        (0.0, [0.03, 0.0]),
    ]
    dates = _trading_dates(date(2024, 1, 2), len(spec))
    price = 100.0
    ret_rows = []
    bar_rows = []

    for sess_date, (ov, bar_rets) in zip(dates, spec):
        open_p = price * math.exp(ov)
        cur = open_p
        for idx, br in enumerate(bar_rets):
            nxt = cur * math.exp(br)
            bar_ts = datetime.combine(sess_date, time(10, 0)) + timedelta(hours=idx + 1)
            bar_rows.append(
                {
                    "asset_id": "REF_01",
                    "timestamp": bar_ts.isoformat(),
                    "session_date": sess_date.isoformat(),
                    "open": cur,
                    "high": max(cur, nxt),
                    "low": min(cur, nxt),
                    "close": nxt,
                    "volume": 1000.0,
                    "is_missing": False,
                }
            )
            cur = nxt
        total_r = ov + sum(bar_rets)
        price = price * math.exp(total_r)
        ret_rows.append(
            {
                "asset_id": "REF_01",
                "timestamp": datetime.combine(sess_date, time(16, 0)).isoformat(),
                "session_date": sess_date.isoformat(),
                "log_return": total_r,
                "overnight_return": ov,
                "close_price": price,
            }
        )

    expected = {
        "daily_rv": [2e-4, 6e-4, 0.0, 8e-4, 3e-4, 9e-4],
        "target_rv_h1": [6e-4, 0.0, 8e-4, 3e-4, 9e-4, None],
        "target_rv_h2": [3e-4, 4e-4, 5.5e-4, 6e-4, None, None],
        "rv_5d_day4": 3.8e-4,
    }
    return pl.DataFrame(ret_rows), pl.DataFrame(bar_rows), expected


def generate_constant_returns_fixture(n_days: int = 80, constant_return: float = 0.0) -> pl.DataFrame:
    """Deterministic fixture with constant returns (zero variance edge case)."""
    dates = _trading_dates(date(2024, 1, 2), n_days)
    rows = [
        {
            "asset_id": "CONST_01",
            "timestamp": datetime.combine(d, time(16, 0)).isoformat(),
            "session_date": d.isoformat(),
            "log_return": constant_return,
            "overnight_return": 0.0,
        }
        for d in dates
    ]
    return pl.DataFrame(rows)


def write_all_fixtures(fixtures_dir: str | Path) -> dict[str, str]:
    """Write deterministic valid, reference, edge-case, and invalid fixtures to disk."""
    out = Path(fixtures_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    # 1. Main multi-regime benchmark fixture (13 bars/day => 5,460 rows, fast & deterministic)
    mr_ret, mr_bars = generate_multi_regime_fixture(n_days=420, bars_per_day=13, seed=42)
    p_mr_ret = out / "synthetic_multi_regime_returns.csv"
    p_mr_bars = out / "synthetic_multi_regime_intraday.csv"
    mr_ret.write_csv(p_mr_ret)
    mr_bars.write_csv(p_mr_bars)
    written["multi_regime_returns"] = str(p_mr_ret)
    written["multi_regime_intraday"] = str(p_mr_bars)

    # 2. Hand-calculated reference fixture
    ref_ret, ref_bars, _ = generate_hand_calculated_reference_fixture()
    p_ref_ret = out / "reference_hand_calc_returns.csv"
    p_ref_bars = out / "reference_hand_calc_intraday.csv"
    ref_ret.write_csv(p_ref_ret)
    ref_bars.write_csv(p_ref_bars)
    written["reference_returns"] = str(p_ref_ret)
    written["reference_intraday"] = str(p_ref_bars)

    # 3. Constant returns boundary fixture
    const_ret = generate_constant_returns_fixture(n_days=80, constant_return=0.0)
    p_const = out / "boundary_constant_returns.csv"
    const_ret.write_csv(p_const)
    written["constant_returns"] = str(p_const)

    # 4. Invalid fixture (missing required column & NaN value)
    p_invalid_missing = out / "invalid_missing_column_returns.csv"
    pl.DataFrame({"asset_id": ["BAD_01"], "timestamp": ["2024-01-02T16:00:00"]}).write_csv(
        p_invalid_missing
    )
    written["invalid_missing_column"] = str(p_invalid_missing)

    p_invalid_ohlc = out / "invalid_ohlc_intraday.csv"
    pl.DataFrame(
        {
            "asset_id": ["BAD_01"],
            "timestamp": ["2024-01-02T10:00:00"],
            "session_date": ["2024-01-02"],
            "open": [100.0],
            "high": [95.0],  # Violates high >= low and high >= open
            "low": [98.0],
            "close": [99.0],
            "volume": [100.0],
        }
    ).write_csv(p_invalid_ohlc)
    written["invalid_ohlc"] = str(p_invalid_ohlc)

    return written
