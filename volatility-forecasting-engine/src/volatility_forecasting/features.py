"""Causal realized-volatility estimation, feature engineering, and horizon label construction."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import polars as pl

from volatility_forecasting.contracts import (
    validate_intraday_bars_df,
    validate_labels_df,
    validate_returns_df,
)
from volatility_forecasting.errors import DataContractError

CAUSAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "rv_1d",
    "rv_5d",
    "rv_22d",
    "vol_1d",
    "vol_5d",
    "vol_22d",
    "log_rv_1d",
    "log_rv_5d",
    "log_rv_22d",
    "neg_return_1d",
    "leverage_5d",
    "abs_return_1d",
    "overnight_gap_sq",
    "intraday_bar_coverage",
    "parkinson_rv_1d",
    "rv_ratio_5_22",
)

_PARKINSON_DENOM = 4.0 * math.log(2.0)


@dataclass(frozen=True)
class FeatureBuildDiagnostics:
    n_rows: int
    n_assets: int
    n_sessions_with_intraday: int
    n_sessions_fallback_daily: int
    n_sessions_low_coverage: int
    n_sessions_with_overnight_gap: int
    min_bar_coverage: float


def compute_daily_realized_variance(
    returns_df: pl.DataFrame,
    intraday_df: pl.DataFrame | None = None,
    *,
    expected_bars_per_session: int = 78,
    include_overnight_gap: bool = True,
    min_bar_coverage_warn: float = 0.80,
) -> tuple[pl.DataFrame, FeatureBuildDiagnostics]:
    """Compute causal daily realized variance per (asset_id, session_date).

    When intraday bars are provided:
      - Intraday RV is the sum of squared intraday log returns across present bars.
        The first bar contributes log(close_0 / open_0)^2; subsequent present bars
        contribute log(close_k / close_{k-1})^2 so interior missing bars telescope
        preserving return continuity while bar coverage records completeness.
      - Parkinson range RV is computed as sum(log(high / low)^2) / (4 * ln(2)).
      - When `include_overnight_gap` is True, the overnight squared return is added
        to intraday RV (using explicit `overnight_return` if non-zero, else inferred
        from log(first_open / prev_session_last_close) when available).
      - Sessions with zero present intraday bars fall back cleanly to daily close-to-close
        squared log return (`log_return^2`) without double-counting overnight return.
    """
    ret = validate_returns_df(returns_df)
    has_explicit_overnight = "overnight_return" in returns_df.columns

    intraday_stats: dict[tuple[str, object], tuple[float, float, int, float | None, float | None]] = {}
    if intraday_df is not None:
        bars = validate_intraday_bars_df(intraday_df)
        present_bars = bars.filter(~pl.col("is_missing"))
        if present_bars.height > 0:
            for (asset_id, sess_date), grp in present_bars.group_by(
                ["asset_id", "session_date"], maintain_order=True
            ):
                grp_sorted = grp.sort("timestamp")
                opens = grp_sorted["open"].to_numpy()
                highs = grp_sorted["high"].to_numpy()
                lows = grp_sorted["low"].to_numpy()
                closes = grp_sorted["close"].to_numpy()
                n_bars = len(closes)

                r0 = math.log(closes[0] / opens[0])
                if n_bars > 1:
                    r_rest = np.log(closes[1:] / closes[:-1])
                    intraday_rv = float(r0 * r0 + np.sum(r_rest * r_rest))
                else:
                    intraday_rv = float(r0 * r0)

                hl_log = np.log(highs / lows)
                parkinson_rv = float(np.sum(hl_log * hl_log) / _PARKINSON_DENOM)
                intraday_stats[(str(asset_id), sess_date)] = (
                    max(0.0, intraday_rv),
                    max(0.0, parkinson_rv),
                    n_bars,
                    float(opens[0]),
                    float(closes[-1]),
                )

    rows: list[dict[str, object]] = []
    n_with_intraday = 0
    n_fallback_daily = 0
    n_low_coverage = 0
    n_with_gap = 0
    min_cov = 1.0

    for (asset_id,), grp in ret.group_by(["asset_id"], maintain_order=True):
        grp_sorted = grp.sort("timestamp")
        prev_last_close: float | None = None
        for rec in grp_sorted.iter_rows(named=True):
            sess_date = rec["session_date"]
            log_ret = float(rec["log_return"])
            explicit_overnight = float(rec["overnight_return"])
            key = (str(asset_id), sess_date)

            if key in intraday_stats:
                intraday_rv, parkinson_rv, n_bars, first_open, last_close = intraday_stats[key]
                coverage = min(1.0, float(n_bars) / float(expected_bars_per_session))
                n_with_intraday += 1
                used_fallback = False

                if has_explicit_overnight and explicit_overnight != 0.0:
                    overnight_ret = explicit_overnight
                elif prev_last_close is not None and first_open is not None and prev_last_close > 0:
                    overnight_ret = math.log(first_open / prev_last_close)
                else:
                    overnight_ret = explicit_overnight

                overnight_sq = float(overnight_ret * overnight_ret)
                daily_rv = intraday_rv + (overnight_sq if include_overnight_gap else 0.0)
                prev_last_close = last_close
            else:
                # Fallback to daily close-to-close squared return (already includes overnight gap)
                n_fallback_daily += 1
                used_fallback = True
                coverage = 0.0 if intraday_df is not None else 1.0
                overnight_ret = explicit_overnight
                overnight_sq = float(overnight_ret * overnight_ret)
                intraday_rv = float(log_ret * log_ret)
                parkinson_rv = intraday_rv
                daily_rv = intraday_rv
                prev_last_close = None

            if coverage < min_bar_coverage_warn:
                n_low_coverage += 1
            if overnight_sq > 1e-12:
                n_with_gap += 1
            min_cov = min(min_cov, coverage)

            rows.append(
                {
                    "asset_id": str(asset_id),
                    "timestamp": rec["timestamp"],
                    "session_date": sess_date,
                    "log_return": log_ret,
                    "overnight_return": overnight_ret,
                    "overnight_gap_sq": overnight_sq,
                    "intraday_rv": max(0.0, intraday_rv),
                    "parkinson_rv": max(0.0, parkinson_rv),
                    "daily_rv": max(0.0, daily_rv),
                    "intraday_bar_coverage": coverage,
                    "used_fallback_daily_rv": used_fallback,
                }
            )

    out_df = pl.DataFrame(rows).sort(["asset_id", "timestamp"])
    diag = FeatureBuildDiagnostics(
        n_rows=out_df.height,
        n_assets=out_df["asset_id"].n_unique(),
        n_sessions_with_intraday=n_with_intraday,
        n_sessions_fallback_daily=n_fallback_daily,
        n_sessions_low_coverage=n_low_coverage,
        n_sessions_with_overnight_gap=n_with_gap,
        min_bar_coverage=min_cov,
    )
    return out_df, diag


def build_causal_dataset(
    returns_df: pl.DataFrame,
    intraday_df: pl.DataFrame | None = None,
    labels_df: pl.DataFrame | None = None,
    *,
    horizons: tuple[int, ...] = (1, 5, 22),
    annualization_factor: int = 252,
    expected_bars_per_session: int = 78,
    include_overnight_gap: bool = True,
    min_bar_coverage_warn: float = 0.80,
    variance_floor: float = 1e-10,
) -> tuple[pl.DataFrame, FeatureBuildDiagnostics]:
    """Build strictly causal features (using <= t) and forward h-day realized-vol labels (t+1..t+h).

    Returns a Polars DataFrame sorted by (asset_id, timestamp) containing:
      - Canonical identifiers & daily realized variance fields
      - All columns in `CAUSAL_FEATURE_COLUMNS`
      - For each h in `horizons`:
          `target_rv_h{h}`: mean realized variance over (t+1 .. t+h)
          `target_vol_h{h}`: sqrt(annualization_factor * target_rv_h{h})
          `label_start_ts_h{h}`: timestamp of t+1
          `label_end_ts_h{h}`: timestamp of t+h
    """
    if not horizons or any(h <= 0 for h in horizons):
        raise DataContractError(f"horizons must contain positive integers, got {horizons}.")

    daily_df, diag = compute_daily_realized_variance(
        returns_df,
        intraday_df,
        expected_bars_per_session=expected_bars_per_session,
        include_overnight_gap=include_overnight_gap,
        min_bar_coverage_warn=min_bar_coverage_warn,
    )

    validated_labels = validate_labels_df(labels_df) if labels_df is not None else None
    ann = float(annualization_factor)
    floor = float(variance_floor)

    asset_frames: list[pl.DataFrame] = []
    for (asset_id,), grp in daily_df.group_by(["asset_id"], maintain_order=True):
        grp = grp.sort("timestamp")
        n = grp.height
        rv = grp["daily_rv"].to_numpy().astype(np.float64)
        ret = grp["log_return"].to_numpy().astype(np.float64)
        ts_list = grp["timestamp"].to_list()

        # Strictly causal trailing rolling means (using <= t only)
        rv_1d = rv.copy()
        rv_5d = np.empty(n, dtype=np.float64)
        rv_22d = np.empty(n, dtype=np.float64)
        neg_ret_sq = np.where(ret < 0.0, ret * ret, 0.0)
        leverage_5d = np.empty(n, dtype=np.float64)

        csum_rv = np.concatenate([[0.0], np.cumsum(rv)])
        csum_neg = np.concatenate([[0.0], np.cumsum(neg_ret_sq)])
        for i in range(n):
            s5 = max(0, i - 4)
            s22 = max(0, i - 21)
            rv_5d[i] = (csum_rv[i + 1] - csum_rv[s5]) / float(i - s5 + 1)
            rv_22d[i] = (csum_rv[i + 1] - csum_rv[s22]) / float(i - s22 + 1)
            leverage_5d[i] = (csum_neg[i + 1] - csum_neg[s5]) / float(i - s5 + 1)

        vol_1d = np.sqrt(ann * np.maximum(rv_1d, 0.0))
        vol_5d = np.sqrt(ann * np.maximum(rv_5d, 0.0))
        vol_22d = np.sqrt(ann * np.maximum(rv_22d, 0.0))

        log_rv_1d = np.log(np.maximum(rv_1d, floor))
        log_rv_5d = np.log(np.maximum(rv_5d, floor))
        log_rv_22d = np.log(np.maximum(rv_22d, floor))

        neg_return_1d = np.maximum(-ret, 0.0)
        abs_return_1d = np.abs(ret)
        rv_ratio_5_22 = rv_5d / np.maximum(rv_22d, floor)

        extra_cols: list[pl.Series] = [
            pl.Series("rv_1d", rv_1d),
            pl.Series("rv_5d", rv_5d),
            pl.Series("rv_22d", rv_22d),
            pl.Series("vol_1d", vol_1d),
            pl.Series("vol_5d", vol_5d),
            pl.Series("vol_22d", vol_22d),
            pl.Series("log_rv_1d", log_rv_1d),
            pl.Series("log_rv_5d", log_rv_5d),
            pl.Series("log_rv_22d", log_rv_22d),
            pl.Series("neg_return_1d", neg_return_1d),
            pl.Series("leverage_5d", leverage_5d),
            pl.Series("abs_return_1d", abs_return_1d),
            pl.Series("parkinson_rv_1d", grp["parkinson_rv"].to_numpy()),
            pl.Series("rv_ratio_5_22", rv_ratio_5_22),
        ]

        # Strictly forward h-day target realized variance over (t+1 .. t+h)
        for h in horizons:
            target_rv: list[float | None] = [None] * n
            target_vol: list[float | None] = [None] * n
            lbl_start: list[object | None] = [None] * n
            lbl_end: list[object | None] = [None] * n

            for i in range(n - h):
                fwd_mean = float((csum_rv[i + h + 1] - csum_rv[i + 1]) / float(h))
                fwd_mean = max(0.0, fwd_mean)
                target_rv[i] = fwd_mean
                target_vol[i] = math.sqrt(ann * fwd_mean)
                lbl_start[i] = ts_list[i + 1]
                lbl_end[i] = ts_list[i + h]

            extra_cols.extend(
                [
                    pl.Series(f"target_rv_h{h}", target_rv, dtype=pl.Float64),
                    pl.Series(f"target_vol_h{h}", target_vol, dtype=pl.Float64),
                    pl.Series(f"label_start_ts_h{h}", lbl_start, dtype=pl.Datetime("us")),
                    pl.Series(f"label_end_ts_h{h}", lbl_end, dtype=pl.Datetime("us")),
                ]
            )

        enriched = grp.with_columns(extra_cols)

        # Optional precomputed labels override (strictly aligned on asset_id, timestamp, horizon_days)
        if validated_labels is not None:
            for h in horizons:
                lbl_h = (
                    validated_labels.filter(
                        (pl.col("asset_id") == str(asset_id)) & (pl.col("horizon_days") == h)
                    )
                    .select(
                        [
                            pl.col("timestamp"),
                            pl.col("realized_variance").alias(f"_ext_rv_h{h}"),
                            pl.col("realized_vol").alias(f"_ext_vol_h{h}"),
                        ]
                    )
                )
                if lbl_h.height > 0:
                    enriched = enriched.join(lbl_h, on="timestamp", how="left").with_columns(
                        pl.coalesce([pl.col(f"_ext_rv_h{h}"), pl.col(f"target_rv_h{h}")]).alias(
                            f"target_rv_h{h}"
                        ),
                        pl.coalesce([pl.col(f"_ext_vol_h{h}"), pl.col(f"target_vol_h{h}")]).alias(
                            f"target_vol_h{h}"
                        ),
                    ).drop([f"_ext_rv_h{h}", f"_ext_vol_h{h}"])

        asset_frames.append(enriched)

    combined = pl.concat(asset_frames, how="vertical").sort(["asset_id", "timestamp"])
    return combined, diag
