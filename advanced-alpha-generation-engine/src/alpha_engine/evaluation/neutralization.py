"""Cross-sectional winsorization, sector/beta/volatility neutralization, and standardization."""

from __future__ import annotations

import numpy as np
import polars as pl

from alpha_engine.config import NeutralizationConfig


class SignalNeutralizer:
    """Applies strictly causal cross-sectional risk neutralization to raw signal panels."""

    def __init__(self, config: NeutralizationConfig) -> None:
        self.config = config

    @staticmethod
    def compute_causal_risk_exposures(
        bars_df: pl.DataFrame,
        *,
        beta_window: int = 60,
        vol_window: int = 20,
    ) -> pl.DataFrame:
        """Compute strictly backward-looking market beta and realized vol per (symbol, event_ts_utc)."""
        ordered = bars_df.sort(["symbol", "event_ts_utc"]).with_columns(
            (
                pl.col("adjusted_close") / pl.col("adjusted_close").shift(1).over("symbol") - 1.0
            ).alias("_ret")
        )
        mkt = (
            ordered.group_by("event_ts_utc")
            .agg(pl.col("_ret").mean().alias("_mkt_ret"))
            .sort("event_ts_utc")
        )
        joined = ordered.join(mkt, on="event_ts_utc", how="left").sort(["symbol", "event_ts_utc"])
        exposures = joined.with_columns(
            pl.col("_ret").rolling_std(window_size=vol_window).over("symbol").alias("_risk_vol"),
            pl.rolling_cov(pl.col("_ret"), pl.col("_mkt_ret"), window_size=beta_window)
            .over("symbol")
            .alias("_cov"),
            pl.col("_mkt_ret").rolling_var(window_size=beta_window).over("symbol").alias("_mvar"),
        ).with_columns(
            pl.when(pl.col("_mvar") > 1e-12)
            .then((pl.col("_cov") / pl.col("_mvar")).clip(-3.0, 4.0))
            .otherwise(1.0)
            .fill_null(1.0)
            .alias("_risk_beta"),
            pl.col("_risk_vol").fill_null(0.015).alias("_risk_vol"),
        )
        cols = ["symbol", "event_ts_utc", "_risk_beta", "_risk_vol"]
        if "sector" in exposures.columns:
            cols.append("sector")
        return exposures.select(cols)

    def neutralize(
        self,
        signal_df: pl.DataFrame,
        bars_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Return signal_df with added 'neutralized_value' column (strictly per-date cross-sectional)."""
        risk_df = self.compute_causal_risk_exposures(bars_df)
        merged = signal_df.join(risk_df, on=["symbol", "event_ts_utc"], how="inner").sort(
            ["event_ts_utc", "symbol"]
        )
        if merged.height == 0:
            return signal_df.with_columns(pl.lit(None, dtype=pl.Float64).alias("neutralized_value"))

        cfg = self.config
        neutralized_vals = np.full(merged.height, np.nan, dtype=np.float64)

        raw_arr = merged["raw_value"].to_numpy()
        beta_arr = merged["_risk_beta"].to_numpy()
        vol_arr = merged["_risk_vol"].to_numpy()
        sectors = merged["sector"].to_list() if "sector" in merged.columns else ["ALL"] * merged.height
        dates = merged["event_ts_utc"].to_list()

        # Process contiguous date blocks (already sorted by event_ts_utc, symbol)
        start = 0
        n = merged.height
        while start < n:
            end = start + 1
            cur_date = dates[start]
            while end < n and dates[end] == cur_date:
                end += 1

            y = raw_arr[start:end].astype(np.float64).copy()
            valid = np.isfinite(y)
            if valid.sum() >= 4:
                yv = y[valid]
                # 1. Winsorize
                lo = float(np.quantile(yv, cfg.winsorize_lower_quantile))
                hi = float(np.quantile(yv, cfg.winsorize_upper_quantile))
                yv = np.clip(yv, lo, hi)

                # 2. Build cross-sectional exposure design matrix
                design_cols: list[np.ndarray] = []
                if cfg.sector_neutral and "sector" in merged.columns:
                    sec_slice = [sectors[idx] for idx in range(start, end) if valid[idx - start]]
                    uniq_sec = sorted(set(sec_slice))
                    for sec in uniq_sec:
                        design_cols.append(
                            np.array([1.0 if s == sec else 0.0 for s in sec_slice], dtype=np.float64)
                        )
                else:
                    design_cols.append(np.ones(int(valid.sum()), dtype=np.float64))

                if cfg.beta_neutral:
                    bv = beta_arr[start:end][valid]
                    if np.std(bv) > 1e-12:
                        design_cols.append((bv - np.mean(bv)) / np.std(bv))

                if cfg.vol_neutral:
                    vv = vol_arr[start:end][valid]
                    if np.std(vv) > 1e-12:
                        design_cols.append((vv - np.mean(vv)) / np.std(vv))

                X = np.column_stack(design_cols)
                if X.shape[0] > X.shape[1]:
                    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
                    resid = yv - X @ coef
                else:
                    resid = yv - np.mean(yv)

                # 3. Standardize to zero mean, unit std
                if cfg.zscore_standardize:
                    sd = float(np.std(resid))
                    if sd > 1e-12:
                        resid = (resid - float(np.mean(resid))) / sd
                    else:
                        resid = np.zeros_like(resid)

                out_block = np.full(end - start, np.nan, dtype=np.float64)
                out_block[valid] = resid
                neutralized_vals[start:end] = out_block

            start = end

        result = merged.with_columns(pl.Series("neutralized_value", neutralized_vals)).drop(
            ["_risk_beta", "_risk_vol"]
        )
        return result.filter(pl.col("neutralized_value").is_finite())
