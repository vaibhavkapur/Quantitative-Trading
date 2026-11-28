"""Cross-asset factor signal engine (momentum, carry, defensive) with strict timestamp gates,
as-of provenance tracking, and training-window-only standardization.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
import hashlib
from typing import Any

import numpy as np
import polars as pl

from cross_asset_factor_engine.config import EngineConfig, SUPPORTED_FACTORS
from cross_asset_factor_engine.contracts import FACTOR_SIGNALS_COLUMNS, MarketDataset
from cross_asset_factor_engine.diagnostics import DiagnosticTracker
from cross_asset_factor_engine.errors import MissingFactorError, TimestampGateError


def parse_decision_as_of_ts(trading_date: date, cutoff_utc_time: str) -> datetime:
    """Combine a trading date and ISO UTC time string (e.g. '21:00:00Z') into a UTC datetime."""
    clean = cutoff_utc_time.rstrip("Z")
    parts = [int(p) for p in clean.split(":")]
    while len(parts) < 3:
        parts.append(0)
    return datetime.combine(trading_date, time(parts[0], parts[1], parts[2]), tzinfo=timezone.utc)


def _provenance_hash(
    signal_date: date,
    as_of_ts: datetime,
    source_max_pub_ts: datetime | None,
    asset_id: str,
    factor_name: str,
    signal_status: str,
) -> str:
    payload = "|".join(
        [
            signal_date.isoformat(),
            as_of_ts.isoformat(),
            source_max_pub_ts.isoformat() if source_max_pub_ts is not None else "NONE",
            asset_id,
            factor_name,
            signal_status,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def compute_raw_factor_signals(
    normalized_returns: pl.DataFrame,
    dataset: MarketDataset,
    config: EngineConfig,
    *,
    diagnostics: DiagnosticTracker | None = None,
) -> pl.DataFrame:
    """Compute point-in-time raw momentum, carry, and defensive signals with strict timestamp gates.

    Every emitted signal row carries:
      - `as_of_ts`: UTC decision cutoff timestamp for `signal_date`
      - `source_max_publication_ts`: latest `publication_ts` among all data rows actually consumed
      - `provenance_hash`: deterministic SHA-256 fingerprint of the signal provenance tuple

    Invariant enforced on every row:
      `source_max_publication_ts <= as_of_ts` (whenever non-null).
    """
    cutoff = config.decision_cutoff_utc_time
    mom_cfg = config.factors.momentum
    carry_cfg = config.factors.carry
    def_cfg = config.factors.defensive
    strict_missing = config.data.strict_missing_factors
    def_fallback = config.data.defensive_fallback_to_low_vol
    ann_days = float(config.walk_forward.annualization_days)

    meta_map: dict[str, dict[str, Any]] = {
        row["asset_id"]: row for row in dataset.contract_metadata.iter_rows(named=True)
    }

    # Index FX rates by currency sorted by publication_ts for point-in-time lookups
    fx_by_ccy: dict[str, list[dict[str, Any]]] = {}
    for row in dataset.fx_rates.sort(["currency", "publication_ts", "date"]).iter_rows(named=True):
        fx_by_ccy.setdefault(row["currency"], []).append(row)

    # Index factor_inputs by asset_id sorted by publication_ts for point-in-time lookups
    fi_by_asset: dict[str, list[dict[str, Any]]] = {}
    for row in dataset.factor_inputs.sort(["asset_id", "publication_ts", "date"]).iter_rows(named=True):
        fi_by_asset.setdefault(row["asset_id"], []).append(row)

    # Group normalized returns by asset_id ordered chronologically
    nr_by_asset: dict[str, list[dict[str, Any]]] = {}
    for row in normalized_returns.sort(["asset_id", "date"]).iter_rows(named=True):
        nr_by_asset.setdefault(row["asset_id"], []).append(row)

    def _latest_published(
        rows: list[dict[str, Any]],
        as_of_ts: datetime,
        required_fields: tuple[str, ...],
        max_staleness_days: int | None,
        signal_date: date,
    ) -> tuple[dict[str, Any] | None, int]:
        """Return latest row with publication_ts <= as_of_ts and non-null required_fields.

        Also returns count of records with date <= signal_date that were blocked by the timestamp gate.
        """
        best: dict[str, Any] | None = None
        blocked = 0
        for r in rows:
            pub_ts: datetime = r["publication_ts"]
            if pub_ts > as_of_ts:
                if r["date"] <= signal_date:
                    blocked += 1
                continue
            if all(r.get(f) is not None and not np.isnan(float(r[f])) for f in required_fields):
                if best is None or pub_ts >= best["publication_ts"]:
                    best = r
        if best is not None and max_staleness_days is not None:
            age_days = (signal_date - best["date"]).days
            if age_days > max_staleness_days:
                return None, blocked
        return best, blocked

    signal_rows: list[dict[str, Any]] = []
    total_blocked_gate = 0

    for asset_id in sorted(meta_map.keys()):
        meta = meta_map[asset_id]
        asset_class = str(meta["asset_class"])
        quote_ccy = str(meta["quote_currency"])
        carry_conv = str(meta["carry_convention"])
        fin_spread = float(meta["financing_spread_bps"]) / 10_000.0
        roll_cost = float(meta["roll_cost_bps"]) / 10_000.0

        asset_nr = nr_by_asset.get(asset_id, [])
        asset_fi = fi_by_asset.get(asset_id, [])
        ccy_fx = fx_by_ccy.get(quote_ccy, [])

        # Chronological history of (date, net_return_before_tc, publication_ts) gated per decision day
        for idx, cur_row in enumerate(asset_nr):
            sig_date: date = cur_row["date"]
            as_of_ts = parse_decision_as_of_ts(sig_date, cutoff)

            # Collect strictly point-in-time visible return history up to idx
            visible_hist: list[dict[str, Any]] = []
            for h_row in asset_nr[: idx + 1]:
                if h_row["publication_ts"] <= as_of_ts:
                    visible_hist.append(h_row)
                else:
                    total_blocked_gate += 1

            valid_ret_rows = [r for r in visible_hist if r["net_return_before_tc"] is not None]
            valid_rets = np.array([float(r["net_return_before_tc"]) for r in valid_ret_rows], dtype=np.float64)

            # ------------------------------------------------------------------
            # 1. MOMENTUM FACTOR (vol-adjusted skip-horizon cumulative return)
            # ------------------------------------------------------------------
            mom_val: float | None = None
            mom_pub_ts: datetime | None = None
            mom_status = "OK"
            mom_reason: str | None = None

            if len(valid_rets) < mom_cfg.min_history_days:
                mom_status = "INSUFFICIENT_HISTORY"
                mom_reason = f"returns_available={len(valid_rets)}_lt_{mom_cfg.min_history_days}"
                if valid_ret_rows:
                    mom_pub_ts = max(r["publication_ts"] for r in valid_ret_rows)
            else:
                end_idx = len(valid_rets) - mom_cfg.skip_days
                start_idx = len(valid_rets) - mom_cfg.lookback_days
                mom_window = valid_rets[start_idx:end_idx]
                vol_window = valid_rets[-mom_cfg.vol_lookback_days :]
                cum_ret = float(np.prod(1.0 + mom_window) - 1.0)
                trailing_vol = float(np.std(vol_window, ddof=1) * np.sqrt(ann_days))
                horizon_scale = float(np.sqrt(len(mom_window) / ann_days))
                mom_val = cum_ret / (max(trailing_vol, 1.0e-6) * max(horizon_scale, 1.0e-6))
                consumed_rows = valid_ret_rows[min(start_idx, len(valid_ret_rows) - mom_cfg.vol_lookback_days) :]
                mom_pub_ts = max(r["publication_ts"] for r in consumed_rows)

            if mom_pub_ts is not None and mom_pub_ts > as_of_ts:
                raise TimestampGateError(
                    "Momentum signal violated timestamp gate invariant.",
                    details={"asset_id": asset_id, "signal_date": sig_date.isoformat()},
                )

            signal_rows.append(
                {
                    "signal_date": sig_date,
                    "as_of_ts": as_of_ts,
                    "source_max_publication_ts": mom_pub_ts,
                    "asset_id": asset_id,
                    "asset_class": asset_class,
                    "factor_name": "momentum",
                    "raw_value": mom_val,
                    "z_score": None,
                    "signal_status": mom_status,
                    "missing_reason": mom_reason,
                    "provenance_hash": _provenance_hash(
                        sig_date, as_of_ts, mom_pub_ts, asset_id, "momentum", mom_status
                    ),
                }
            )

            # ------------------------------------------------------------------
            # 2. CARRY FACTOR (explicit convention per asset class)
            # ------------------------------------------------------------------
            carry_val: float | None = None
            carry_pub_ts: datetime | None = None
            carry_status = "OK"
            carry_reason: str | None = None

            if carry_conv == "dividend_minus_financing":
                fi_rec, blocked = _latest_published(
                    asset_fi, as_of_ts, ("dividend_yield",), carry_cfg.max_staleness_days, sig_date
                )
                total_blocked_gate += blocked
                fx_rec, fx_blocked = _latest_published(
                    ccy_fx, as_of_ts, ("short_rate",), carry_cfg.max_staleness_days, sig_date
                )
                total_blocked_gate += fx_blocked
                if fi_rec is None:
                    carry_status = "MISSING_INPUT"
                    carry_reason = "missing_or_stale_dividend_yield"
                elif fx_rec is None:
                    carry_status = "MISSING_INPUT"
                    carry_reason = f"missing_local_short_rate_{quote_ccy}"
                else:
                    carry_val = float(fi_rec["dividend_yield"]) - float(fx_rec["short_rate"]) - fin_spread
                    carry_pub_ts = max(fi_rec["publication_ts"], fx_rec["publication_ts"])

            elif carry_conv == "interest_rate_differential":
                fi_rec, blocked = _latest_published(
                    asset_fi,
                    as_of_ts,
                    ("foreign_short_rate", "domestic_short_rate"),
                    carry_cfg.max_staleness_days,
                    sig_date,
                )
                total_blocked_gate += blocked
                if fi_rec is None:
                    carry_status = "MISSING_INPUT"
                    carry_reason = "missing_fx_interest_rate_differential"
                else:
                    carry_val = (
                        float(fi_rec["foreign_short_rate"])
                        - float(fi_rec["domestic_short_rate"])
                        - fin_spread
                    )
                    carry_pub_ts = fi_rec["publication_ts"]

            elif carry_conv == "futures_term_structure_roll":
                if cur_row["publication_ts"] > as_of_ts:
                    carry_status = "MISSING_INPUT"
                    carry_reason = "price_blocked_by_timestamp_gate"
                elif (
                    cur_row["implied_roll_yield"] is None
                    or cur_row["next_expiry_days"] is None
                    or cur_row["next_expiry_days"] <= cur_row["front_expiry_days"]
                ):
                    carry_status = "MISSING_INPUT"
                    carry_reason = "missing_next_contract_term_structure"
                else:
                    dt_days = float(cur_row["next_expiry_days"] - cur_row["front_expiry_days"])
                    ann_roll_cost = roll_cost * (carry_cfg.annualization_days / dt_days)
                    carry_val = float(cur_row["implied_roll_yield"]) - ann_roll_cost
                    carry_pub_ts = cur_row["publication_ts"]

            if carry_status == "MISSING_INPUT" and strict_missing:
                raise MissingFactorError(
                    f"Missing carry factor input for {asset_id} on {sig_date}: {carry_reason}",
                    details={"asset_id": asset_id, "signal_date": sig_date.isoformat(), "reason": carry_reason},
                )
            if carry_pub_ts is not None and carry_pub_ts > as_of_ts:
                raise TimestampGateError(
                    "Carry signal violated timestamp gate invariant.",
                    details={"asset_id": asset_id, "signal_date": sig_date.isoformat()},
                )

            signal_rows.append(
                {
                    "signal_date": sig_date,
                    "as_of_ts": as_of_ts,
                    "source_max_publication_ts": carry_pub_ts,
                    "asset_id": asset_id,
                    "asset_class": asset_class,
                    "factor_name": "carry",
                    "raw_value": carry_val,
                    "z_score": None,
                    "signal_status": carry_status,
                    "missing_reason": carry_reason,
                    "provenance_hash": _provenance_hash(
                        sig_date, as_of_ts, carry_pub_ts, asset_id, "carry", carry_status
                    ),
                }
            )

            # ------------------------------------------------------------------
            # 3. DEFENSIVE FACTOR (low-risk / downside-vol + lagged equity quality)
            # ------------------------------------------------------------------
            def_val: float | None = None
            def_pub_ts: datetime | None = None
            def_status = "OK"
            def_reason: str | None = None

            if len(valid_rets) < def_cfg.min_history_days:
                def_status = "INSUFFICIENT_HISTORY"
                def_reason = f"returns_available={len(valid_rets)}_lt_{def_cfg.min_history_days}"
                if valid_ret_rows:
                    def_pub_ts = max(r["publication_ts"] for r in valid_ret_rows)
            else:
                win_len = min(len(valid_rets), def_cfg.vol_lookback_days)
                w_rets = valid_rets[-win_len:]
                w_rows = valid_ret_rows[-win_len:]
                realized_vol = float(np.std(w_rets, ddof=1) * np.sqrt(ann_days))
                downside = np.minimum(w_rets, 0.0)
                downside_vol = float(np.sqrt(np.mean(downside**2)) * np.sqrt(ann_days))
                low_risk = -0.5 * (realized_vol + downside_vol)
                ret_pub_ts = max(r["publication_ts"] for r in w_rows)

                if asset_class == "equity":
                    q_rec, q_blocked = _latest_published(
                        asset_fi,
                        as_of_ts,
                        ("earnings_quality",),
                        def_cfg.max_staleness_days,
                        sig_date,
                    )
                    total_blocked_gate += q_blocked
                    if q_rec is not None:
                        wq = def_cfg.equity_quality_weight
                        def_val = (1.0 - wq) * low_risk + wq * (float(q_rec["earnings_quality"]) * 0.10)
                        def_pub_ts = max(ret_pub_ts, q_rec["publication_ts"])
                    else:
                        if strict_missing:
                            raise MissingFactorError(
                                f"Missing earnings_quality for defensive factor on {asset_id} ({sig_date}).",
                                details={"asset_id": asset_id, "signal_date": sig_date.isoformat()},
                            )
                        if def_fallback:
                            def_val = low_risk
                            def_pub_ts = ret_pub_ts
                            def_status = "OK_FALLBACK_LOW_VOL"
                            def_reason = "missing_earnings_quality_fallback_to_low_vol"
                        else:
                            def_val = None
                            def_pub_ts = ret_pub_ts
                            def_status = "MISSING_INPUT"
                            def_reason = "missing_earnings_quality"
                else:
                    def_val = low_risk
                    def_pub_ts = ret_pub_ts

            if def_pub_ts is not None and def_pub_ts > as_of_ts:
                raise TimestampGateError(
                    "Defensive signal violated timestamp gate invariant.",
                    details={"asset_id": asset_id, "signal_date": sig_date.isoformat()},
                )

            signal_rows.append(
                {
                    "signal_date": sig_date,
                    "as_of_ts": as_of_ts,
                    "source_max_publication_ts": def_pub_ts,
                    "asset_id": asset_id,
                    "asset_class": asset_class,
                    "factor_name": "defensive",
                    "raw_value": def_val,
                    "z_score": None,
                    "signal_status": def_status,
                    "missing_reason": def_reason,
                    "provenance_hash": _provenance_hash(
                        sig_date, as_of_ts, def_pub_ts, asset_id, "defensive", def_status
                    ),
                }
            )

    df = pl.DataFrame(
        signal_rows,
        schema={
            "signal_date": pl.Date,
            "as_of_ts": pl.Datetime("us", "UTC"),
            "source_max_publication_ts": pl.Datetime("us", "UTC"),
            "asset_id": pl.String,
            "asset_class": pl.String,
            "factor_name": pl.String,
            "raw_value": pl.Float64,
            "z_score": pl.Float64,
            "signal_status": pl.String,
            "missing_reason": pl.String,
            "provenance_hash": pl.String,
        },
    ).select(list(FACTOR_SIGNALS_COLUMNS)).sort(["signal_date", "asset_id", "factor_name"])

    if diagnostics is not None:
        n_ok = int(df.filter(pl.col("signal_status").str.starts_with("OK")).height)
        n_missing = int(df.filter(pl.col("signal_status") == "MISSING_INPUT").height)
        n_insuff = int(df.filter(pl.col("signal_status") == "INSUFFICIENT_HISTORY").height)
        diagnostics.inc("signals_computed_ok", n_ok)
        diagnostics.inc("signals_missing_input", n_missing)
        diagnostics.inc("signals_insufficient_history", n_insuff)
        diagnostics.inc("rows_blocked_by_timestamp_gate", total_blocked_gate)
        diagnostics.log(
            "INFO",
            "raw_factors_computed",
            total_signals=df.height,
            ok=n_ok,
            missing_input=n_missing,
            insufficient_history=n_insuff,
            rows_blocked_by_timestamp_gate=total_blocked_gate,
        )

    verify_signal_provenance(df)
    return df


def verify_signal_provenance(signals_df: pl.DataFrame) -> dict[str, Any]:
    """Verify 100% of signals satisfy as-of provenance and no-look-ahead invariants."""
    missing_asof = signals_df.filter(pl.col("as_of_ts").is_null()).height
    if missing_asof > 0:
        raise TimestampGateError(
            f"{missing_asof} signal rows are missing as_of_ts provenance.",
            details={"missing_as_of_count": missing_asof},
        )
    missing_hash = signals_df.filter(
        pl.col("provenance_hash").is_null() | (pl.col("provenance_hash") == "")
    ).height
    if missing_hash > 0:
        raise TimestampGateError(
            f"{missing_hash} signal rows are missing provenance_hash.",
            details={"missing_hash_count": missing_hash},
        )
    lookahead = signals_df.filter(
        pl.col("source_max_publication_ts").is_not_null()
        & (pl.col("source_max_publication_ts") > pl.col("as_of_ts"))
    )
    if lookahead.height > 0:
        raise TimestampGateError(
            f"Detected {lookahead.height} look-ahead violations (source_max_publication_ts > as_of_ts).",
            details={
                "violation_count": lookahead.height,
                "samples": lookahead.head(5).select(
                    ["signal_date", "asset_id", "factor_name", "as_of_ts", "source_max_publication_ts"]
                ).to_dicts(),
            },
        )
    return {
        "total_signals": signals_df.height,
        "lookahead_violations": 0,
        "all_have_as_of_provenance": True,
        "status_counts": {
            row["signal_status"]: int(row["len"])
            for row in signals_df.group_by("signal_status").len().iter_rows(named=True)
        },
    }


@dataclass(frozen=True)
class ScalingParams:
    mean: float
    std: float
    n_obs: int


class FactorScaler:
    """Within-asset-class z-score scaler fit STRICTLY on training-window observations."""

    def __init__(self, *, winsorize_z: float = 3.0, min_std: float = 1.0e-6) -> None:
        self.winsorize_z = winsorize_z
        self.min_std = min_std
        self.params: dict[tuple[str, str], ScalingParams] = {}
        self.train_start: date | None = None
        self.train_end: date | None = None

    def fit(self, train_signals_df: pl.DataFrame) -> "FactorScaler":
        """Fit mean and std per (asset_class, factor_name) using ONLY training window rows."""
        if train_signals_df.height == 0:
            return self
        self.train_start = train_signals_df["signal_date"].min()
        self.train_end = train_signals_df["signal_date"].max()

        valid = train_signals_df.filter(pl.col("raw_value").is_not_null() & ~pl.col("raw_value").is_nan())
        self.params = {}
        for (asset_class, factor_name), grp in valid.group_by(["asset_class", "factor_name"]):
            vals = grp["raw_value"].to_numpy()
            if len(vals) == 0:
                continue
            mu = float(np.mean(vals))
            sigma = float(np.std(vals, ddof=1)) if len(vals) > 1 else 1.0
            if not np.isfinite(sigma) or sigma < self.min_std:
                sigma = 1.0
            self.params[(str(asset_class), str(factor_name))] = ScalingParams(
                mean=mu, std=sigma, n_obs=int(len(vals))
            )
        return self

    def transform(self, signals_df: pl.DataFrame) -> pl.DataFrame:
        """Transform signals using training-fitted parameters without leaking evaluation statistics."""
        if not self.params:
            return signals_df.with_columns(pl.lit(None, dtype=pl.Float64).alias("z_score"))

        param_rows = [
            {
                "asset_class": ac,
                "factor_name": fn,
                "_mu": sp.mean,
                "_sigma": sp.std,
            }
            for (ac, fn), sp in self.params.items()
        ]
        param_df = pl.DataFrame(param_rows)
        transformed = (
            signals_df.drop("z_score")
            .join(param_df, on=["asset_class", "factor_name"], how="left")
            .with_columns(
                pl.when(pl.col("raw_value").is_not_null() & pl.col("_mu").is_not_null())
                .then(
                    ((pl.col("raw_value") - pl.col("_mu")) / pl.col("_sigma")).clip(
                        -self.winsorize_z, self.winsorize_z
                    )
                )
                .otherwise(pl.lit(None, dtype=pl.Float64))
                .alias("z_score")
            )
            .drop(["_mu", "_sigma"])
            .select(list(FACTOR_SIGNALS_COLUMNS))
        )
        return transformed

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_start": self.train_start.isoformat() if self.train_start else None,
            "train_end": self.train_end.isoformat() if self.train_end else None,
            "winsorize_z": self.winsorize_z,
            "params": {
                f"{ac}:{fn}": {"mean": round(sp.mean, 8), "std": round(sp.std, 8), "n_obs": sp.n_obs}
                for (ac, fn), sp in sorted(self.params.items())
            },
        }


def build_blended_signals(
    scaled_signals_df: pl.DataFrame,
    blend_weights: dict[str, float],
) -> pl.DataFrame:
    """Combine scaled single-factor z-scores into a comparable cross-asset blended signal.

    Renormalizes weights over available (non-null) factors on each (signal_date, asset_id)
    and tracks exact as-of provenance (max source_max_publication_ts across components).
    """
    w_df = pl.DataFrame(
        [{"factor_name": fn, "_bw": float(blend_weights[fn])} for fn in SUPPORTED_FACTORS]
    )
    joined = scaled_signals_df.join(w_df, on="factor_name", how="inner")

    blended = (
        joined.group_by(["signal_date", "as_of_ts", "asset_id", "asset_class"])
        .agg(
            pl.col("source_max_publication_ts").max().alias("source_max_publication_ts"),
            (
                pl.when(pl.col("z_score").is_not_null())
                .then(pl.col("z_score") * pl.col("_bw"))
                .otherwise(0.0)
                .sum()
            ).alias("_weighted_sum"),
            (
                pl.when(pl.col("z_score").is_not_null())
                .then(pl.col("_bw"))
                .otherwise(0.0)
                .sum()
            ).alias("_avail_weight"),
            pl.col("z_score").is_not_null().sum().alias("_n_avail"),
        )
        .with_columns(
            pl.lit("blended").alias("factor_name"),
            pl.when(pl.col("_avail_weight") > 0)
            .then(pl.col("_weighted_sum") / pl.col("_avail_weight"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("raw_value"),
            pl.when(pl.col("_avail_weight") > 0)
            .then(pl.col("_weighted_sum") / pl.col("_avail_weight"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("z_score"),
            pl.when(pl.col("_n_avail") == len(SUPPORTED_FACTORS))
            .then(pl.lit("OK"))
            .when(pl.col("_n_avail") > 0)
            .then(pl.lit("OK_PARTIAL_BLEND"))
            .otherwise(pl.lit("MISSING_INPUT"))
            .alias("signal_status"),
            pl.when(pl.col("_n_avail") == len(SUPPORTED_FACTORS))
            .then(pl.lit(None, dtype=pl.String))
            .when(pl.col("_n_avail") > 0)
            .then(pl.lit("partial_component_factors_available"))
            .otherwise(pl.lit("no_component_factors_available"))
            .alias("missing_reason"),
        )
        .sort(["signal_date", "asset_id"])
    )

    hashes = [
        _provenance_hash(
            row["signal_date"],
            row["as_of_ts"],
            row["source_max_publication_ts"],
            row["asset_id"],
            "blended",
            row["signal_status"],
        )
        for row in blended.iter_rows(named=True)
    ]

    out = blended.with_columns(pl.Series("provenance_hash", hashes)).select(list(FACTOR_SIGNALS_COLUMNS))
    verify_signal_provenance(out)
    return out
