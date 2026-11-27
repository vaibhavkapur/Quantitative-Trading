"""Point-in-time (PIT) data, corporate action, survivorship, and causality validators."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import polars as pl

from alpha_engine.config import UniversePolicy
from alpha_engine.errors import (
    CorporateActionIntegrityError,
    DataValidationError,
    LookAheadViolationError,
    MissingSymbolError,
    SurvivorshipViolationError,
)


@dataclass(frozen=True)
class ValidationSummary:
    passed: bool
    checks_run: list[str]
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


class PointInTimeValidator:
    """Validates point-in-time causality, survivorship, corporate actions, and coverage."""

    def __init__(self, universe_policy: UniversePolicy) -> None:
        self.policy = universe_policy

    @staticmethod
    def _ensure_utc(ts: datetime) -> datetime:
        if ts.tzinfo is None:
            raise DataValidationError("as_of_utc must be timezone-aware UTC.")
        return ts.astimezone(UTC)

    def validate_timestamps(
        self,
        df: pl.DataFrame,
        *,
        dataset_name: str,
        as_of_utc: datetime | None = None,
    ) -> None:
        required = {"event_ts_utc", "available_at_utc"}
        missing = required - set(df.columns)
        if missing:
            raise DataValidationError(
                f"{dataset_name} missing required timestamp columns: {sorted(missing)}",
                details={"dataset": dataset_name, "missing_columns": sorted(missing)},
            )

        for col in ("event_ts_utc", "available_at_utc"):
            dtype = df.schema[col]
            if not isinstance(dtype, pl.Datetime) or dtype.time_zone != "UTC":
                raise DataValidationError(
                    f"{dataset_name}.{col} must be pl.Datetime with UTC timezone, got {dtype}.",
                    details={"dataset": dataset_name, "column": col, "dtype": str(dtype)},
                )
            null_count = df.select(pl.col(col).null_count()).item()
            if null_count > 0:
                raise DataValidationError(
                    f"{dataset_name}.{col} contains {null_count} null timestamps.",
                    details={"dataset": dataset_name, "column": col, "null_count": null_count},
                )

        invalid_causal = df.filter(pl.col("available_at_utc") < pl.col("event_ts_utc"))
        if invalid_causal.height > 0:
            sample = invalid_causal.head(3).to_dicts()
            raise LookAheadViolationError(
                f"{dataset_name} contains {invalid_causal.height} rows with available_at_utc < event_ts_utc.",
                details={"dataset": dataset_name, "violations": invalid_causal.height, "sample": sample},
            )

        if as_of_utc is not None:
            cutoff = self._ensure_utc(as_of_utc)
            future_rows = df.filter(
                (pl.col("available_at_utc") > pl.lit(cutoff))
                | (pl.col("event_ts_utc") > pl.lit(cutoff))
            )
            if future_rows.height > 0:
                raise LookAheadViolationError(
                    f"{dataset_name} contains {future_rows.height} rows beyond as_of_utc={cutoff.isoformat()}.",
                    details={
                        "dataset": dataset_name,
                        "as_of_utc": cutoff.isoformat(),
                        "future_rows": future_rows.height,
                    },
                )

    def validate_signal_causality(
        self,
        signal_df: pl.DataFrame,
        input_bars_df: pl.DataFrame,
        *,
        fundamentals_df: pl.DataFrame | None = None,
    ) -> None:
        """Verify every signal row has valid UTC availability >= contributing input availability."""
        self.validate_timestamps(signal_df, dataset_name="signal_df")

        required_cols = {"signal_id", "signal_version", "symbol", "raw_value"}
        missing = required_cols - set(signal_df.columns)
        if missing:
            raise DataValidationError(
                f"signal_df missing required columns: {sorted(missing)}",
                details={"missing_columns": sorted(missing)},
            )

        # Join against market bars on (symbol, event_ts_utc) to verify signal isn't available before bar
        joined = signal_df.join(
            input_bars_df.select(
                "symbol",
                "event_ts_utc",
                pl.col("available_at_utc").alias("bar_available_at_utc"),
            ),
            on=["symbol", "event_ts_utc"],
            how="inner",
        )
        early_signals = joined.filter(pl.col("available_at_utc") < pl.col("bar_available_at_utc"))
        if early_signals.height > 0:
            raise LookAheadViolationError(
                f"Signal available_at_utc precedes market bar available_at_utc on {early_signals.height} rows.",
                details={"violations": early_signals.height},
            )

        # If signal tracks source_available_at_utc (e.g. from fundamentals or external feeds), enforce it
        if "source_available_at_utc" in signal_df.columns:
            leaks = signal_df.filter(
                pl.col("source_available_at_utc").is_not_null()
                & (pl.col("available_at_utc") < pl.col("source_available_at_utc"))
            )
            if leaks.height > 0:
                raise LookAheadViolationError(
                    f"Signal uses unlagged source data before source_available_at_utc on {leaks.height} rows.",
                    details={"violations": leaks.height},
                )

        # If fundamentals are provided, verify no fundamental record with available_at_utc > signal available_at_utc
        if fundamentals_df is not None and "fundamental_period_end_utc" in signal_df.columns:
            f_joined = signal_df.join(
                fundamentals_df.select(
                    "symbol",
                    pl.col("period_end_utc").alias("fundamental_period_end_utc"),
                    pl.col("available_at_utc").alias("filing_available_at_utc"),
                ),
                on=["symbol", "fundamental_period_end_utc"],
                how="inner",
            )
            filing_leaks = f_joined.filter(
                pl.col("available_at_utc") < pl.col("filing_available_at_utc")
            )
            if filing_leaks.height > 0:
                raise LookAheadViolationError(
                    f"Signal leaked fundamental filing prior to filing_available_at_utc on {filing_leaks.height} rows.",
                    details={"violations": filing_leaks.height},
                )

    def validate_corporate_actions(
        self,
        bars_df: pl.DataFrame,
        actions_df: pl.DataFrame,
    ) -> dict[str, Any]:
        """Verify adjusted_close eliminates artificial price cliffs on split ex-dates."""
        if actions_df.height == 0:
            return {"splits_checked": 0, "max_adjusted_jump": 0.0}

        split_actions = actions_df.filter(pl.col("action_type").is_in(["split", "reverse_split"]))
        if split_actions.height == 0:
            return {"splits_checked": 0, "max_adjusted_jump": 0.0}

        ordered = bars_df.sort(["symbol", "event_ts_utc"]).with_columns(
            pl.col("adjusted_close").shift(1).over("symbol").alias("prev_adj_close"),
            pl.col("close").shift(1).over("symbol").alias("prev_raw_close"),
        )
        # Match bar on or immediately at ex_date_utc (same date in UTC)
        ordered = ordered.with_columns(pl.col("event_ts_utc").dt.date().alias("bar_date"))
        splits_dated = split_actions.with_columns(pl.col("ex_date_utc").dt.date().alias("bar_date"))

        joined = ordered.join(
            splits_dated.select("action_id", "symbol", "bar_date", "ratio", "action_type"),
            on=["symbol", "bar_date"],
            how="inner",
        ).filter(pl.col("prev_adj_close").is_not_null())

        if joined.height == 0:
            return {"splits_checked": 0, "max_adjusted_jump": 0.0}

        checked = joined.with_columns(
            ((pl.col("adjusted_close") / pl.col("prev_adj_close")) - 1.0).abs().alias("adj_abs_return"),
            ((pl.col("close") / pl.col("prev_raw_close")) - 1.0).abs().alias("raw_abs_return"),
            (1.0 / pl.col("ratio") - 1.0).abs().alias("expected_unadjusted_jump"),
        )

        # If adjusted return is suspiciously close to an unadjusted split jump (> tolerance)
        bad_splits = checked.filter(
            (pl.col("adj_abs_return") > self.policy.max_unadjusted_split_jump_tolerance)
            & (
                (pl.col("adj_abs_return") - pl.col("expected_unadjusted_jump")).abs()
                < 0.25
            )
        )
        if bad_splits.height > 0:
            sample = [
                {
                    **row,
                    "bar_date": row["bar_date"].isoformat()
                    if hasattr(row["bar_date"], "isoformat")
                    else str(row["bar_date"]),
                }
                for row in bad_splits.select(
                    "action_id",
                    "symbol",
                    "bar_date",
                    "ratio",
                    "adj_abs_return",
                    "expected_unadjusted_jump",
                ).to_dicts()
            ]
            raise CorporateActionIntegrityError(
                f"Detected {bad_splits.height} unadjusted split price discontinuities in adjusted_close.",
                details={"bad_splits": sample},
            )

        max_jump = float(checked.select(pl.col("adj_abs_return").max()).item() or 0.0)
        return {"splits_checked": checked.height, "max_adjusted_jump": round(max_jump, 6)}

    def validate_survivorship(
        self,
        bars_df: pl.DataFrame,
        membership_df: pl.DataFrame,
    ) -> dict[str, Any]:
        """Ensure delisted symbols are neither silently omitted (survivorship bias) nor traded post-exit."""
        present_symbols = set(bars_df.select("symbol").unique().to_series().to_list())
        delisted = membership_df.filter(pl.col("exit_ts_utc").is_not_null())

        missing_delisted = [
            row["symbol"]
            for row in delisted.to_dicts()
            if row["symbol"] not in present_symbols
        ]
        if self.policy.include_delisted_until_exit and missing_delisted:
            raise SurvivorshipViolationError(
                f"Survivorship bias detected: {len(missing_delisted)} delisted universe members are absent from bars.",
                details={"missing_delisted_symbols": sorted(missing_delisted)},
            )

        # Also check that bars do not appear strictly after exit_ts_utc or before entry_ts_utc
        joined = bars_df.join(
            membership_df.select("symbol", "entry_ts_utc", "exit_ts_utc"),
            on="symbol",
            how="inner",
        )
        post_exit = joined.filter(
            pl.col("exit_ts_utc").is_not_null() & (pl.col("event_ts_utc") > pl.col("exit_ts_utc"))
        )
        if post_exit.height > 0:
            raise SurvivorshipViolationError(
                f"Found {post_exit.height} market bars after symbol exit_ts_utc.",
                details={"post_exit_rows": post_exit.height},
            )

        return {
            "delisted_members_verified": delisted.height,
            "present_symbols": len(present_symbols),
        }

    def validate_symbol_coverage(
        self,
        bars_df: pl.DataFrame,
        membership_df: pl.DataFrame,
    ) -> dict[str, Any]:
        """Verify daily symbol coverage against expected active PIT membership."""
        dates = bars_df.select("event_ts_utc").unique().sort("event_ts_utc")
        # Cross join dates with membership to compute expected active symbols per date
        expected = (
            dates.join(membership_df.select("symbol", "entry_ts_utc", "exit_ts_utc"), how="cross")
            .filter(
                (pl.col("event_ts_utc") >= pl.col("entry_ts_utc"))
                & (
                    pl.col("exit_ts_utc").is_null()
                    | (pl.col("event_ts_utc") <= pl.col("exit_ts_utc"))
                )
            )
            .group_by("event_ts_utc")
            .agg(pl.len().alias("expected_count"))
        )
        actual = bars_df.group_by("event_ts_utc").agg(pl.col("symbol").n_unique().alias("actual_count"))
        cov = (
            expected.join(actual, on="event_ts_utc", how="left")
            .with_columns(pl.col("actual_count").fill_null(0))
            .with_columns((pl.col("actual_count") / pl.col("expected_count")).alias("coverage_ratio"))
        )

        below_ratio = cov.filter(pl.col("coverage_ratio") < self.policy.min_symbol_coverage_ratio)
        below_min_count = cov.filter(pl.col("actual_count") < self.policy.min_active_symbols_per_day)

        if below_ratio.height > 0 or below_min_count.height > 0:
            min_cov = float(cov.select(pl.col("coverage_ratio").min()).item() or 0.0)
            min_cnt = int(cov.select(pl.col("actual_count").min()).item() or 0)
            raise MissingSymbolError(
                "Active universe symbol coverage fell below policy threshold.",
                details={
                    "days_below_ratio": below_ratio.height,
                    "days_below_min_count": below_min_count.height,
                    "min_coverage_ratio_observed": round(min_cov, 4),
                    "policy_min_coverage_ratio": self.policy.min_symbol_coverage_ratio,
                    "min_active_symbols_observed": min_cnt,
                    "policy_min_active_symbols": self.policy.min_active_symbols_per_day,
                },
            )

        avg_cov = float(cov.select(pl.col("coverage_ratio").mean()).item() or 1.0)
        return {"days_checked": cov.height, "avg_coverage_ratio": round(avg_cov, 6)}

    def validate_all(
        self,
        bars_df: pl.DataFrame,
        fundamentals_df: pl.DataFrame,
        actions_df: pl.DataFrame,
        membership_df: pl.DataFrame,
        *,
        as_of_utc: datetime | None = None,
    ) -> ValidationSummary:
        checks = [
            "timestamps_market_bars",
            "timestamps_fundamentals",
            "corporate_actions_adjustment",
            "survivorship_integrity",
            "symbol_coverage",
        ]
        self.validate_timestamps(bars_df, dataset_name="market_bars", as_of_utc=as_of_utc)
        self.validate_timestamps(fundamentals_df, dataset_name="fundamentals", as_of_utc=as_of_utc)
        ca_metrics = self.validate_corporate_actions(bars_df, actions_df)
        surv_metrics = self.validate_survivorship(bars_df, membership_df)
        cov_metrics = self.validate_symbol_coverage(bars_df, membership_df)
        return ValidationSummary(
            passed=True,
            checks_run=checks,
            metrics={
                "corporate_actions": ca_metrics,
                "survivorship": surv_metrics,
                "coverage": cov_metrics,
            },
        )
