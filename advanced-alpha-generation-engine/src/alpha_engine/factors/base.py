"""Base factor contract and causal feature matrix assembler."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import polars as pl

from alpha_engine.contracts.models import SignalDefinition
from alpha_engine.errors import DataValidationError

REQUIRED_SIGNAL_COLUMNS = (
    "signal_id",
    "signal_version",
    "symbol",
    "event_ts_utc",
    "available_at_utc",
    "raw_value",
)


class BaseFactor(ABC):
    """Canonical causal factor interface producing point-in-time signal frames."""

    @property
    @abstractmethod
    def definition(self) -> SignalDefinition:
        """Return immutable versioned SignalDefinition."""

    @abstractmethod
    def compute(
        self,
        bars_df: pl.DataFrame,
        fundamentals_df: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Compute causal factor values with explicit event_ts_utc and available_at_utc."""

    def _finalize(self, df: pl.DataFrame) -> pl.DataFrame:
        defn = self.definition
        out = df.with_columns(
            pl.lit(defn.signal_id).alias("signal_id"),
            pl.lit(defn.version).alias("signal_version"),
        ).filter(pl.col("raw_value").is_not_null() & pl.col("raw_value").is_finite())

        missing = [c for c in REQUIRED_SIGNAL_COLUMNS if c not in out.columns]
        if missing:
            raise DataValidationError(
                f"Factor {defn.signal_id} output missing required columns: {missing}",
                details={"signal_id": defn.signal_id, "missing_columns": missing},
            )
        base_cols = list(REQUIRED_SIGNAL_COLUMNS)
        extra_cols = [c for c in ("source_available_at_utc", "fundamental_period_end_utc") if c in out.columns]
        return out.select(base_cols + extra_cols).sort(["event_ts_utc", "symbol"])


class FeatureMatrixBuilder:
    """Assembles multiple causal factors into a single aligned PIT feature matrix."""

    def __init__(self, factors: Sequence[BaseFactor]) -> None:
        self.factors = list(factors)
        ids = [f.definition.signal_id for f in self.factors]
        if len(ids) != len(set(ids)):
            raise DataValidationError(
                "Duplicate signal_id in FeatureMatrixBuilder.",
                details={"signal_ids": ids},
            )

    def build(
        self,
        bars_df: pl.DataFrame,
        fundamentals_df: pl.DataFrame | None = None,
    ) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
        """Return (wide_feature_matrix, individual_signal_frames)."""
        frames: dict[str, pl.DataFrame] = {}
        base = bars_df.select(
            "symbol",
            "event_ts_utc",
            "available_at_utc",
            *[c for c in ("sector", "adjusted_close", "volume") if c in bars_df.columns],
        )

        wide = base
        for factor in self.factors:
            sig_id = factor.definition.signal_id
            sig_df = factor.compute(bars_df, fundamentals_df=fundamentals_df)
            frames[sig_id] = sig_df

            renamed = sig_df.select(
                "symbol",
                "event_ts_utc",
                pl.col("raw_value").alias(sig_id),
                pl.col("available_at_utc").alias(f"__avail_{sig_id}"),
            )
            wide = wide.join(renamed, on=["symbol", "event_ts_utc"], how="left")
            # Ensure composite feature row availability is the max across all contributing inputs
            wide = wide.with_columns(
                pl.max_horizontal("available_at_utc", f"__avail_{sig_id}").alias("available_at_utc")
            ).drop(f"__avail_{sig_id}")

        return wide.sort(["event_ts_utc", "symbol"]), frames
