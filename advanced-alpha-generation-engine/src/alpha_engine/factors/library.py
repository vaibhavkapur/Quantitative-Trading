"""Core M2 factor library: momentum, reversal, volatility, and PIT fundamentals."""

from __future__ import annotations

import polars as pl

from alpha_engine.contracts.models import SignalDefinition
from alpha_engine.data.loader import PointInTimeDataLoader
from alpha_engine.errors import DataValidationError
from alpha_engine.factors.base import BaseFactor


class MomentumFactor(BaseFactor):
    """Cross-sectional causal momentum over adjusted_close skipping the most recent days."""

    def __init__(
        self,
        *,
        lookback_days: int = 20,
        skip_days: int = 5,
        signal_id: str | None = None,
        version: str = "1.0.0",
    ) -> None:
        if lookback_days <= skip_days:
            raise ValueError("lookback_days must be greater than skip_days.")
        self.lookback_days = lookback_days
        self.skip_days = skip_days
        self._def = SignalDefinition(
            signal_id=signal_id or f"mom_{lookback_days}_{skip_days}",
            name=f"Price Momentum ({lookback_days}d skip {skip_days}d)",
            version=version,
            family="momentum",
            description=(
                f"Cumulative adjusted return from t-{lookback_days} to t-{skip_days}, "
                "using corporate-action adjusted_close."
            ),
            parameters={"lookback_days": lookback_days, "skip_days": skip_days},
            lookback_bars=lookback_days,
            horizon_bars=5,
            inputs_required=["market_bars"],
        )

    @property
    def definition(self) -> SignalDefinition:
        return self._def

    def compute(
        self,
        bars_df: pl.DataFrame,
        fundamentals_df: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        ordered = bars_df.sort(["symbol", "event_ts_utc"])
        out = ordered.with_columns(
            (
                pl.col("adjusted_close").shift(self.skip_days).over("symbol")
                / pl.col("adjusted_close").shift(self.lookback_days).over("symbol")
                - 1.0
            ).alias("raw_value")
        )
        return self._finalize(out)


class LongMomentumFactor(MomentumFactor):
    """Medium-horizon 60-day momentum skipping the most recent 5 days."""

    def __init__(self, version: str = "1.0.0") -> None:
        super().__init__(lookback_days=60, skip_days=5, signal_id="mom_60_5", version=version)


class ShortTermReversalFactor(BaseFactor):
    """Short-term contrarian reversal (negative recent return); typically high-turnover."""

    def __init__(self, *, lookback_days: int = 3, version: str = "1.0.0") -> None:
        self.lookback_days = lookback_days
        self._def = SignalDefinition(
            signal_id=f"rev_{lookback_days}d",
            name=f"Short-Term Reversal ({lookback_days}d)",
            version=version,
            family="reversal",
            description=(
                f"Negative {lookback_days}-day adjusted return capturing short-horizon mean reversion."
            ),
            parameters={"lookback_days": lookback_days},
            lookback_bars=lookback_days,
            horizon_bars=5,
            inputs_required=["market_bars"],
        )

    @property
    def definition(self) -> SignalDefinition:
        return self._def

    def compute(
        self,
        bars_df: pl.DataFrame,
        fundamentals_df: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        ordered = bars_df.sort(["symbol", "event_ts_utc"])
        out = ordered.with_columns(
            (
                -(
                    pl.col("adjusted_close")
                    / pl.col("adjusted_close").shift(self.lookback_days).over("symbol")
                    - 1.0
                )
            ).alias("raw_value")
        )
        return self._finalize(out)


class InverseVolatilityFactor(BaseFactor):
    """Low-volatility anomaly factor: negative rolling realized volatility."""

    def __init__(self, *, window_days: int = 20, version: str = "1.0.0") -> None:
        self.window_days = window_days
        self._def = SignalDefinition(
            signal_id=f"inv_vol_{window_days}d",
            name=f"Inverse Realized Volatility ({window_days}d)",
            version=version,
            family="volatility",
            description=(
                f"Negative {window_days}-day rolling standard deviation of daily adjusted returns."
            ),
            parameters={"window_days": window_days},
            lookback_bars=window_days + 1,
            horizon_bars=5,
            inputs_required=["market_bars"],
        )

    @property
    def definition(self) -> SignalDefinition:
        return self._def

    def compute(
        self,
        bars_df: pl.DataFrame,
        fundamentals_df: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        ordered = bars_df.sort(["symbol", "event_ts_utc"]).with_columns(
            (
                pl.col("adjusted_close") / pl.col("adjusted_close").shift(1).over("symbol") - 1.0
            ).alias("_ret_1d")
        )
        out = ordered.with_columns(
            (-pl.col("_ret_1d").rolling_std(window_size=self.window_days).over("symbol")).alias(
                "raw_value"
            )
        )
        return self._finalize(out)


class EarningsYieldFactor(BaseFactor):
    """Point-in-time earnings yield (TTM earnings / market cap) respecting filing lag."""

    def __init__(self, *, version: str = "1.0.0") -> None:
        self._def = SignalDefinition(
            signal_id="val_earnings_yield",
            name="PIT Earnings Yield",
            version=version,
            family="fundamental",
            description=(
                "Trailing 12-month earnings divided by market cap, joined strictly backward "
                "on fundamental filing available_at_utc."
            ),
            parameters={"join_strategy": "backward_available_at_utc"},
            lookback_bars=1,
            horizon_bars=5,
            inputs_required=["market_bars", "fundamentals"],
        )

    @property
    def definition(self) -> SignalDefinition:
        return self._def

    def compute(
        self,
        bars_df: pl.DataFrame,
        fundamentals_df: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        if fundamentals_df is None:
            raise DataValidationError(
                "EarningsYieldFactor requires fundamentals_df.",
                details={"signal_id": self._def.signal_id},
            )
        joined = PointInTimeDataLoader.attach_pit_fundamentals(bars_df, fundamentals_df)
        out = joined.with_columns(
            (pl.col("earnings_ttm_usd") / pl.col("market_cap_usd")).alias("raw_value"),
            pl.col("filing_available_at_utc").alias("source_available_at_utc"),
        )
        return self._finalize(out)


class QualityROEFactor(BaseFactor):
    """Point-in-time Return on Equity (ROE) respecting filing lag."""

    def __init__(self, *, version: str = "1.0.0") -> None:
        self._def = SignalDefinition(
            signal_id="qual_roe",
            name="PIT Return on Equity",
            version=version,
            family="fundamental",
            description=(
                "Return on equity from point-in-time filings joined strictly backward on available_at_utc."
            ),
            parameters={"join_strategy": "backward_available_at_utc"},
            lookback_bars=1,
            horizon_bars=5,
            inputs_required=["market_bars", "fundamentals"],
        )

    @property
    def definition(self) -> SignalDefinition:
        return self._def

    def compute(
        self,
        bars_df: pl.DataFrame,
        fundamentals_df: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        if fundamentals_df is None:
            raise DataValidationError(
                "QualityROEFactor requires fundamentals_df.",
                details={"signal_id": self._def.signal_id},
            )
        joined = PointInTimeDataLoader.attach_pit_fundamentals(bars_df, fundamentals_df)
        out = joined.with_columns(
            pl.col("roe").alias("raw_value"),
            pl.col("filing_available_at_utc").alias("source_available_at_utc"),
        )
        return self._finalize(out)


def default_factor_suite() -> list[BaseFactor]:
    """Return the canonical M2 core factor suite."""
    return [
        MomentumFactor(lookback_days=20, skip_days=5),
        LongMomentumFactor(),
        ShortTermReversalFactor(lookback_days=3),
        InverseVolatilityFactor(window_days=20),
        EarningsYieldFactor(),
        QualityROEFactor(),
    ]
