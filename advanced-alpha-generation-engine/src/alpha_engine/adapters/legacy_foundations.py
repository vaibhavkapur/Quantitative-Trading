"""Clean-room contract adapters for legacy portfolio foundation prototypes.

Per plan.md Section 2, external personal repositories (Quantum-Trading, Trading-Software)
could not be verified for public access, license, and SAAT background IP terms; therefore
these adapters implement clean, documented domain contracts that preserve the canonical
point-in-time signal boundary without copying external prototype code.
"""

from __future__ import annotations

from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from alpha_engine.adapters.base import AdapterProvenance, ExternalSignalAdapter
from alpha_engine.contracts.models import SignalDefinition
from alpha_engine.errors import DataValidationError
from alpha_engine.factors.base import BaseFactor
from alpha_engine.factors.library import (
    InverseVolatilityFactor,
    MomentumFactor,
    ShortTermReversalFactor,
)


class QuantumStatArbAdapter(ExternalSignalAdapter):
    """Clean-room adapter contract for statistical arbitrage residual mean-reversion."""

    def __init__(self, *, window_days: int = 20, version: str = "1.0.0") -> None:
        self.window_days = window_days
        self._def = SignalDefinition(
            signal_id=f"adapter_stat_arb_z_{window_days}d",
            name=f"Stat-Arb Sector Residual Z-Score ({window_days}d)",
            version=version,
            family="adapter_legacy",
            description=(
                "Clean-room contract adapter for 01-quantum-statistical-arbitrage: "
                "rolling negative z-score of cumulative return relative to sector mean."
            ),
            parameters={"window_days": window_days},
            lookback_bars=window_days + 1,
            horizon_bars=5,
            inputs_required=["market_bars"],
        )
        self._prov = AdapterProvenance(
            adapter_name="QuantumStatArbAdapter",
            upstream_reference="Quantum-Trading/projects/01-quantum-statistical-arbitrage",
            ip_clearance_status="unresolved_clean_room_contract_only",
            pinned_commit=None,
            known_limitations=[
                "Uses deterministic sector-demeaned residual z-score contract pending upstream IP/commit audit.",
                "Does not claim quantum advantage; intended strictly as baseline stat-arb signal contract.",
            ],
        )

    @property
    def definition(self) -> SignalDefinition:
        return self._def

    @property
    def provenance(self) -> AdapterProvenance:
        return self._prov

    def compute(
        self,
        bars_df: pl.DataFrame,
        fundamentals_df: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        if "sector" not in bars_df.columns:
            raise DataValidationError("QuantumStatArbAdapter requires 'sector' column on bars_df.")
        ordered = bars_df.sort(["symbol", "event_ts_utc"]).with_columns(
            (
                pl.col("adjusted_close") / pl.col("adjusted_close").shift(1).over("symbol") - 1.0
            ).alias("_ret")
        )
        with_sec = ordered.with_columns(
            (pl.col("_ret") - pl.col("_ret").mean().over(["event_ts_utc", "sector"])).alias("_resid")
        )
        rolled = with_sec.with_columns(
            pl.col("_resid").rolling_sum(self.window_days).over("symbol").alias("_cum_resid"),
        ).with_columns(
            pl.col("_cum_resid").rolling_mean(self.window_days).over("symbol").alias("_mu"),
            pl.col("_cum_resid").rolling_std(self.window_days).over("symbol").alias("_sd"),
        )
        out = rolled.with_columns(
            pl.when(pl.col("_sd") > 1e-12)
            .then(-((pl.col("_cum_resid") - pl.col("_mu")) / pl.col("_sd")))
            .otherwise(None)
            .alias("raw_value")
        )
        return self._finalize(out)


class QuantumPairsSpreadAdapter(ExternalSignalAdapter):
    """Clean-room adapter contract for intra-sector pairs spread mean reversion."""

    def __init__(self, *, window_days: int = 30, version: str = "1.0.0") -> None:
        self.window_days = window_days
        self._def = SignalDefinition(
            signal_id=f"adapter_pairs_spread_{window_days}d",
            name=f"Intra-Sector Pairs Spread Z-Score ({window_days}d)",
            version=version,
            family="adapter_legacy",
            description=(
                "Clean-room contract adapter for 10-quantum-pairs-trading: "
                "log-price spread vs sector median normalized by rolling spread volatility."
            ),
            parameters={"window_days": window_days},
            lookback_bars=window_days,
            horizon_bars=5,
            inputs_required=["market_bars"],
        )
        self._prov = AdapterProvenance(
            adapter_name="QuantumPairsSpreadAdapter",
            upstream_reference="Quantum-Trading/projects/10-quantum-pairs-trading",
            ip_clearance_status="unresolved_clean_room_contract_only",
            pinned_commit=None,
            known_limitations=[
                "Approximates pair spread against intra-sector median log-price pending cointegration pair registry.",
                "Subject to regime breakdown during structural corporate divergence.",
            ],
        )

    @property
    def definition(self) -> SignalDefinition:
        return self._def

    @property
    def provenance(self) -> AdapterProvenance:
        return self._prov

    def compute(
        self,
        bars_df: pl.DataFrame,
        fundamentals_df: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        if "sector" not in bars_df.columns:
            raise DataValidationError("QuantumPairsSpreadAdapter requires 'sector' column on bars_df.")
        with_log = bars_df.sort(["symbol", "event_ts_utc"]).with_columns(
            pl.col("adjusted_close").log().alias("_log_p")
        )
        with_spread = with_log.with_columns(
            (pl.col("_log_p") - pl.col("_log_p").median().over(["event_ts_utc", "sector"])).alias(
                "_spread"
            )
        )
        rolled = with_spread.with_columns(
            pl.col("_spread").rolling_mean(self.window_days).over("symbol").alias("_mu"),
            pl.col("_spread").rolling_std(self.window_days).over("symbol").alias("_sd"),
        )
        out = rolled.with_columns(
            pl.when(pl.col("_sd") > 1e-12)
            .then(-((pl.col("_spread") - pl.col("_mu")) / pl.col("_sd")))
            .otherwise(None)
            .alias("raw_value")
        )
        return self._finalize(out)


class QuantumOrderBookAdapter(ExternalSignalAdapter):
    """Clean-room adapter contract for order-book imbalance (daily OHLCV pressure proxy)."""

    def __init__(self, *, smoothing_days: int = 5, version: str = "1.0.0") -> None:
        self.smoothing_days = smoothing_days
        self._def = SignalDefinition(
            signal_id=f"adapter_ob_imbalance_{smoothing_days}d",
            name=f"Order-Book Imbalance Proxy ({smoothing_days}d)",
            version=version,
            family="adapter_legacy",
            description=(
                "Clean-room contract adapter for 07-quantum-order-book-predictor: "
                "volume-weighted close-location-value proxy when L2 order book feeds are absent."
            ),
            parameters={"smoothing_days": smoothing_days, "feed_mode": "ohlcv_proxy"},
            lookback_bars=smoothing_days,
            horizon_bars=5,
            inputs_required=["market_bars"],
        )
        self._prov = AdapterProvenance(
            adapter_name="QuantumOrderBookAdapter",
            upstream_reference="Quantum-Trading/projects/07-quantum-order-book-predictor",
            ip_clearance_status="unresolved_clean_room_contract_only",
            pinned_commit=None,
            known_limitations=[
                "Daily OHLCV proxy only; true L2 order-book imbalance requires tick/depth provider integration.",
                "Microstructure signals degrade rapidly at daily horizons and under realistic spread costs.",
            ],
        )

    @property
    def definition(self) -> SignalDefinition:
        return self._def

    @property
    def provenance(self) -> AdapterProvenance:
        return self._prov

    def compute(
        self,
        bars_df: pl.DataFrame,
        fundamentals_df: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        ordered = bars_df.sort(["symbol", "event_ts_utc"]).with_columns(
            pl.when((pl.col("high") - pl.col("low")) > 1e-9)
            .then(((pl.col("close") - pl.col("low")) - (pl.col("high") - pl.col("close"))) / (pl.col("high") - pl.col("low")))
            .otherwise(0.0)
            .alias("_clv"),
            pl.col("volume").log1p().alias("_log_vol"),
        )
        out = ordered.with_columns(
            (pl.col("_clv") * pl.col("_log_vol"))
            .rolling_mean(self.smoothing_days)
            .over("symbol")
            .alias("raw_value")
        )
        return self._finalize(out)


class ResearchHypothesisPayload(BaseModel):
    """Structured candidate proposal contract from an AI quant research agent."""

    model_config = ConfigDict(frozen=True)

    proposal_id: str = Field(min_length=1)
    proposed_by: str = Field(default="14-ai-quant-research-agent")
    template: Literal["momentum", "reversal", "inverse_volatility"]
    signal_id: str = Field(min_length=1)
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$", default="1.0.0")
    lookback_days: int = Field(ge=2, le=252)
    skip_days: int = Field(ge=0, le=30, default=5)
    rationale: str = Field(min_length=5)


class AIQuantResearchAgentAdapter:
    """Validates and compiles structured AI research-agent proposals into causal BaseFactor objects.

    Proposals NEVER bypass walk-forward evaluation or AlphaCatalog promotion gates.
    """

    provenance = AdapterProvenance(
        adapter_name="AIQuantResearchAgentAdapter",
        upstream_reference="Trading-Software/projects/14-ai-quant-research-agent",
        ip_clearance_status="unresolved_clean_room_contract_only",
        pinned_commit=None,
        known_limitations=[
            "Restricts agent-generated hypotheses to whitelisted causal templates (no arbitrary code execution).",
            "All compiled candidates must pass full walk-forward evaluation and catalog promotion gates.",
        ],
    )

    @classmethod
    def compile_proposal(cls, payload: ResearchHypothesisPayload) -> BaseFactor:
        if payload.template == "momentum":
            if payload.lookback_days <= payload.skip_days:
                raise DataValidationError(
                    "Momentum proposal requires lookback_days > skip_days.",
                    details={"proposal_id": payload.proposal_id},
                )
            return MomentumFactor(
                lookback_days=payload.lookback_days,
                skip_days=payload.skip_days,
                signal_id=payload.signal_id,
                version=payload.version,
            )
        if payload.template == "reversal":
            return ShortTermReversalFactor(
                lookback_days=payload.lookback_days,
                version=payload.version,
            )
        if payload.template == "inverse_volatility":
            return InverseVolatilityFactor(
                window_days=payload.lookback_days,
                version=payload.version,
            )
        raise DataValidationError(f"Unsupported template: {payload.template}")
