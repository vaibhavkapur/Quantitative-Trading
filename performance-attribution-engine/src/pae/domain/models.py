"""Canonical domain contracts with explicit currency, UTC timestamps, and stable IDs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PositionSnapshot:
    """Reconciled daily position record linking holdings to strategy, instrument, and sector."""

    position_id: str
    trade_date: str
    portfolio_id: str
    strategy_id: str
    instrument_id: str
    sector: str
    asset_class: str
    local_currency: str
    start_quantity: float
    end_quantity: float
    start_price_local: float
    end_price_local: float
    income_local: float = 0.0
    explicit_fees_local: float = 0.0
    start_market_value_base: Optional[float] = None
    end_market_value_base: Optional[float] = None
    as_of_utc: str = "2026-01-01T21:00:00Z"
    availability_utc: str = "2026-01-01T21:05:00Z"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TradeExecution:
    """Executed trade record supporting intraday rebalance P&L and Implementation Shortfall TCA."""

    trade_id: str
    order_id: str
    trade_date: str
    execution_ts_utc: str
    portfolio_id: str
    strategy_id: str
    instrument_id: str
    sector: str
    side: str  # "BUY" or "SELL"
    order_quantity: float
    quantity: float
    local_currency: str
    decision_price_local: float
    arrival_price_local: float
    execution_price_local: float
    close_price_local: float
    benchmark_vwap_local: Optional[float] = None
    commission_local: float = 0.0
    exchange_fee_local: float = 0.0
    tax_local: float = 0.0
    venue: str = "SIM_EXCHANGE"
    availability_utc: str = "2026-01-01T21:05:00Z"

    @property
    def signed_quantity(self) -> float:
        return self.quantity if self.side.upper() == "BUY" else -self.quantity

    @property
    def unfilled_signed_quantity(self) -> float:
        unfilled = max(0.0, self.order_quantity - self.quantity)
        return unfilled if self.side.upper() == "BUY" else -unfilled

    @property
    def total_explicit_fee_local(self) -> float:
        return self.commission_local + self.exchange_fee_local + self.tax_local

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CashFlow:
    """External investor subscription (+) or redemption (-)."""

    cash_flow_id: str
    trade_date: str
    portfolio_id: str
    amount_local: float
    currency: str
    timing: str = "START_OF_DAY"  # "START_OF_DAY" or "END_OF_DAY"
    description: str = ""
    availability_utc: str = "2026-01-01T14:00:00Z"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeeAdjustment:
    """Fund-level or strategy/instrument fee or retroactive fee correction."""

    fee_id: str
    trade_date: str
    portfolio_id: str
    fee_type: str
    amount_local: float  # positive = cost reducing net return; negative = rebate/credit
    currency: str
    strategy_id: Optional[str] = None
    instrument_id: Optional[str] = None
    is_correction: bool = False
    supersedes_fee_id: Optional[str] = None
    correction_reason: Optional[str] = None
    availability_utc: str = "2026-01-01T21:05:00Z"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FxRateObservation:
    """Point-in-time FX rate: 1 base_currency = rate quote_currency."""

    pair: str
    base_currency: str
    quote_currency: str
    trade_date: str
    rate: float
    as_of_utc: str
    availability_utc: str
    source: str = "SYNTHETIC_FIXING"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DailyNavSnapshot:
    """Reconciled portfolio-level daily NAV ledger snapshot in reporting currency."""

    nav_id: str
    trade_date: str
    portfolio_id: str
    reporting_currency: str
    start_nav_base: float
    end_nav_base: float
    start_cash_base: float = 0.0
    end_cash_base: float = 0.0
    cash_interest_base: float = 0.0
    as_of_utc: str = "2026-01-01T21:00:00Z"
    availability_utc: str = "2026-01-01T21:05:00Z"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkObservation:
    """Daily benchmark total return and optional sector weights/returns in benchmark currency."""

    benchmark_id: str
    trade_date: str
    currency: str
    total_return: float
    sector_weights: Dict[str, float] = field(default_factory=dict)
    sector_returns: Dict[str, float] = field(default_factory=dict)
    as_of_utc: str = "2026-01-01T21:00:00Z"
    availability_utc: str = "2026-01-01T21:05:00Z"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FactorObservation:
    """Daily point-in-time risk factor returns."""

    trade_date: str
    factor_returns: Dict[str, float]
    risk_free_rate: float = 0.0
    currency: str = "USD"
    availability_utc: str = "2026-01-01T21:10:00Z"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DataQualityException:
    """Structured exception surfaced in attribution reports rather than silently hidden."""

    code: str
    severity: str  # "INFO", "WARNING", "ERROR", "CRITICAL"
    message: str
    trade_date: Optional[str] = None
    entity_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AttributionInputBatch:
    """Versioned, content-addressed canonical input bundle for attribution."""

    batch_id: str
    portfolio_id: str
    reporting_currency: str
    as_of_utc: str
    data_source: str = "SYNTHETIC_FIXTURE"
    entitlement_tag: str = "PERMITTED_SYNTHETIC"
    benchmark_id: Optional[str] = None
    nav_snapshots: List[DailyNavSnapshot] = field(default_factory=list)
    positions: List[PositionSnapshot] = field(default_factory=list)
    trades: List[TradeExecution] = field(default_factory=list)
    cash_flows: List[CashFlow] = field(default_factory=list)
    fee_adjustments: List[FeeAdjustment] = field(default_factory=list)
    fx_rates: List[FxRateObservation] = field(default_factory=list)
    benchmark: List[BenchmarkObservation] = field(default_factory=list)
    factors: List[FactorObservation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "portfolio_id": self.portfolio_id,
            "reporting_currency": self.reporting_currency,
            "as_of_utc": self.as_of_utc,
            "data_source": self.data_source,
            "entitlement_tag": self.entitlement_tag,
            "benchmark_id": self.benchmark_id,
            "nav_snapshots": [x.to_dict() for x in self.nav_snapshots],
            "positions": [x.to_dict() for x in self.positions],
            "trades": [x.to_dict() for x in self.trades],
            "cash_flows": [x.to_dict() for x in self.cash_flows],
            "fee_adjustments": [x.to_dict() for x in self.fee_adjustments],
            "fx_rates": [x.to_dict() for x in self.fx_rates],
            "benchmark": [x.to_dict() for x in self.benchmark],
            "factors": [x.to_dict() for x in self.factors],
        }

    def content_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "AttributionInputBatch":
        return cls(
            batch_id=raw["batch_id"],
            portfolio_id=raw["portfolio_id"],
            reporting_currency=str(raw["reporting_currency"]).upper(),
            as_of_utc=raw["as_of_utc"],
            data_source=raw.get("data_source", "SYNTHETIC_FIXTURE"),
            entitlement_tag=raw.get("entitlement_tag", "PERMITTED_SYNTHETIC"),
            benchmark_id=raw.get("benchmark_id"),
            nav_snapshots=[DailyNavSnapshot(**x) for x in raw.get("nav_snapshots", [])],
            positions=[PositionSnapshot(**x) for x in raw.get("positions", [])],
            trades=[TradeExecution(**x) for x in raw.get("trades", [])],
            cash_flows=[CashFlow(**x) for x in raw.get("cash_flows", [])],
            fee_adjustments=[FeeAdjustment(**x) for x in raw.get("fee_adjustments", [])],
            fx_rates=[FxRateObservation(**x) for x in raw.get("fx_rates", [])],
            benchmark=[BenchmarkObservation(**x) for x in raw.get("benchmark", [])],
            factors=[FactorObservation(**x) for x in raw.get("factors", [])],
        )
