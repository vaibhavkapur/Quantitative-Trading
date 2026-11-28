"""Typed data contracts, validation rules, and error types for optimal execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCHEMA_VERSION = "1.0.0"
MINUTES_PER_TRADING_DAY = 390.0
CONSERVATION_TOLERANCE = 1e-6


class ExecutionEngineError(Exception):
    """Base exception for the optimal execution research engine."""


class ValidationError(ExecutionEngineError):
    """Raised when input contracts or invariants fail schema/domain validation."""


class UnsupportedScenarioError(ExecutionEngineError):
    """Raised when a requested strategy, policy, or parameter regime is unsupported."""


class ScheduleError(ExecutionEngineError):
    """Raised when a schedule cannot be constructed under given constraints."""


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> float:
        return 1.0 if self is OrderSide.BUY else -1.0


class StrategyType(str, Enum):
    TWAP = "twap"
    VWAP = "vwap"
    PARTICIPATION = "participation"
    ALMGREN_CHRISS_ANALYTIC = "almgren_chriss_analytic"
    ALMGREN_CHRISS_NUMERIC = "almgren_chriss_numeric"


class CapacityPolicy(str, Enum):
    """How unfilled slice quantity is handled when capacity caps a planned slice."""

    NONE = "none"  # Unfilled slice quantity becomes residual immediately.
    SPILLOVER = "spillover"  # Unfilled slice quantity rolls into subsequent slices.


def _require_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a numeric value, got {type(value).__name__}")
    val = float(value)
    if not math.isfinite(val):
        raise ValidationError(f"{name} must be finite, got {value}")
    return val


@dataclass(frozen=True)
class PriceShock:
    """Exogenous mid-price shock applied at the start of `slice_index`.

    `shock_bps` is expressed in basis points of arrival price S_0:
    positive values move the mid-price up; negative values move the mid-price down.
    """

    slice_index: int
    shock_bps: float
    label: str = "exogenous_shock"

    def validate(self, num_slices: int) -> None:
        if isinstance(self.slice_index, bool) or not isinstance(self.slice_index, int):
            raise ValidationError(
                f"PriceShock.slice_index must be int, got {type(self.slice_index).__name__}"
            )
        if self.slice_index < 0 or self.slice_index >= num_slices:
            raise ValidationError(
                f"PriceShock.slice_index {self.slice_index} out of bounds [0, {num_slices - 1}]"
            )
        _require_finite("PriceShock.shock_bps", self.shock_bps)


@dataclass(frozen=True)
class ParentOrder:
    """Canonical parent order specification."""

    order_id: str
    symbol: str
    side: OrderSide
    quantity: float  # shares (> 0)
    arrival_price: float  # USD per share (> 0)
    horizon_slices: int  # number of discrete execution slices (>= 1)
    slice_duration_minutes: float = 30.0  # length of each slice in minutes (> 0)
    start_timestamp_utc: str = "2026-01-05T14:30:00Z"  # ISO-8601 UTC start of slice 0

    @property
    def arrival_notional(self) -> float:
        return self.quantity * self.arrival_price

    @property
    def slice_duration_days(self) -> float:
        return self.slice_duration_minutes / MINUTES_PER_TRADING_DAY

    @property
    def horizon_days(self) -> float:
        return self.horizon_slices * self.slice_duration_days

    def validate(self) -> None:
        if not isinstance(self.order_id, str) or not self.order_id.strip():
            raise ValidationError("ParentOrder.order_id must be a non-empty string")
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValidationError("ParentOrder.symbol must be a non-empty string")
        if not isinstance(self.side, OrderSide):
            raise ValidationError(f"ParentOrder.side must be OrderSide, got {self.side!r}")
        qty = _require_finite("ParentOrder.quantity", self.quantity)
        if qty <= 0.0:
            raise ValidationError(f"ParentOrder.quantity must be > 0, got {qty}")
        px = _require_finite("ParentOrder.arrival_price", self.arrival_price)
        if px <= 0.0:
            raise ValidationError(f"ParentOrder.arrival_price must be > 0, got {px}")
        if isinstance(self.horizon_slices, bool) or not isinstance(self.horizon_slices, int):
            raise ValidationError("ParentOrder.horizon_slices must be an integer")
        if self.horizon_slices < 1:
            raise ValidationError(
                f"ParentOrder.horizon_slices must be >= 1, got {self.horizon_slices}"
            )
        dur = _require_finite("ParentOrder.slice_duration_minutes", self.slice_duration_minutes)
        if dur <= 0.0:
            raise ValidationError(
                f"ParentOrder.slice_duration_minutes must be > 0, got {dur}"
            )


@dataclass(frozen=True)
class MarketProfile:
    """Market volume curve, volatility, spread, and capacity profile."""

    volume_curve: Tuple[float, ...]  # Expected market volume per slice (shares >= 0)
    daily_volatility: float  # Fractional daily volatility, e.g., 0.02 = 2%/day (>= 0)
    bid_ask_spread_bps: float  # Full quoted bid-ask spread in bps (>= 0)
    max_participation_rate: float = 0.25  # Max fillable fraction of slice volume in (0, 1]
    max_slice_quantity: Optional[float] = None  # Optional absolute share cap per slice
    price_shocks: Tuple[PriceShock, ...] = ()

    @property
    def total_market_volume(self) -> float:
        return float(sum(self.volume_curve))

    def half_spread_usd(self, arrival_price: float) -> float:
        return 0.5 * (self.bid_ask_spread_bps * 1e-4) * arrival_price

    def slice_capacities(self) -> Tuple[float, ...]:
        caps: List[float] = []
        for vol in self.volume_curve:
            cap = self.max_participation_rate * vol
            if self.max_slice_quantity is not None:
                cap = min(cap, self.max_slice_quantity)
            caps.append(max(0.0, float(cap)))
        return tuple(caps)

    def validate(self, expected_slices: int) -> None:
        if not isinstance(self.volume_curve, (tuple, list)):
            raise ValidationError("MarketProfile.volume_curve must be a sequence of floats")
        if len(self.volume_curve) != expected_slices:
            raise ValidationError(
                f"MarketProfile.volume_curve length ({len(self.volume_curve)}) "
                f"must match horizon_slices ({expected_slices})"
            )
        for idx, vol in enumerate(self.volume_curve):
            v = _require_finite(f"MarketProfile.volume_curve[{idx}]", vol)
            if v < 0.0:
                raise ValidationError(
                    f"MarketProfile.volume_curve[{idx}] must be >= 0, got {v}"
                )
        vol_day = _require_finite("MarketProfile.daily_volatility", self.daily_volatility)
        if vol_day < 0.0:
            raise ValidationError(
                f"MarketProfile.daily_volatility must be >= 0, got {vol_day}"
            )
        spread = _require_finite("MarketProfile.bid_ask_spread_bps", self.bid_ask_spread_bps)
        if spread < 0.0:
            raise ValidationError(
                f"MarketProfile.bid_ask_spread_bps must be >= 0, got {spread}"
            )
        pov_cap = _require_finite(
            "MarketProfile.max_participation_rate", self.max_participation_rate
        )
        if pov_cap <= 0.0 or pov_cap > 1.0:
            raise ValidationError(
                f"MarketProfile.max_participation_rate must be in (0, 1], got {pov_cap}"
            )
        if self.max_slice_quantity is not None:
            cap = _require_finite("MarketProfile.max_slice_quantity", self.max_slice_quantity)
            if cap < 0.0:
                raise ValidationError(
                    f"MarketProfile.max_slice_quantity must be >= 0, got {cap}"
                )
        for shock in self.price_shocks:
            shock.validate(expected_slices)


@dataclass(frozen=True)
class ImpactParams:
    """Permanent and temporary market impact parameters (Almgren-Chriss convention).

    Units:
      - `permanent_impact_gamma`: USD / share^2 (permanent mid-price shift = gamma * shares)
      - `temporary_impact_eta`: USD * day / share^(1 + alpha)
        For linear temporary impact (alpha = 1.0): h(v) = half_spread + eta * (shares / tau_days)
      - `temporary_impact_alpha`: Exponent (> 0); 1.0 = linear (Almgren-Chriss), 0.5 = square-root.
    """

    permanent_impact_gamma: float
    temporary_impact_eta: float
    temporary_impact_alpha: float = 1.0

    def effective_eta(self, slice_duration_days: float) -> float:
        """Almgren-Chriss effective temporary impact coefficient eta_tilde = eta - 0.5 * gamma * tau."""
        return self.temporary_impact_eta - 0.5 * self.permanent_impact_gamma * slice_duration_days

    def validate(self, slice_duration_days: float) -> None:
        gamma = _require_finite(
            "ImpactParams.permanent_impact_gamma", self.permanent_impact_gamma
        )
        if gamma < 0.0:
            raise ValidationError(
                f"ImpactParams.permanent_impact_gamma must be >= 0, got {gamma}"
            )
        eta = _require_finite("ImpactParams.temporary_impact_eta", self.temporary_impact_eta)
        if eta < 0.0:
            raise ValidationError(
                f"ImpactParams.temporary_impact_eta must be >= 0, got {eta}"
            )
        alpha = _require_finite(
            "ImpactParams.temporary_impact_alpha", self.temporary_impact_alpha
        )
        if alpha <= 0.0:
            raise ValidationError(
                f"ImpactParams.temporary_impact_alpha must be > 0, got {alpha}"
            )
        # Check for non-convexity only when gamma > 0 and linear impact is used:
        # We allow eta >= 0 in general simulation, while Almgren-Chriss schedule checks eta_tilde > 0.
        _ = slice_duration_days


@dataclass(frozen=True)
class ScheduleRequest:
    """Strategy selection and strategy-specific tuning parameters."""

    strategy: StrategyType
    risk_aversion_lambda: float = 1e-6  # USD^-1 risk aversion for Almgren-Chriss (>= 0)
    target_participation_rate: float = 0.10  # Target POV rate in (0, 1]
    enforce_capacity_in_optimizer: bool = False  # Optional hard capacity bounds for numeric AC

    def validate(self) -> None:
        if not isinstance(self.strategy, StrategyType):
            raise UnsupportedScenarioError(f"Unsupported strategy: {self.strategy!r}")
        lam = _require_finite("ScheduleRequest.risk_aversion_lambda", self.risk_aversion_lambda)
        if lam < 0.0:
            raise ValidationError(
                f"ScheduleRequest.risk_aversion_lambda must be >= 0, got {lam}"
            )
        pov = _require_finite(
            "ScheduleRequest.target_participation_rate", self.target_participation_rate
        )
        if pov <= 0.0 or pov > 1.0:
            raise ValidationError(
                f"ScheduleRequest.target_participation_rate must be in (0, 1], got {pov}"
            )


@dataclass(frozen=True)
class ExecutionScenario:
    """Complete versioned scenario input combining order, market, impact, and simulation config."""

    scenario_id: str
    order: ParentOrder
    market: MarketProfile
    impact: ImpactParams
    capacity_policy: CapacityPolicy = CapacityPolicy.NONE
    seed: int = 42
    num_monte_carlo_paths: int = 500
    schema_version: str = SCHEMA_VERSION
    description: str = ""

    def validate(self) -> None:
        if not isinstance(self.scenario_id, str) or not self.scenario_id.strip():
            raise ValidationError("ExecutionScenario.scenario_id must be a non-empty string")
        if self.schema_version != SCHEMA_VERSION:
            raise UnsupportedScenarioError(
                f"Unsupported schema_version {self.schema_version!r}; expected {SCHEMA_VERSION!r}"
            )
        self.order.validate()
        self.market.validate(self.order.horizon_slices)
        self.impact.validate(self.order.slice_duration_days)
        if not isinstance(self.capacity_policy, CapacityPolicy):
            raise UnsupportedScenarioError(
                f"Unsupported capacity_policy: {self.capacity_policy!r}"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValidationError(f"ExecutionScenario.seed must be a non-negative int, got {self.seed!r}")
        if (
            isinstance(self.num_monte_carlo_paths, bool)
            or not isinstance(self.num_monte_carlo_paths, int)
            or self.num_monte_carlo_paths < 1
        ):
            raise ValidationError(
                "ExecutionScenario.num_monte_carlo_paths must be an integer >= 1"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "description": self.description,
            "capacity_policy": self.capacity_policy.value,
            "seed": self.seed,
            "num_monte_carlo_paths": self.num_monte_carlo_paths,
            "order": {
                "order_id": self.order.order_id,
                "symbol": self.order.symbol,
                "side": self.order.side.value,
                "quantity": self.order.quantity,
                "arrival_price": self.order.arrival_price,
                "horizon_slices": self.order.horizon_slices,
                "slice_duration_minutes": self.order.slice_duration_minutes,
                "start_timestamp_utc": self.order.start_timestamp_utc,
            },
            "market": {
                "volume_curve": list(self.market.volume_curve),
                "daily_volatility": self.market.daily_volatility,
                "bid_ask_spread_bps": self.market.bid_ask_spread_bps,
                "max_participation_rate": self.market.max_participation_rate,
                "max_slice_quantity": self.market.max_slice_quantity,
                "price_shocks": [asdict(s) for s in self.market.price_shocks],
            },
            "impact": asdict(self.impact),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionScenario":
        if not isinstance(data, dict):
            raise ValidationError("Scenario payload must be a JSON object")
        required_top = ("scenario_id", "order", "market", "impact")
        missing = [k for k in required_top if k not in data]
        if missing:
            raise ValidationError(f"Missing required scenario fields: {', '.join(missing)}")

        order_raw = data["order"]
        market_raw = data["market"]
        impact_raw = data["impact"]
        if not isinstance(order_raw, dict) or not isinstance(market_raw, dict) or not isinstance(impact_raw, dict):
            raise ValidationError("'order', 'market', and 'impact' must be objects")

        for section_name, section, keys in (
            ("order", order_raw, ("order_id", "symbol", "side", "quantity", "arrival_price", "horizon_slices")),
            ("market", market_raw, ("volume_curve", "daily_volatility", "bid_ask_spread_bps")),
            ("impact", impact_raw, ("permanent_impact_gamma", "temporary_impact_eta")),
        ):
            sec_missing = [k for k in keys if k not in section]
            if sec_missing:
                raise ValidationError(
                    f"Missing required '{section_name}' fields: {', '.join(sec_missing)}"
                )

        try:
            side = OrderSide(order_raw["side"])
        except ValueError as exc:
            raise ValidationError(f"Invalid order side: {order_raw['side']!r}") from exc

        policy_raw = data.get("capacity_policy", CapacityPolicy.NONE.value)
        try:
            policy = CapacityPolicy(policy_raw)
        except ValueError as exc:
            raise UnsupportedScenarioError(f"Unsupported capacity_policy: {policy_raw!r}") from exc

        shocks_raw = market_raw.get("price_shocks", [])
        if not isinstance(shocks_raw, list):
            raise ValidationError("market.price_shocks must be a list")
        shocks: List[PriceShock] = []
        for idx, item in enumerate(shocks_raw):
            if not isinstance(item, dict) or "slice_index" not in item or "shock_bps" not in item:
                raise ValidationError(
                    f"market.price_shocks[{idx}] must contain 'slice_index' and 'shock_bps'"
                )
            shocks.append(
                PriceShock(
                    slice_index=item["slice_index"],
                    shock_bps=item["shock_bps"],
                    label=str(item.get("label", f"shock_{idx}")),
                )
            )

        order = ParentOrder(
            order_id=order_raw["order_id"],
            symbol=order_raw["symbol"],
            side=side,
            quantity=order_raw["quantity"],
            arrival_price=order_raw["arrival_price"],
            horizon_slices=order_raw["horizon_slices"],
            slice_duration_minutes=order_raw.get("slice_duration_minutes", 30.0),
            start_timestamp_utc=order_raw.get("start_timestamp_utc", "2026-01-05T14:30:00Z"),
        )
        market = MarketProfile(
            volume_curve=tuple(market_raw["volume_curve"]),
            daily_volatility=market_raw["daily_volatility"],
            bid_ask_spread_bps=market_raw["bid_ask_spread_bps"],
            max_participation_rate=market_raw.get("max_participation_rate", 0.25),
            max_slice_quantity=market_raw.get("max_slice_quantity"),
            price_shocks=tuple(shocks),
        )
        impact = ImpactParams(
            permanent_impact_gamma=impact_raw["permanent_impact_gamma"],
            temporary_impact_eta=impact_raw["temporary_impact_eta"],
            temporary_impact_alpha=impact_raw.get("temporary_impact_alpha", 1.0),
        )
        scenario = cls(
            scenario_id=data["scenario_id"],
            order=order,
            market=market,
            impact=impact,
            capacity_policy=policy,
            seed=data.get("seed", 42),
            num_monte_carlo_paths=data.get("num_monte_carlo_paths", 500),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            description=str(data.get("description", "")),
        )
        scenario.validate()
        return scenario


@dataclass(frozen=True)
class Schedule:
    """Planned execution trajectory across discrete slices."""

    strategy: StrategyType
    planned_quantities: Tuple[float, ...]  # Length N, each >= 0
    remaining_trajectory: Tuple[float, ...]  # Length N + 1, x_0 = Q ... x_N = residual
    planned_residual: float  # Q - sum(planned_quantities) >= 0
    parameters: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_planned(self) -> float:
        return float(sum(self.planned_quantities))

    def validate(self, order: ParentOrder) -> None:
        if len(self.planned_quantities) != order.horizon_slices:
            raise ValidationError(
                f"Schedule length {len(self.planned_quantities)} != horizon_slices {order.horizon_slices}"
            )
        if len(self.remaining_trajectory) != order.horizon_slices + 1:
            raise ValidationError(
                f"Trajectory length {len(self.remaining_trajectory)} != horizon_slices + 1"
            )
        for idx, q in enumerate(self.planned_quantities):
            if not math.isfinite(q) or q < -CONSERVATION_TOLERANCE:
                raise ValidationError(f"Schedule slice {idx} has invalid quantity {q}")
        if self.planned_residual < -CONSERVATION_TOLERANCE:
            raise ValidationError(f"Schedule planned_residual must be >= 0, got {self.planned_residual}")
        total = self.total_planned + self.planned_residual
        if abs(total - order.quantity) > CONSERVATION_TOLERANCE * max(1.0, order.quantity):
            raise ValidationError(
                f"Order conservation violated in schedule: planned ({self.total_planned}) "
                f"+ residual ({self.planned_residual}) != parent ({order.quantity})"
            )


@dataclass(frozen=True)
class FillSlice:
    """Per-slice simulated market state, capacity, fill, and cost breakdown."""

    slice_index: int
    market_volume: float
    capacity: float
    planned_quantity: float
    target_quantity: float  # planned + carryover (if spillover enabled)
    filled_quantity: float
    unfilled_quantity: float
    remaining_before: float
    remaining_after: float
    mid_price_before: float
    mid_price_after: float
    execution_price: Optional[float]  # None if filled_quantity == 0
    participation_rate: float  # filled / market_volume (0.0 if market_volume == 0)
    spread_cost_usd: float
    temporary_impact_cost_usd: float
    permanent_impact_cost_usd: float
    timing_drift_cost_usd: float
    shock_cost_usd: float
    total_slice_shortfall_usd: float
    was_capacity_capped: bool
    was_zero_liquidity: bool
    shock_applied_usd: float


@dataclass(frozen=True)
class CostMetrics:
    """Aggregated implementation shortfall and timing risk metrics."""

    arrival_notional_usd: float
    filled_quantity: float
    residual_quantity: float
    completion_rate: float
    vwap_execution_price: Optional[float]
    terminal_mid_price: float
    spread_cost_usd: float
    temporary_impact_cost_usd: float
    permanent_impact_cost_usd: float
    timing_drift_cost_usd: float
    shock_cost_usd: float
    execution_shortfall_usd: float
    opportunity_cost_usd: float
    total_is_usd: float
    execution_shortfall_bps: float
    opportunity_cost_bps: float
    total_is_bps: float
    analytic_expected_is_usd: Optional[float]
    analytic_timing_risk_usd: Optional[float]
    analytic_expected_is_bps: Optional[float]
    analytic_timing_risk_bps: Optional[float]


@dataclass(frozen=True)
class SimulationDiagnostics:
    """Structured diagnostic counters for operational review and auditing."""

    capacity_capped_slices: int
    zero_liquidity_slices: int
    shock_events_applied: int
    spillover_events: int
    partial_fill: bool
    max_observed_participation: float
    conservation_error: float


@dataclass(frozen=True)
class SimulationResult:
    """Deterministic or single-path simulation output."""

    scenario_id: str
    strategy: StrategyType
    schedule: Schedule
    slices: Tuple[FillSlice, ...]
    metrics: CostMetrics
    diagnostics: SimulationDiagnostics
    stochastic: bool
    seed: Optional[int]


@dataclass(frozen=True)
class MonteCarloSummary:
    """Distributional summary across Monte Carlo price paths."""

    scenario_id: str
    strategy: StrategyType
    seed: int
    num_paths: int
    mean_total_is_usd: float
    std_total_is_usd: float
    mean_total_is_bps: float
    std_total_is_bps: float
    p05_total_is_bps: float
    p50_total_is_bps: float
    p95_total_is_bps: float
    var_95_usd: float
    cvar_95_usd: float
    var_95_bps: float
    cvar_95_bps: float
    analytic_expected_is_usd: Optional[float]
    analytic_timing_risk_usd: Optional[float]
    analytic_expected_is_bps: Optional[float]
    analytic_timing_risk_bps: Optional[float]
    mean_completion_rate: float
    mean_execution_shortfall_bps: float
    mean_opportunity_cost_bps: float


@dataclass(frozen=True)
class SensitivityPoint:
    """Single grid point in a liquidity/urgency sensitivity sweep."""

    regime_name: str
    liquidity_multiplier: float
    spread_multiplier: float
    volatility_multiplier: float
    risk_aversion_lambda: float
    strategy: str
    completion_rate: float
    deterministic_is_bps: float
    mc_mean_is_bps: float
    mc_std_is_bps: float
    mc_p95_is_bps: float
    mc_cvar_95_bps: float
    analytic_expected_is_bps: Optional[float]
    analytic_timing_risk_bps: Optional[float]
    capacity_capped_slices: int


@dataclass(frozen=True)
class RunMetadata:
    """Versioned runtime and reproducibility metadata recorded alongside every report."""

    run_id: str
    schema_version: str
    engine_version: str
    timestamp_utc: str
    python_version: str
    platform: str
    package_versions: Dict[str, str]
    scenario_id: str
    seed: int
    config_sha256: str
