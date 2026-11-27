"""Canonical domain contracts for the Dynamic Capital Allocation Engine.

All timestamps are strictly UTC ISO-8601. All monetary values carry explicit
currency (default USD) and all identifiers are stable strings.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple

from dcae.errors import ValidationError

SCHEMA_VERSION = "1.0.0"


def parse_utc(value: str | datetime) -> datetime:
    """Parse an ISO-8601 UTC timestamp and enforce timezone awareness."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValidationError(
                "Timestamp must be timezone-aware UTC",
                context={"value": value.isoformat()},
            )
        return value.astimezone(timezone.utc)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError("Timestamp must be a non-empty ISO-8601 string")
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValidationError(
            f"Invalid ISO-8601 timestamp: {value}",
            context={"value": value},
        ) from exc
    if dt.tzinfo is None:
        raise ValidationError(
            f"Timestamp missing UTC offset: {value}",
            context={"value": value},
        )
    return dt.astimezone(timezone.utc)


def format_utc(dt: datetime) -> str:
    """Format a timezone-aware datetime as canonical UTC ISO-8601 with 'Z'."""
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(obj: Any) -> str:
    """Serialize an object into deterministic canonical JSON for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_hex(obj: Any) -> str:
    """Compute SHA-256 hex digest of a canonical JSON representation."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def _check_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be numeric", context={"field": name, "value": value})
    fval = float(value)
    if not math.isfinite(fval):
        raise ValidationError(f"{name} must be finite", context={"field": name, "value": value})
    return fval


class AllocationStatus(str, Enum):
    OPTIMAL = "OPTIMAL"
    CONVERGED_APPROX = "CONVERGED_APPROX"
    FALLBACK_NO_CHANGE = "FALLBACK_NO_CHANGE"
    FALLBACK_CASH = "FALLBACK_CASH"
    REJECTED = "REJECTED"


class ProposalStatus(str, Enum):
    DRAFT = "DRAFT"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    PAPER_EXECUTED = "PAPER_EXECUTED"


@dataclass(frozen=True)
class Provenance:
    """Explicit data lineage, licensing, and synthetic/permitted markers."""

    source: str
    entitlement: str  # e.g., "synthetic_internal", "permitted_historical"
    generator_version: str = "1.0.0"
    seed: Optional[int] = None
    notes: str = ""

    def validate(self) -> None:
        if not self.source or not self.entitlement:
            raise ValidationError("Provenance requires non-empty source and entitlement")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "entitlement": self.entitlement,
            "generator_version": self.generator_version,
            "seed": self.seed,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Provenance:
        return cls(
            source=str(data["source"]),
            entitlement=str(data["entitlement"]),
            generator_version=str(data.get("generator_version", "1.0.0")),
            seed=data.get("seed"),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True)
class ReturnObservation:
    """Point-in-time strategy return and NAV observation."""

    event_at_utc: str
    available_at_utc: str
    nav_usd: float
    period_return: float

    def validate(self) -> None:
        ev = parse_utc(self.event_at_utc)
        av = parse_utc(self.available_at_utc)
        if av < ev:
            raise ValidationError(
                "available_at_utc cannot precede event_at_utc",
                context={"event_at_utc": self.event_at_utc, "available_at_utc": self.available_at_utc},
            )
        nav = _check_finite("nav_usd", self.nav_usd)
        ret = _check_finite("period_return", self.period_return)
        if nav <= 0.0:
            raise ValidationError("nav_usd must be positive", context={"nav_usd": nav})
        if ret <= -1.0 or ret > 5.0:
            raise ValidationError(
                "period_return out of plausible bounds (-1.0, 5.0]",
                context={"period_return": ret},
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_at_utc": format_utc(parse_utc(self.event_at_utc)),
            "available_at_utc": format_utc(parse_utc(self.available_at_utc)),
            "nav_usd": float(self.nav_usd),
            "period_return": float(self.period_return),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReturnObservation:
        return cls(
            event_at_utc=str(data["event_at_utc"]),
            available_at_utc=str(data["available_at_utc"]),
            nav_usd=float(data["nav_usd"]),
            period_return=float(data["period_return"]),
        )


@dataclass(frozen=True)
class StrategyExposure:
    """Canonical strategy exposure, liquidity, and capacity attributes."""

    strategy_group: str
    asset_class: str
    gross_exposure: float
    net_exposure: float
    market_beta: float
    adv_usd: float
    max_capacity_usd: float
    transaction_cost_bps: float
    liquidity_days_to_exit: float = 1.0

    def validate(self) -> None:
        if not self.strategy_group or not self.asset_class:
            raise ValidationError("strategy_group and asset_class must be non-empty")
        gross = _check_finite("gross_exposure", self.gross_exposure)
        _check_finite("net_exposure", self.net_exposure)
        _check_finite("market_beta", self.market_beta)
        adv = _check_finite("adv_usd", self.adv_usd)
        cap = _check_finite("max_capacity_usd", self.max_capacity_usd)
        tcost = _check_finite("transaction_cost_bps", self.transaction_cost_bps)
        days = _check_finite("liquidity_days_to_exit", self.liquidity_days_to_exit)
        if gross < 0.0:
            raise ValidationError("gross_exposure must be >= 0", context={"gross_exposure": gross})
        if adv <= 0.0:
            raise ValidationError("adv_usd must be > 0", context={"adv_usd": adv})
        if cap < 0.0:
            raise ValidationError("max_capacity_usd must be >= 0", context={"max_capacity_usd": cap})
        if tcost < 0.0 or tcost > 1000.0:
            raise ValidationError("transaction_cost_bps must be in [0, 1000]")
        if days < 0.0:
            raise ValidationError("liquidity_days_to_exit must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_group": self.strategy_group,
            "asset_class": self.asset_class,
            "gross_exposure": float(self.gross_exposure),
            "net_exposure": float(self.net_exposure),
            "market_beta": float(self.market_beta),
            "adv_usd": float(self.adv_usd),
            "max_capacity_usd": float(self.max_capacity_usd),
            "transaction_cost_bps": float(self.transaction_cost_bps),
            "liquidity_days_to_exit": float(self.liquidity_days_to_exit),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StrategyExposure:
        return cls(
            strategy_group=str(data["strategy_group"]),
            asset_class=str(data["asset_class"]),
            gross_exposure=float(data["gross_exposure"]),
            net_exposure=float(data["net_exposure"]),
            market_beta=float(data["market_beta"]),
            adv_usd=float(data["adv_usd"]),
            max_capacity_usd=float(data["max_capacity_usd"]),
            transaction_cost_bps=float(data["transaction_cost_bps"]),
            liquidity_days_to_exit=float(data.get("liquidity_days_to_exit", 1.0)),
        )


@dataclass(frozen=True)
class StrategySnapshot:
    """Canonical performance, NAV, eligibility, and exposure snapshot for one strategy."""

    strategy_id: str
    name: str
    currency: str
    eligible: bool
    eligibility_reason: str
    exposure: StrategyExposure
    returns: Tuple[ReturnObservation, ...]
    min_weight: Optional[float] = None
    max_weight: Optional[float] = None
    risk_budget: Optional[float] = None
    current_drawdown: float = 0.0

    def validate(self) -> None:
        if not self.strategy_id or not self.name:
            raise ValidationError("strategy_id and name must be non-empty")
        if self.currency != "USD":
            raise ValidationError(
                "Only USD base currency is supported in MVP",
                context={"strategy_id": self.strategy_id, "currency": self.currency},
            )
        self.exposure.validate()
        prev_ev: Optional[datetime] = None
        for obs in self.returns:
            obs.validate()
            ev = parse_utc(obs.event_at_utc)
            if prev_ev is not None and ev <= prev_ev:
                raise ValidationError(
                    "Return observations must be strictly chronologically ordered",
                    context={"strategy_id": self.strategy_id, "event_at_utc": obs.event_at_utc},
                )
            prev_ev = ev
        if self.min_weight is not None:
            _check_finite("min_weight", self.min_weight)
            if self.min_weight < 0.0 or self.min_weight > 1.0:
                raise ValidationError("min_weight must be in [0, 1]")
        if self.max_weight is not None:
            _check_finite("max_weight", self.max_weight)
            if self.max_weight < 0.0 or self.max_weight > 1.0:
                raise ValidationError("max_weight must be in [0, 1]")
        if self.risk_budget is not None:
            rb = _check_finite("risk_budget", self.risk_budget)
            if rb < 0.0:
                raise ValidationError("risk_budget must be >= 0")
        dd = _check_finite("current_drawdown", self.current_drawdown)
        if dd > 0.0 or dd < -1.0:
            raise ValidationError(
                "current_drawdown must be in [-1.0, 0.0]",
                context={"strategy_id": self.strategy_id, "current_drawdown": dd},
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "currency": self.currency,
            "eligible": self.eligible,
            "eligibility_reason": self.eligibility_reason,
            "exposure": self.exposure.to_dict(),
            "returns": [r.to_dict() for r in self.returns],
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "risk_budget": self.risk_budget,
            "current_drawdown": float(self.current_drawdown),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StrategySnapshot:
        return cls(
            strategy_id=str(data["strategy_id"]),
            name=str(data["name"]),
            currency=str(data.get("currency", "USD")),
            eligible=bool(data["eligible"]),
            eligibility_reason=str(data.get("eligibility_reason", "ok")),
            exposure=StrategyExposure.from_dict(data["exposure"]),
            returns=tuple(ReturnObservation.from_dict(r) for r in data.get("returns", [])),
            min_weight=None if data.get("min_weight") is None else float(data["min_weight"]),
            max_weight=None if data.get("max_weight") is None else float(data["max_weight"]),
            risk_budget=None if data.get("risk_budget") is None else float(data["risk_budget"]),
            current_drawdown=float(data.get("current_drawdown", 0.0)),
        )


@dataclass(frozen=True)
class PositionSnapshot:
    """Current fund-level paper positions and cash reserve."""

    as_of_utc: str
    updated_at_utc: str
    currency: str
    total_nav_usd: float
    cash_usd: float
    strategy_notional_usd: Dict[str, float]

    def validate(self) -> None:
        parse_utc(self.as_of_utc)
        parse_utc(self.updated_at_utc)
        if self.currency != "USD":
            raise ValidationError("PositionSnapshot currency must be USD")
        nav = _check_finite("total_nav_usd", self.total_nav_usd)
        cash = _check_finite("cash_usd", self.cash_usd)
        if nav <= 0.0:
            raise ValidationError("total_nav_usd must be > 0", context={"total_nav_usd": nav})
        if cash < -1e-6:
            raise ValidationError("cash_usd cannot be negative", context={"cash_usd": cash})
        total = cash
        for sid, notional in self.strategy_notional_usd.items():
            val = _check_finite(f"strategy_notional_usd[{sid}]", notional)
            if val < -1e-6:
                raise ValidationError(
                    "Strategy notional cannot be negative in long-only capital budget",
                    context={"strategy_id": sid, "notional_usd": val},
                )
            total += val
        if abs(total - nav) > max(1.0, 1e-4 * nav):
            raise ValidationError(
                "Position snapshot cash + strategy notionals do not reconcile to total_nav_usd",
                context={"sum_usd": total, "total_nav_usd": nav, "diff_usd": total - nav},
            )

    def current_weights(self, strategy_ids: List[str]) -> Tuple[Dict[str, float], float]:
        weights = {
            sid: float(self.strategy_notional_usd.get(sid, 0.0)) / float(self.total_nav_usd)
            for sid in strategy_ids
        }
        cash_w = float(self.cash_usd) / float(self.total_nav_usd)
        return weights, cash_w

    def to_dict(self) -> Dict[str, Any]:
        return {
            "as_of_utc": format_utc(parse_utc(self.as_of_utc)),
            "updated_at_utc": format_utc(parse_utc(self.updated_at_utc)),
            "currency": self.currency,
            "total_nav_usd": float(self.total_nav_usd),
            "cash_usd": float(self.cash_usd),
            "strategy_notional_usd": {
                k: float(v) for k, v in sorted(self.strategy_notional_usd.items())
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PositionSnapshot:
        return cls(
            as_of_utc=str(data["as_of_utc"]),
            updated_at_utc=str(data["updated_at_utc"]),
            currency=str(data.get("currency", "USD")),
            total_nav_usd=float(data["total_nav_usd"]),
            cash_usd=float(data["cash_usd"]),
            strategy_notional_usd={
                str(k): float(v) for k, v in data.get("strategy_notional_usd", {}).items()
            },
        )


@dataclass(frozen=True)
class FundSnapshot:
    """Versioned point-in-time fund input snapshot published to allocators."""

    snapshot_id: str
    as_of_utc: str
    published_at_utc: str
    provenance: Provenance
    positions: PositionSnapshot
    strategies: Tuple[StrategySnapshot, ...]
    schema_version: str = SCHEMA_VERSION
    external_covariance: Optional[List[List[float]]] = None
    external_covariance_ids: Optional[Tuple[str, ...]] = None

    def validate(self) -> None:
        if not self.snapshot_id:
            raise ValidationError("snapshot_id must be non-empty")
        parse_utc(self.as_of_utc)
        parse_utc(self.published_at_utc)
        self.provenance.validate()
        self.positions.validate()
        if not self.strategies:
            raise ValidationError("FundSnapshot must contain at least one strategy")
        seen = set()
        for strat in self.strategies:
            strat.validate()
            if strat.strategy_id in seen:
                raise ValidationError(
                    f"Duplicate strategy_id in snapshot: {strat.strategy_id}",
                    context={"strategy_id": strat.strategy_id},
                )
            seen.add(strat.strategy_id)
        if self.external_covariance is not None:
            n = len(self.external_covariance)
            if self.external_covariance_ids is None or len(self.external_covariance_ids) != n:
                raise ValidationError("external_covariance_ids must match external_covariance dimension")
            for row in self.external_covariance:
                if len(row) != n:
                    raise ValidationError("external_covariance must be square")
                for val in row:
                    _check_finite("external_covariance", val)

    def content_hash(self) -> str:
        return sha256_hex(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "snapshot_id": self.snapshot_id,
            "schema_version": self.schema_version,
            "as_of_utc": format_utc(parse_utc(self.as_of_utc)),
            "published_at_utc": format_utc(parse_utc(self.published_at_utc)),
            "provenance": self.provenance.to_dict(),
            "positions": self.positions.to_dict(),
            "strategies": [s.to_dict() for s in self.strategies],
        }
        if self.external_covariance is not None:
            out["external_covariance"] = [[float(v) for v in row] for row in self.external_covariance]
            out["external_covariance_ids"] = list(self.external_covariance_ids or ())
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FundSnapshot:
        ext_cov = data.get("external_covariance")
        ext_ids = data.get("external_covariance_ids")
        return cls(
            snapshot_id=str(data["snapshot_id"]),
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            as_of_utc=str(data["as_of_utc"]),
            published_at_utc=str(data["published_at_utc"]),
            provenance=Provenance.from_dict(data["provenance"]),
            positions=PositionSnapshot.from_dict(data["positions"]),
            strategies=tuple(StrategySnapshot.from_dict(s) for s in data["strategies"]),
            external_covariance=None if ext_cov is None else [[float(x) for x in r] for r in ext_cov],
            external_covariance_ids=None if ext_ids is None else tuple(str(x) for x in ext_ids),
        )


@dataclass(frozen=True)
class ConstraintSpec:
    """Explicit capital, exposure, liquidity, turnover, and risk constraints."""

    cash_reserve_min: float = 0.05
    cash_reserve_max: float = 0.50
    default_min_weight: float = 0.0
    default_max_weight: float = 0.35
    max_one_way_turnover: float = 0.25
    max_annualized_volatility: Optional[float] = 0.14
    max_stressed_annualized_volatility: Optional[float] = 0.20
    max_adv_participation: float = 0.10
    max_gross_leverage: float = 1.0
    group_caps: Dict[str, float] = field(default_factory=dict)
    strategy_min_weights: Dict[str, float] = field(default_factory=dict)
    strategy_max_weights: Dict[str, float] = field(default_factory=dict)
    constraint_tolerance: float = 1e-6

    def validate(self) -> None:
        cmin = _check_finite("cash_reserve_min", self.cash_reserve_min)
        cmax = _check_finite("cash_reserve_max", self.cash_reserve_max)
        if not (0.0 <= cmin <= cmax <= 1.0):
            raise ValidationError(
                "cash_reserve_min and cash_reserve_max must satisfy 0 <= min <= max <= 1",
                context={"cash_reserve_min": cmin, "cash_reserve_max": cmax},
            )
        dmin = _check_finite("default_min_weight", self.default_min_weight)
        dmax = _check_finite("default_max_weight", self.default_max_weight)
        if not (0.0 <= dmin <= dmax <= 1.0):
            raise ValidationError("default weight bounds must satisfy 0 <= min <= max <= 1")
        t = _check_finite("max_one_way_turnover", self.max_one_way_turnover)
        if t < 0.0 or t > 2.0:
            raise ValidationError("max_one_way_turnover must be in [0, 2]")
        adv = _check_finite("max_adv_participation", self.max_adv_participation)
        if adv <= 0.0 or adv > 1.0:
            raise ValidationError("max_adv_participation must be in (0, 1]")
        tol = _check_finite("constraint_tolerance", self.constraint_tolerance)
        if tol <= 0.0 or tol > 1e-2:
            raise ValidationError("constraint_tolerance must be in (0, 1e-2]")
        for grp, cap in self.group_caps.items():
            v = _check_finite(f"group_caps[{grp}]", cap)
            if v < 0.0 or v > 1.0:
                raise ValidationError(f"group_caps[{grp}] must be in [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cash_reserve_min": float(self.cash_reserve_min),
            "cash_reserve_max": float(self.cash_reserve_max),
            "default_min_weight": float(self.default_min_weight),
            "default_max_weight": float(self.default_max_weight),
            "max_one_way_turnover": float(self.max_one_way_turnover),
            "max_annualized_volatility": self.max_annualized_volatility,
            "max_stressed_annualized_volatility": self.max_stressed_annualized_volatility,
            "max_adv_participation": float(self.max_adv_participation),
            "max_gross_leverage": float(self.max_gross_leverage),
            "group_caps": {k: float(v) for k, v in sorted(self.group_caps.items())},
            "strategy_min_weights": {k: float(v) for k, v in sorted(self.strategy_min_weights.items())},
            "strategy_max_weights": {k: float(v) for k, v in sorted(self.strategy_max_weights.items())},
            "constraint_tolerance": float(self.constraint_tolerance),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ConstraintSpec:
        return cls(
            cash_reserve_min=float(data.get("cash_reserve_min", 0.05)),
            cash_reserve_max=float(data.get("cash_reserve_max", 0.50)),
            default_min_weight=float(data.get("default_min_weight", 0.0)),
            default_max_weight=float(data.get("default_max_weight", 0.35)),
            max_one_way_turnover=float(data.get("max_one_way_turnover", 0.25)),
            max_annualized_volatility=(
                None
                if data.get("max_annualized_volatility") is None
                else float(data["max_annualized_volatility"])
            ),
            max_stressed_annualized_volatility=(
                None
                if data.get("max_stressed_annualized_volatility") is None
                else float(data["max_stressed_annualized_volatility"])
            ),
            max_adv_participation=float(data.get("max_adv_participation", 0.10)),
            max_gross_leverage=float(data.get("max_gross_leverage", 1.0)),
            group_caps={str(k): float(v) for k, v in data.get("group_caps", {}).items()},
            strategy_min_weights={
                str(k): float(v) for k, v in data.get("strategy_min_weights", {}).items()
            },
            strategy_max_weights={
                str(k): float(v) for k, v in data.get("strategy_max_weights", {}).items()
            },
            constraint_tolerance=float(data.get("constraint_tolerance", 1e-6)),
        )


@dataclass(frozen=True)
class EngineConfig:
    """Versioned engine configuration; contains no secrets."""

    config_version: str = "1.0.0"
    environment: str = "local_prototype"
    base_currency: str = "USD"
    # Execution policy
    live_execution_enabled: bool = False
    paper_execution_enabled: bool = True
    require_explicit_approval: bool = True
    enforce_dual_control: bool = True
    proposal_ttl_minutes: int = 120
    # Data quality
    min_history_observations: int = 20
    max_position_staleness_hours: float = 24.0
    max_return_staleness_hours: float = 72.0
    zero_volatility_eps: float = 1e-8
    zero_volatility_policy: str = "exclude"  # "exclude" | "reject" | "floor"
    insufficient_history_policy: str = "exclude"  # "exclude" | "reject"
    # Estimator
    annualization_factor: int = 252
    return_shrinkage: float = 0.25
    covariance_shrinkage: float = 0.15
    non_psd_policy: str = "repair"  # "repair" | "reject"
    psd_min_eigenvalue: float = 1e-7
    correlation_stress_enabled: bool = False
    correlation_stress_blend: float = 0.35
    correlation_stress_floor: float = 0.65
    drawdown_stress_trigger: float = -0.05
    # Optimizer
    risk_aversion: float = 6.0
    turnover_l2_penalty: float = 0.5
    default_transaction_cost_bps: float = 10.0
    max_iterations: int = 600
    convergence_tol: float = 1e-8
    fallback_policy: str = "no_change_then_cash"  # "no_change_then_cash" | "cash_only"
    solver_backend: str = "builtin"
    # Constraints
    constraints: ConstraintSpec = field(default_factory=ConstraintSpec)

    def validate(self) -> None:
        if self.live_execution_enabled:
            raise ValidationError(
                "live_execution_enabled must remain False in MVP; live reallocation requires separate approval"
            )
        if not self.require_explicit_approval:
            raise ValidationError("require_explicit_approval cannot be disabled")
        if self.non_psd_policy not in ("repair", "reject"):
            raise ValidationError("non_psd_policy must be 'repair' or 'reject'")
        if self.zero_volatility_policy not in ("exclude", "reject", "floor"):
            raise ValidationError("zero_volatility_policy must be 'exclude', 'reject', or 'floor'")
        if self.insufficient_history_policy not in ("exclude", "reject"):
            raise ValidationError("insufficient_history_policy must be 'exclude' or 'reject'")
        if self.fallback_policy not in ("no_change_then_cash", "cash_only"):
            raise ValidationError("fallback_policy must be 'no_change_then_cash' or 'cash_only'")
        self.constraints.validate()

    def config_hash(self) -> str:
        return sha256_hex(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_version": self.config_version,
            "environment": self.environment,
            "base_currency": self.base_currency,
            "execution_policy": {
                "live_execution_enabled": self.live_execution_enabled,
                "paper_execution_enabled": self.paper_execution_enabled,
                "require_explicit_approval": self.require_explicit_approval,
                "enforce_dual_control": self.enforce_dual_control,
                "proposal_ttl_minutes": self.proposal_ttl_minutes,
            },
            "data_quality": {
                "min_history_observations": self.min_history_observations,
                "max_position_staleness_hours": self.max_position_staleness_hours,
                "max_return_staleness_hours": self.max_return_staleness_hours,
                "zero_volatility_eps": self.zero_volatility_eps,
                "zero_volatility_policy": self.zero_volatility_policy,
                "insufficient_history_policy": self.insufficient_history_policy,
            },
            "estimator": {
                "annualization_factor": self.annualization_factor,
                "return_shrinkage": self.return_shrinkage,
                "covariance_shrinkage": self.covariance_shrinkage,
                "non_psd_policy": self.non_psd_policy,
                "psd_min_eigenvalue": self.psd_min_eigenvalue,
                "correlation_stress_enabled": self.correlation_stress_enabled,
                "correlation_stress_blend": self.correlation_stress_blend,
                "correlation_stress_floor": self.correlation_stress_floor,
                "drawdown_stress_trigger": self.drawdown_stress_trigger,
            },
            "optimizer": {
                "risk_aversion": self.risk_aversion,
                "turnover_l2_penalty": self.turnover_l2_penalty,
                "default_transaction_cost_bps": self.default_transaction_cost_bps,
                "max_iterations": self.max_iterations,
                "convergence_tol": self.convergence_tol,
                "fallback_policy": self.fallback_policy,
                "solver_backend": self.solver_backend,
            },
            "constraints": self.constraints.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EngineConfig:
        ep = data.get("execution_policy", {})
        dq = data.get("data_quality", {})
        est = data.get("estimator", {})
        opt = data.get("optimizer", {})
        con = data.get("constraints", {})
        cfg = cls(
            config_version=str(data.get("config_version", "1.0.0")),
            environment=str(data.get("environment", "local_prototype")),
            base_currency=str(data.get("base_currency", "USD")),
            live_execution_enabled=bool(ep.get("live_execution_enabled", False)),
            paper_execution_enabled=bool(ep.get("paper_execution_enabled", True)),
            require_explicit_approval=bool(ep.get("require_explicit_approval", True)),
            enforce_dual_control=bool(ep.get("enforce_dual_control", True)),
            proposal_ttl_minutes=int(ep.get("proposal_ttl_minutes", 120)),
            min_history_observations=int(dq.get("min_history_observations", 20)),
            max_position_staleness_hours=float(dq.get("max_position_staleness_hours", 24.0)),
            max_return_staleness_hours=float(dq.get("max_return_staleness_hours", 72.0)),
            zero_volatility_eps=float(dq.get("zero_volatility_eps", 1e-8)),
            zero_volatility_policy=str(dq.get("zero_volatility_policy", "exclude")),
            insufficient_history_policy=str(dq.get("insufficient_history_policy", "exclude")),
            annualization_factor=int(est.get("annualization_factor", 252)),
            return_shrinkage=float(est.get("return_shrinkage", 0.25)),
            covariance_shrinkage=float(est.get("covariance_shrinkage", 0.15)),
            non_psd_policy=str(est.get("non_psd_policy", "repair")),
            psd_min_eigenvalue=float(est.get("psd_min_eigenvalue", 1e-7)),
            correlation_stress_enabled=bool(est.get("correlation_stress_enabled", False)),
            correlation_stress_blend=float(est.get("correlation_stress_blend", 0.35)),
            correlation_stress_floor=float(est.get("correlation_stress_floor", 0.65)),
            drawdown_stress_trigger=float(est.get("drawdown_stress_trigger", -0.05)),
            risk_aversion=float(opt.get("risk_aversion", 6.0)),
            turnover_l2_penalty=float(opt.get("turnover_l2_penalty", 0.5)),
            default_transaction_cost_bps=float(opt.get("default_transaction_cost_bps", 10.0)),
            max_iterations=int(opt.get("max_iterations", 600)),
            convergence_tol=float(opt.get("convergence_tol", 1e-8)),
            fallback_policy=str(opt.get("fallback_policy", "no_change_then_cash")),
            solver_backend=str(opt.get("solver_backend", "builtin")),
            constraints=ConstraintSpec.from_dict(con),
        )
        cfg.validate()
        return cfg

    @classmethod
    def from_json_file(cls, path: str) -> EngineConfig:
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


@dataclass(frozen=True)
class ObjectiveBreakdown:
    """Explicit objective decomposition exposed on every allocation decision."""

    expected_return_annual: float
    portfolio_volatility_annual: float
    stressed_volatility_annual: float
    variance_penalty: float
    transaction_cost_drag: float
    turnover_l2_penalty: float
    net_objective: float
    one_way_turnover: float
    estimated_cost_usd: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_return_annual": float(self.expected_return_annual),
            "portfolio_volatility_annual": float(self.portfolio_volatility_annual),
            "stressed_volatility_annual": float(self.stressed_volatility_annual),
            "variance_penalty": float(self.variance_penalty),
            "transaction_cost_drag": float(self.transaction_cost_drag),
            "turnover_l2_penalty": float(self.turnover_l2_penalty),
            "net_objective": float(self.net_objective),
            "one_way_turnover": float(self.one_way_turnover),
            "estimated_cost_usd": float(self.estimated_cost_usd),
        }


@dataclass(frozen=True)
class ConstraintCheckResult:
    """Result of one independent post-solve constraint check."""

    name: str
    passed: bool
    actual: float
    limit: float
    slack: float
    unit: str
    severity: str  # "hard" | "warning"
    message: str
    scope: str = "fund"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "actual": float(self.actual),
            "limit": float(self.limit),
            "slack": float(self.slack),
            "unit": self.unit,
            "severity": self.severity,
            "message": self.message,
            "scope": self.scope,
        }


@dataclass(frozen=True)
class FeasibilityReport:
    """Complete independent verification report for an allocation decision."""

    feasible: bool
    hard_violations: Tuple[ConstraintCheckResult, ...]
    warnings: Tuple[ConstraintCheckResult, ...]
    checks: Tuple[ConstraintCheckResult, ...]
    allocation_sum: float
    cash_weight: float
    reconciliation_error: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feasible": self.feasible,
            "hard_violation_count": len(self.hard_violations),
            "warning_count": len(self.warnings),
            "allocation_sum": float(self.allocation_sum),
            "cash_weight": float(self.cash_weight),
            "reconciliation_error": float(self.reconciliation_error),
            "hard_violations": [c.to_dict() for c in self.hard_violations],
            "warnings": [c.to_dict() for c in self.warnings],
            "checks": [c.to_dict() for c in self.checks],
        }


@dataclass(frozen=True)
class AllocationDecision:
    """Full transparent output of an allocator run."""

    decision_id: str
    allocator_name: str
    snapshot_id: str
    snapshot_hash: str
    config_version: str
    config_hash: str
    as_of_utc: str
    status: AllocationStatus
    target_weights: Dict[str, float]
    cash_weight: float
    current_weights: Dict[str, float]
    current_cash_weight: float
    objective: ObjectiveBreakdown
    inputs_summary: Dict[str, Any]
    diagnostics: Dict[str, Any]
    infeasibility_reasons: Tuple[str, ...] = ()
    fallback_applied: Optional[str] = None
    feasibility_report: Optional[FeasibilityReport] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "allocator_name": self.allocator_name,
            "snapshot_id": self.snapshot_id,
            "snapshot_hash": self.snapshot_hash,
            "config_version": self.config_version,
            "config_hash": self.config_hash,
            "as_of_utc": self.as_of_utc,
            "status": self.status.value,
            "target_weights": {k: float(v) for k, v in sorted(self.target_weights.items())},
            "cash_weight": float(self.cash_weight),
            "current_weights": {k: float(v) for k, v in sorted(self.current_weights.items())},
            "current_cash_weight": float(self.current_cash_weight),
            "objective": self.objective.to_dict(),
            "inputs_summary": self.inputs_summary,
            "diagnostics": self.diagnostics,
            "infeasibility_reasons": list(self.infeasibility_reasons),
            "fallback_applied": self.fallback_applied,
            "feasibility_report": (
                None if self.feasibility_report is None else self.feasibility_report.to_dict()
            ),
        }
