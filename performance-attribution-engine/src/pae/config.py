"""Versioned configuration loader with fail-safe operational guardrails."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from pae.errors import ConfigurationError, OrderExecutionProhibitedError

VALID_DAILY_METHODS = {"TWR_SUBPERIOD"}
VALID_LINKING_METHODS = {"CARINO", "GEOMETRIC_PROPORTIONAL"}
VALID_CF_TIMINGS = {"START_OF_DAY", "END_OF_DAY"}
VALID_BRINSON_MODELS = {"BRINSON_FACHLER", "BRINSON_HOOD_BEEBOWER"}
VALID_ENV_MODES = {"SIMULATED_PROTOTYPE", "PAPER_SANDBOX"}


@dataclass(frozen=True)
class AttributionConfig:
    """Immutable, versioned configuration for an attribution run."""

    config_version: str = "1.0.0"
    methodology_version: str = "pae-v1.0.0"
    reporting_currency: str = "USD"

    # Return conventions
    daily_method: str = "TWR_SUBPERIOD"
    multi_period_linking: str = "CARINO"
    default_cash_flow_timing: str = "START_OF_DAY"
    compute_modified_dietz_crosscheck: bool = True
    gross_vs_net_strict_separation: bool = True

    # Brinson
    brinson_model: str = "BRINSON_FACHLER"
    require_aligned_benchmark_dates: bool = True
    require_aligned_benchmark_currency: bool = True

    # Factor attribution
    factor_enabled: bool = True
    factor_min_observations: int = 5
    factor_ridge_regularization: float = 1e-10
    non_causation_disclaimer: str = (
        "Factor attribution measures statistical co-movement and exposure decomposition only; "
        "factor correlation does not prove causation."
    )

    # Execution cost (TCA)
    tca_default_benchmark: str = "ARRIVAL_PRICE"
    tca_include_opportunity_cost: bool = True
    tca_include_explicit_fees: bool = True

    # Reconciliation
    return_tolerance: float = 1e-6
    return_tolerance_bps: float = 0.01
    nav_bridge_tolerance_abs: float = 1e-4
    fx_max_staleness_hours: float = 48.0
    fail_on_zero_nav: bool = False
    emit_exception_on_residual_breach: bool = True

    # Operational controls
    environment_mode: str = "SIMULATED_PROTOTYPE"
    order_execution_enabled: bool = False
    require_dual_control_approval: bool = True
    audit_hash_chain_enabled: bool = True

    # Storage
    sqlite_path: str = "reports/pae_state.sqlite3"

    def __post_init__(self) -> None:
        if self.order_execution_enabled:
            raise OrderExecutionProhibitedError(
                "Order execution is strictly disabled in the Performance Attribution Engine.",
                details={"order_execution_enabled": True},
            )
        if not self.gross_vs_net_strict_separation:
            raise ConfigurationError(
                "gross_vs_net_strict_separation must remain True; gross returns may never be represented as net.",
                details={"gross_vs_net_strict_separation": False},
            )
        if len(self.reporting_currency) != 3 or not self.reporting_currency.isalpha():
            raise ConfigurationError(
                f"Invalid ISO-4217 reporting currency: {self.reporting_currency!r}"
            )
        if self.daily_method not in VALID_DAILY_METHODS:
            raise ConfigurationError(f"Unsupported daily_method: {self.daily_method}")
        if self.multi_period_linking not in VALID_LINKING_METHODS:
            raise ConfigurationError(
                f"Unsupported multi_period_linking: {self.multi_period_linking}"
            )
        if self.default_cash_flow_timing not in VALID_CF_TIMINGS:
            raise ConfigurationError(
                f"Unsupported default_cash_flow_timing: {self.default_cash_flow_timing}"
            )
        if self.brinson_model not in VALID_BRINSON_MODELS:
            raise ConfigurationError(f"Unsupported brinson_model: {self.brinson_model}")
        if self.environment_mode not in VALID_ENV_MODES:
            raise ConfigurationError(
                f"Unsupported environment_mode: {self.environment_mode!r}. "
                "Live capital deployment requires separate approval outside MVP scope."
            )
        if self.return_tolerance <= 0 or self.nav_bridge_tolerance_abs <= 0:
            raise ConfigurationError("Reconciliation tolerances must be strictly positive.")
        if self.factor_min_observations < 2:
            raise ConfigurationError("factor_min_observations must be >= 2.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_version": self.config_version,
            "methodology_version": self.methodology_version,
            "reporting_currency": self.reporting_currency,
            "return_convention": {
                "daily_method": self.daily_method,
                "multi_period_linking": self.multi_period_linking,
                "default_cash_flow_timing": self.default_cash_flow_timing,
                "compute_modified_dietz_crosscheck": self.compute_modified_dietz_crosscheck,
                "gross_vs_net_strict_separation": self.gross_vs_net_strict_separation,
            },
            "brinson": {
                "model": self.brinson_model,
                "require_aligned_benchmark_dates": self.require_aligned_benchmark_dates,
                "require_aligned_benchmark_currency": self.require_aligned_benchmark_currency,
            },
            "factor_model": {
                "enabled": self.factor_enabled,
                "min_observations": self.factor_min_observations,
                "ridge_regularization": self.factor_ridge_regularization,
                "non_causation_disclaimer": self.non_causation_disclaimer,
            },
            "tca": {
                "default_benchmark": self.tca_default_benchmark,
                "include_opportunity_cost": self.tca_include_opportunity_cost,
                "include_explicit_fees": self.tca_include_explicit_fees,
            },
            "reconciliation": {
                "return_tolerance": self.return_tolerance,
                "return_tolerance_bps": self.return_tolerance_bps,
                "nav_bridge_tolerance_abs": self.nav_bridge_tolerance_abs,
                "fx_max_staleness_hours": self.fx_max_staleness_hours,
                "fail_on_zero_nav": self.fail_on_zero_nav,
                "emit_exception_on_residual_breach": self.emit_exception_on_residual_breach,
            },
            "operational_controls": {
                "environment_mode": self.environment_mode,
                "order_execution_enabled": self.order_execution_enabled,
                "require_dual_control_approval": self.require_dual_control_approval,
                "audit_hash_chain_enabled": self.audit_hash_chain_enabled,
            },
            "storage": {
                "sqlite_path": self.sqlite_path,
            },
        }

    def config_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "AttributionConfig":
        rc = raw.get("return_convention", {})
        br = raw.get("brinson", {})
        fm = raw.get("factor_model", {})
        tca = raw.get("tca", {})
        rec = raw.get("reconciliation", {})
        ops = raw.get("operational_controls", {})
        st = raw.get("storage", {})
        return cls(
            config_version=raw.get("config_version", "1.0.0"),
            methodology_version=raw.get("methodology_version", "pae-v1.0.0"),
            reporting_currency=str(raw.get("reporting_currency", "USD")).upper(),
            daily_method=rc.get("daily_method", "TWR_SUBPERIOD"),
            multi_period_linking=rc.get("multi_period_linking", "CARINO"),
            default_cash_flow_timing=rc.get("default_cash_flow_timing", "START_OF_DAY"),
            compute_modified_dietz_crosscheck=rc.get("compute_modified_dietz_crosscheck", True),
            gross_vs_net_strict_separation=rc.get("gross_vs_net_strict_separation", True),
            brinson_model=br.get("model", "BRINSON_FACHLER"),
            require_aligned_benchmark_dates=br.get("require_aligned_benchmark_dates", True),
            require_aligned_benchmark_currency=br.get("require_aligned_benchmark_currency", True),
            factor_enabled=fm.get("enabled", True),
            factor_min_observations=int(fm.get("min_observations", 5)),
            factor_ridge_regularization=float(fm.get("ridge_regularization", 1e-10)),
            non_causation_disclaimer=fm.get(
                "non_causation_disclaimer",
                "Factor attribution measures statistical co-movement and exposure decomposition only; "
                "factor correlation does not prove causation.",
            ),
            tca_default_benchmark=tca.get("default_benchmark", "ARRIVAL_PRICE"),
            tca_include_opportunity_cost=tca.get("include_opportunity_cost", True),
            tca_include_explicit_fees=tca.get("include_explicit_fees", True),
            return_tolerance=float(rec.get("return_tolerance", 1e-6)),
            return_tolerance_bps=float(rec.get("return_tolerance_bps", 0.01)),
            nav_bridge_tolerance_abs=float(rec.get("nav_bridge_tolerance_abs", 1e-4)),
            fx_max_staleness_hours=float(rec.get("fx_max_staleness_hours", 48.0)),
            fail_on_zero_nav=bool(rec.get("fail_on_zero_nav", False)),
            emit_exception_on_residual_breach=bool(
                rec.get("emit_exception_on_residual_breach", True)
            ),
            environment_mode=ops.get("environment_mode", "SIMULATED_PROTOTYPE"),
            order_execution_enabled=bool(ops.get("order_execution_enabled", False)),
            require_dual_control_approval=bool(ops.get("require_dual_control_approval", True)),
            audit_hash_chain_enabled=bool(ops.get("audit_hash_chain_enabled", True)),
            sqlite_path=st.get("sqlite_path", "reports/pae_state.sqlite3"),
        )

    @classmethod
    def from_file(cls, path: Optional[Path] = None) -> "AttributionConfig":
        if path is None:
            path = Path(__file__).resolve().parents[2] / "config" / "default_config.json"
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)
