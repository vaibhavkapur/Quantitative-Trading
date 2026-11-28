"""Configuration dataclasses, strict YAML loading, validation, and hashing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from cross_asset_factor_engine import SCHEMA_VERSION
from cross_asset_factor_engine.errors import ConfigurationError

SUPPORTED_ASSET_CLASSES = ("equity", "fx", "commodity")
SUPPORTED_FACTORS = ("momentum", "carry", "defensive")


@dataclass(frozen=True)
class DataConfig:
    dataset_dir: str = "data/fixtures/synthetic_multi_year"
    strict_missing_factors: bool = False
    defensive_fallback_to_low_vol: bool = True


@dataclass(frozen=True)
class MomentumConfig:
    lookback_days: int = 126
    skip_days: int = 21
    vol_lookback_days: int = 63
    min_history_days: int = 126


@dataclass(frozen=True)
class CarryConfig:
    annualization_days: float = 365.0
    max_staleness_days: int = 120


@dataclass(frozen=True)
class DefensiveConfig:
    vol_lookback_days: int = 63
    min_history_days: int = 42
    equity_quality_weight: float = 0.5
    max_staleness_days: int = 120


@dataclass(frozen=True)
class ScalingConfig:
    winsorize_z: float = 3.0
    min_std: float = 1.0e-6


@dataclass(frozen=True)
class FactorsConfig:
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    carry: CarryConfig = field(default_factory=CarryConfig)
    defensive: DefensiveConfig = field(default_factory=DefensiveConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    blend_weights: dict[str, float] = field(
        default_factory=lambda: {
            "momentum": 1.0 / 3.0,
            "carry": 1.0 / 3.0,
            "defensive": 1.0 / 3.0,
        }
    )


@dataclass(frozen=True)
class PortfolioConstraints:
    max_asset_weight: float = 0.30
    max_gross_exposure: float = 1.60
    min_net_exposure: float = -0.50
    max_net_exposure: float = 0.50
    max_turnover_per_rebalance: float = 0.35
    max_class_gross_exposure: dict[str, float] = field(
        default_factory=lambda: {"equity": 0.80, "fx": 0.80, "commodity": 0.80}
    )


@dataclass(frozen=True)
class PortfolioConfig:
    target_volatility: float = 0.10
    volatility_cap: float = 0.12
    cov_lookback_days: int = 63
    cov_shrinkage: float = 0.20
    min_asset_vol: float = 0.02
    rebalance_frequency_days: int = 5
    execution_lag_days: int = 1
    turnover_penalty: float = 0.005
    constraints: PortfolioConstraints = field(default_factory=PortfolioConstraints)
    enforce_strict_constraints: bool = True
    constraint_tolerance: float = 1.0e-5


@dataclass(frozen=True)
class WalkForwardConfig:
    train_window_days: int = 252
    test_window_days: int = 126
    step_days: int = 126
    embargo_days: int = 1
    anchored: bool = False
    annualization_days: int = 252


@dataclass(frozen=True)
class ReportingConfig:
    output_dir: str = "reports/runs"
    write_detailed_weights: bool = True
    write_signal_audit: bool = True


@dataclass(frozen=True)
class EngineConfig:
    schema_version: str = SCHEMA_VERSION
    config_name: str = "default_cross_asset_walkforward"
    seed: int = 42
    base_currency: str = "USD"
    decision_cutoff_utc_time: str = "21:00:00Z"
    data: DataConfig = field(default_factory=DataConfig)
    factors: FactorsConfig = field(default_factory=FactorsConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _require_mapping(obj: Any, section: str) -> dict[str, Any]:
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ConfigurationError(
            f"Configuration section '{section}' must be a mapping.",
            details={"section": section, "actual_type": type(obj).__name__},
        )
    return obj


def _check_keys(raw: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(raw.keys()) - allowed)
    if unknown:
        raise ConfigurationError(
            f"Unknown configuration keys in '{section}': {unknown}",
            details={"section": section, "unknown_keys": unknown},
        )


def validate_config(cfg: EngineConfig) -> EngineConfig:
    """Validate numerical and logical invariants across all config sections."""
    if cfg.schema_version != SCHEMA_VERSION:
        raise ConfigurationError(
            f"Unsupported config schema_version '{cfg.schema_version}'; expected '{SCHEMA_VERSION}'.",
            details={"schema_version": cfg.schema_version, "expected": SCHEMA_VERSION},
        )
    if not cfg.base_currency or len(cfg.base_currency) != 3 or not cfg.base_currency.isupper():
        raise ConfigurationError(
            f"base_currency must be a 3-letter uppercase ISO currency code, got '{cfg.base_currency}'."
        )
    if not cfg.decision_cutoff_utc_time.endswith("Z"):
        raise ConfigurationError(
            "decision_cutoff_utc_time must be an ISO UTC time ending with 'Z' (e.g. '21:00:00Z')."
        )

    m = cfg.factors.momentum
    if m.lookback_days <= m.skip_days or m.skip_days < 0 or m.vol_lookback_days < 2:
        raise ConfigurationError(
            "Invalid momentum configuration: require lookback_days > skip_days >= 0 and vol_lookback_days >= 2.",
            details=asdict(m),
        )
    if m.min_history_days < m.lookback_days:
        raise ConfigurationError(
            "momentum.min_history_days must be >= momentum.lookback_days.",
            details=asdict(m),
        )

    c = cfg.factors.carry
    if c.annualization_days <= 0 or c.max_staleness_days < 0:
        raise ConfigurationError("Invalid carry configuration.", details=asdict(c))

    d = cfg.factors.defensive
    if d.vol_lookback_days < 2 or d.min_history_days < 2 or not (0.0 <= d.equity_quality_weight <= 1.0):
        raise ConfigurationError("Invalid defensive configuration.", details=asdict(d))

    s = cfg.factors.scaling
    if s.winsorize_z <= 0 or s.min_std <= 0:
        raise ConfigurationError("Invalid scaling configuration.", details=asdict(s))

    bw = cfg.factors.blend_weights
    if set(bw.keys()) != set(SUPPORTED_FACTORS):
        raise ConfigurationError(
            f"blend_weights must specify exact keys {SUPPORTED_FACTORS}, got {sorted(bw.keys())}."
        )
    if any(v < 0 for v in bw.values()) or sum(bw.values()) <= 0:
        raise ConfigurationError("blend_weights must be non-negative with positive sum.", details=bw)

    p = cfg.portfolio
    if p.target_volatility <= 0 or p.volatility_cap <= 0:
        raise ConfigurationError(
            "Portfolio target_volatility and volatility_cap must be strictly positive.",
            details={"target_volatility": p.target_volatility, "volatility_cap": p.volatility_cap},
        )
    if p.target_volatility > p.volatility_cap + 1e-12:
        raise ConfigurationError(
            "Portfolio target_volatility cannot exceed volatility_cap.",
            details={"target_volatility": p.target_volatility, "volatility_cap": p.volatility_cap},
        )
    if p.cov_lookback_days < 2 or not (0.0 <= p.cov_shrinkage <= 1.0) or p.min_asset_vol <= 0:
        raise ConfigurationError("Invalid portfolio covariance parameters.")
    if p.rebalance_frequency_days < 1 or p.execution_lag_days < 1:
        raise ConfigurationError(
            "rebalance_frequency_days and execution_lag_days must be >= 1 (strictly no same-day look-ahead)."
        )
    if p.turnover_penalty < 0 or p.constraint_tolerance <= 0:
        raise ConfigurationError("turnover_penalty must be >= 0 and constraint_tolerance > 0.")

    pc = p.constraints
    if pc.max_asset_weight <= 0 or pc.max_gross_exposure <= 0 or pc.max_turnover_per_rebalance <= 0:
        raise ConfigurationError(
            "Exposure and turnover caps must be strictly positive.",
            details=asdict(pc),
        )
    if pc.min_net_exposure > pc.max_net_exposure:
        raise ConfigurationError(
            "min_net_exposure cannot exceed max_net_exposure.",
            details={"min_net_exposure": pc.min_net_exposure, "max_net_exposure": pc.max_net_exposure},
        )
    if abs(pc.min_net_exposure) > pc.max_gross_exposure and abs(pc.max_net_exposure) > pc.max_gross_exposure:
        if (pc.min_net_exposure > pc.max_gross_exposure) or (pc.max_net_exposure < -pc.max_gross_exposure):
            raise ConfigurationError(
                "Net exposure band is disjoint from [-max_gross_exposure, max_gross_exposure].",
                details=asdict(pc),
            )
    for cls_name, cap in pc.max_class_gross_exposure.items():
        if cls_name not in SUPPORTED_ASSET_CLASSES:
            raise ConfigurationError(f"Unsupported asset class in max_class_gross_exposure: '{cls_name}'.")
        if cap <= 0:
            raise ConfigurationError(
                f"max_class_gross_exposure['{cls_name}'] must be positive, got {cap}."
            )

    wf = cfg.walk_forward
    if wf.train_window_days < 2 or wf.test_window_days < 1 or wf.step_days < 1 or wf.embargo_days < 0:
        raise ConfigurationError("Invalid walk_forward window parameters.", details=asdict(wf))
    if wf.annualization_days <= 0:
        raise ConfigurationError("walk_forward.annualization_days must be > 0.")

    return cfg


def load_config(path: str | Path) -> EngineConfig:
    """Load and validate an EngineConfig from a YAML file."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {cfg_path}",
            details={"path": str(cfg_path)},
        )
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Malformed YAML in configuration file {cfg_path}: {exc}",
            details={"path": str(cfg_path)},
        ) from exc

    root = _require_mapping(raw, "root")
    if not root:
        raise ConfigurationError(
            f"Configuration file is empty: {cfg_path}",
            details={"path": str(cfg_path)},
        )
    _check_keys(
        root,
        {
            "schema_version",
            "config_name",
            "seed",
            "base_currency",
            "decision_cutoff_utc_time",
            "data",
            "factors",
            "portfolio",
            "walk_forward",
            "reporting",
        },
        "root",
    )

    data_raw = _require_mapping(root.get("data"), "data")
    _check_keys(data_raw, {"dataset_dir", "strict_missing_factors", "defensive_fallback_to_low_vol"}, "data")

    factors_raw = _require_mapping(root.get("factors"), "factors")
    _check_keys(factors_raw, {"momentum", "carry", "defensive", "scaling", "blend_weights"}, "factors")
    mom_raw = _require_mapping(factors_raw.get("momentum"), "factors.momentum")
    _check_keys(mom_raw, {"lookback_days", "skip_days", "vol_lookback_days", "min_history_days"}, "factors.momentum")
    carry_raw = _require_mapping(factors_raw.get("carry"), "factors.carry")
    _check_keys(carry_raw, {"annualization_days", "max_staleness_days"}, "factors.carry")
    def_raw = _require_mapping(factors_raw.get("defensive"), "factors.defensive")
    _check_keys(
        def_raw,
        {"vol_lookback_days", "min_history_days", "equity_quality_weight", "max_staleness_days"},
        "factors.defensive",
    )
    scale_raw = _require_mapping(factors_raw.get("scaling"), "factors.scaling")
    _check_keys(scale_raw, {"winsorize_z", "min_std"}, "factors.scaling")

    port_raw = _require_mapping(root.get("portfolio"), "portfolio")
    _check_keys(
        port_raw,
        {
            "target_volatility",
            "volatility_cap",
            "cov_lookback_days",
            "cov_shrinkage",
            "min_asset_vol",
            "rebalance_frequency_days",
            "execution_lag_days",
            "turnover_penalty",
            "constraints",
            "enforce_strict_constraints",
            "constraint_tolerance",
        },
        "portfolio",
    )
    cons_raw = _require_mapping(port_raw.get("constraints"), "portfolio.constraints")
    _check_keys(
        cons_raw,
        {
            "max_asset_weight",
            "max_gross_exposure",
            "min_net_exposure",
            "max_net_exposure",
            "max_turnover_per_rebalance",
            "max_class_gross_exposure",
        },
        "portfolio.constraints",
    )

    wf_raw = _require_mapping(root.get("walk_forward"), "walk_forward")
    _check_keys(
        wf_raw,
        {"train_window_days", "test_window_days", "step_days", "embargo_days", "anchored", "annualization_days"},
        "walk_forward",
    )

    rep_raw = _require_mapping(root.get("reporting"), "reporting")
    _check_keys(rep_raw, {"output_dir", "write_detailed_weights", "write_signal_audit"}, "reporting")

    try:
        blend_weights = factors_raw.get("blend_weights")
        factors_cfg = FactorsConfig(
            momentum=MomentumConfig(**mom_raw),
            carry=CarryConfig(**carry_raw),
            defensive=DefensiveConfig(**def_raw),
            scaling=ScalingConfig(**scale_raw),
            **({"blend_weights": {k: float(v) for k, v in blend_weights.items()}} if blend_weights is not None else {}),
        )
        port_kwargs = {k: v for k, v in port_raw.items() if k != "constraints"}
        if "max_class_gross_exposure" in cons_raw and cons_raw["max_class_gross_exposure"] is not None:
            cons_raw = {
                **cons_raw,
                "max_class_gross_exposure": {
                    k: float(v) for k, v in cons_raw["max_class_gross_exposure"].items()
                },
            }
        portfolio_cfg = PortfolioConfig(
            **port_kwargs,
            constraints=PortfolioConstraints(**cons_raw),
        )
        cfg = EngineConfig(
            schema_version=str(root.get("schema_version", SCHEMA_VERSION)),
            config_name=str(root.get("config_name", cfg_path.stem)),
            seed=int(root.get("seed", 42)),
            base_currency=str(root.get("base_currency", "USD")),
            decision_cutoff_utc_time=str(root.get("decision_cutoff_utc_time", "21:00:00Z")),
            data=DataConfig(**data_raw),
            factors=factors_cfg,
            portfolio=portfolio_cfg,
            walk_forward=WalkForwardConfig(**wf_raw),
            reporting=ReportingConfig(**rep_raw),
        )
    except TypeError as exc:
        raise ConfigurationError(f"Invalid configuration field types: {exc}") from exc

    return validate_config(cfg)
