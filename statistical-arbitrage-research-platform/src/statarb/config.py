"""Typed configuration schema, validation, and hashing."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from statarb.exceptions import ConfigurationError, UnsupportedInputError


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prices_path: str
    borrow_path: str
    use_log_prices: bool = True
    max_missing_fraction: float = Field(default=0.02, ge=0.0, le=1.0)
    missing_price_policy: Literal["raise", "drop_symbol"] = "raise"


class UniverseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    min_price: float = Field(default=5.0, gt=0.0)
    min_adv_usd: float = Field(default=1_000_000.0, ge=0.0)
    min_history_days: int = Field(default=120, ge=20)
    require_active: bool = True
    same_sector_only: bool = True


class SelectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal["engle_granger", "johansen_basket"] = "engle_granger"
    hedge_ratio_estimator: Literal["ols", "tls"] = "ols"
    max_adf_pvalue: float = Field(default=0.05, gt=0.0, lt=1.0)
    fdr_alpha: float = Field(default=0.10, gt=0.0, lt=1.0)
    multiple_testing_method: Literal["bh", "bonferroni", "none"] = "bh"
    min_half_life_days: float = Field(default=2.0, gt=0.0)
    max_half_life_days: float = Field(default=40.0, gt=0.0)
    min_correlation: float = Field(default=0.50, ge=-1.0, le=1.0)
    max_pairs_per_fold: int = Field(default=4, ge=1)
    include_baskets: bool = True
    basket_size: int = Field(default=3, ge=3, le=5)
    max_baskets_per_fold: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def validate_half_life_bounds(self) -> SelectionConfig:
        if self.min_half_life_days >= self.max_half_life_days:
            raise ValueError("min_half_life_days must be strictly less than max_half_life_days")
        return self


class SignalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    z_entry: float = Field(default=1.75, gt=0.0)
    z_exit: float = Field(default=0.25, ge=0.0)
    z_stop: float = Field(default=3.50, gt=0.0)
    max_holding_days: int = Field(default=30, ge=1)
    zscore_mode: Literal["training_frozen", "causal_rolling"] = "training_frozen"
    zscore_lookback_days: int = Field(default=60, ge=5)

    @model_validator(mode="after")
    def validate_thresholds(self) -> SignalConfig:
        if not (self.z_exit < self.z_entry < self.z_stop):
            raise ValueError("Signal thresholds must satisfy 0 <= z_exit < z_entry < z_stop")
        return self


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    initial_capital: float = Field(default=1_000_000.0, gt=0.0)
    gross_leverage_target: float = Field(default=1.0, gt=0.0, le=5.0)
    max_pair_weight: float = Field(default=0.35, gt=0.0, le=1.0)
    commission_bps: float = Field(default=1.5, ge=0.0)
    slippage_bps: float = Field(default=3.0, ge=0.0)
    default_borrow_fee_bps_annual: float = Field(default=50.0, ge=0.0)
    enforce_borrow_availability: bool = True
    strict_borrow_errors: bool = False
    recall_on_borrow_loss: bool = True
    execution_delay_days: int = Field(default=1, ge=1, le=5)


class WalkForwardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    train_window_days: int = Field(default=180, ge=30)
    test_window_days: int = Field(default=60, ge=5)
    step_days: int = Field(default=60, ge=1)
    embargo_days: int = Field(default=1, ge=0)
    window_type: Literal["rolling", "expanding"] = "rolling"


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    annualization_days: int = Field(default=252, ge=1)
    bootstrap_samples: int = Field(default=500, ge=50)
    bootstrap_block_size: int = Field(default=10, ge=1)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    risk_free_rate_annual: float = Field(default=0.0)
    sweep_z_entries: list[float] = Field(default_factory=lambda: [1.5, 1.75, 2.0, 2.25])


class PlatformConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0.0"
    run_name: str = "default_run"
    seed: int = 42
    data: DataConfig
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    signals: SignalConfig = Field(default_factory=SignalConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @model_validator(mode="after")
    def validate_cross_section(self) -> PlatformConfig:
        if self.walk_forward.train_window_days < self.universe.min_history_days:
            raise ValueError("walk_forward.train_window_days must be >= universe.min_history_days")
        for z in self.evaluation.sweep_z_entries:
            if not (self.signals.z_exit < z < self.signals.z_stop):
                raise ValueError(f"sweep_z_entries value {z} must satisfy z_exit < z < z_stop")
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def config_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def load_config(path: str | Path) -> PlatformConfig:
    """Load and validate a PlatformConfig from a TOML or JSON file."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {cfg_path}. "
            "Provide a valid .toml or .json configuration path."
        )
    suffix = cfg_path.suffix.lower()
    try:
        raw_bytes = cfg_path.read_bytes()
        if suffix == ".toml":
            data = tomllib.loads(raw_bytes.decode("utf-8"))
        elif suffix == ".json":
            data = json.loads(raw_bytes.decode("utf-8"))
        else:
            raise UnsupportedInputError(
                f"Unsupported configuration format '{suffix}'. Supported formats: .toml, .json"
            )
    except UnsupportedInputError:
        raise
    except Exception as exc:
        raise ConfigurationError(f"Failed to parse configuration '{cfg_path}': {exc}") from exc

    return validate_config_dict(data)


def validate_config_dict(data: dict[str, Any]) -> PlatformConfig:
    """Validate a raw dictionary into a PlatformConfig."""
    try:
        return PlatformConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid configuration: {exc}") from exc
