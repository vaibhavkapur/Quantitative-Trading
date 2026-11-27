"""Versioned configuration loaders and typed policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alpha_engine.errors import ExecutionDisabledError


class WalkForwardConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_splits: int = Field(ge=2, default=4)
    min_train_days: int = Field(ge=20, default=120)
    test_days: int = Field(ge=10, default=60)
    embargo_days: int = Field(ge=1, default=5)
    forward_horizon_days: int = Field(ge=1, default=5)
    expanding_window: bool = True


class NeutralizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    winsorize_lower_quantile: float = Field(ge=0.0, lt=0.5, default=0.02)
    winsorize_upper_quantile: float = Field(gt=0.5, le=1.0, default=0.98)
    sector_neutral: bool = True
    beta_neutral: bool = True
    vol_neutral: bool = True
    zscore_standardize: bool = True


class CostModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_spread_bps: float = Field(ge=0.0, default=5.0)
    impact_coeff_bps: float = Field(ge=0.0, default=10.0)
    cost_grid_bps: list[float] = Field(default_factory=lambda: [0.0, 5.0, 10.0, 25.0, 50.0])
    max_single_weight: float = Field(gt=0.0, le=1.0, default=0.15)
    turnover_smoothing_alpha: float = Field(gt=0.0, le=1.0, default=0.65)
    no_trade_buffer: float = Field(ge=0.0, le=0.25, default=0.005)
    annualization_days: int = Field(ge=1, default=252)


class ObservabilityConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    metrics_output_path: str = "reports/engine_metrics.json"


class EngineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    config_version: str
    environment: str = "local_research_prototype"
    seed: int = 42
    currency: str = "USD"
    execution_mode: Literal["paper_only"] = "paper_only"
    execution_enabled: bool = False
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    neutralization: NeutralizationConfig = Field(default_factory=NeutralizationConfig)
    cost_model: CostModelConfig = Field(default_factory=CostModelConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    @model_validator(mode="after")
    def _enforce_execution_disabled(self) -> EngineConfig:
        if self.execution_enabled:
            raise ExecutionDisabledError(
                "Live or automated order execution is disabled in MVP research engine.",
                details={"execution_mode": self.execution_mode},
            )
        return self


class UniversePolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    policy_version: str
    universe_id: str
    asset_class: str = "equities_synthetic"
    currency: str = "USD"
    point_in_time_enforced: bool = True
    include_delisted_until_exit: bool = True
    min_active_symbols_per_day: int = Field(ge=3, default=15)
    min_symbol_coverage_ratio: float = Field(gt=0.0, le=1.0, default=0.90)
    corporate_action_adjustment_required: bool = True
    max_unadjusted_split_jump_tolerance: float = Field(gt=0.0, default=0.15)
    default_filing_lag_days: int = Field(ge=0, default=45)


class RegistryPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    policy_version: str
    min_oos_mean_ic: float = 0.015
    min_oos_icir: float = 0.20
    max_bh_adjusted_pvalue: float = 0.10
    max_avg_daily_turnover: float = 0.45
    max_abs_sector_exposure: float = 0.10
    max_peer_factor_correlation: float = 0.85
    min_net_sharpe_at_base_cost: float = 0.25
    max_coefficient_sign_flip_rate: float = 0.35
    max_coefficient_cv: float = 1.50
    min_ic_sign_consistency: float = 0.50
    allow_promotion_on_gate_failure: bool = False
    export_mode: Literal["paper_only"] = "paper_only"


def load_engine_config(path: Path) -> EngineConfig:
    with path.open("r", encoding="utf-8") as fh:
        return EngineConfig.model_validate(json.load(fh))


def load_universe_policy(path: Path) -> UniversePolicy:
    with path.open("r", encoding="utf-8") as fh:
        return UniversePolicy.model_validate(json.load(fh))


def load_registry_policy(path: Path) -> RegistryPolicy:
    with path.open("r", encoding="utf-8") as fh:
        return RegistryPolicy.model_validate(json.load(fh))
