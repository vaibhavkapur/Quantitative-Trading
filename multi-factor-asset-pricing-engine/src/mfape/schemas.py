"""Typed data contracts, configurations, and result schemas for mfape."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from mfape.errors import ConfigurationError, DateLeakageError, UnsupportedInputError

SCHEMA_VERSION = "1.0.0"

SUPPORTED_FREQUENCIES = ("M", "D", "W", "Q")
SUPPORTED_COVARIANCE_TYPES = ("OLS", "HC3", "HAC")
SUPPORTED_UNIVERSE_POLICIES = ("strict_balanced", "pit_active", "min_obs_balanced")
SUPPORTED_RF_MISSING_POLICIES = ("error", "drop_period")
SUPPORTED_ALIGNMENT_POLICIES = ("strict", "inner")


@dataclass(frozen=True)
class FactorModelSpec:
    """Specification for a single factor pricing model (e.g., CAPM, FF3, FF5)."""

    name: str
    factors: tuple[str, ...]
    include_ts_intercept: bool = True
    include_cs_intercept: bool = True
    description: str = ""

    def validate(self) -> None:
        if not self.name or not self.name.strip():
            raise ConfigurationError("Model spec name must be non-empty.")
        if not self.factors:
            raise ConfigurationError(
                f"Model '{self.name}' must specify at least one factor.",
                {"model": self.name},
            )
        if len(set(self.factors)) != len(self.factors):
            raise ConfigurationError(
                f"Model '{self.name}' contains duplicate factors: {list(self.factors)}.",
                {"model": self.name, "factors": list(self.factors)},
            )


@dataclass(frozen=True)
class SplitConfig:
    """Chronological train/test split boundaries enforcing zero date leakage."""

    train_start: str
    train_end: str
    test_start: str | None = None
    test_end: str | None = None

    def validate(self) -> None:
        if self.train_start > self.train_end:
            raise ConfigurationError(
                f"train_start ({self.train_start}) must be <= train_end ({self.train_end})."
            )
        if (self.test_start is None) != (self.test_end is None):
            raise ConfigurationError(
                "Both test_start and test_end must be provided together, or both omitted."
            )
        if self.test_start is not None and self.test_end is not None:
            if self.test_start > self.test_end:
                raise ConfigurationError(
                    f"test_start ({self.test_start}) must be <= test_end ({self.test_end})."
                )
            if self.test_start <= self.train_end:
                raise DateLeakageError(
                    f"Date leakage detected: test_start ({self.test_start}) must be strictly "
                    f"after train_end ({self.train_end}).",
                    {"train_end": self.train_end, "test_start": self.test_start},
                )


@dataclass(frozen=True)
class EngineConfig:
    """Top-level configuration for a reproducible asset-pricing engine run."""

    config_version: str
    run_label: str
    asset_returns_path: str
    factor_returns_path: str
    risk_free_path: str
    memberships_path: str
    models: tuple[FactorModelSpec, ...]
    baseline_model_name: str = "CAPM"
    frequency: str = "M"
    annualization_factor: float = 12.0
    cov_type: str = "HAC"
    hac_maxlags: int = 3
    fm_hac_maxlags: int = 3
    universe_policy: str = "pit_active"
    rf_missing_policy: str = "error"
    period_alignment_policy: str = "strict"
    min_ts_observations: int = 12
    min_cs_assets: int = 3
    rank_tol: float = 1e-10
    max_condition_number: float = 1e8
    split: SplitConfig | None = None
    seed: int = 42
    output_dir: str | None = None

    def validate(self) -> None:
        if not self.config_version:
            raise ConfigurationError("config_version must be non-empty.")
        if not self.run_label:
            raise ConfigurationError("run_label must be non-empty.")
        if self.frequency not in SUPPORTED_FREQUENCIES:
            raise UnsupportedInputError(
                f"Unsupported frequency '{self.frequency}'. Supported: {SUPPORTED_FREQUENCIES}.",
                {"frequency": self.frequency},
            )
        if self.cov_type not in SUPPORTED_COVARIANCE_TYPES:
            raise UnsupportedInputError(
                f"Unsupported cov_type '{self.cov_type}'. Supported: {SUPPORTED_COVARIANCE_TYPES}.",
                {"cov_type": self.cov_type},
            )
        if self.universe_policy not in SUPPORTED_UNIVERSE_POLICIES:
            raise UnsupportedInputError(
                f"Unsupported universe_policy '{self.universe_policy}'. "
                f"Supported: {SUPPORTED_UNIVERSE_POLICIES}.",
                {"universe_policy": self.universe_policy},
            )
        if self.rf_missing_policy not in SUPPORTED_RF_MISSING_POLICIES:
            raise UnsupportedInputError(
                f"Unsupported rf_missing_policy '{self.rf_missing_policy}'. "
                f"Supported: {SUPPORTED_RF_MISSING_POLICIES}.",
                {"rf_missing_policy": self.rf_missing_policy},
            )
        if self.period_alignment_policy not in SUPPORTED_ALIGNMENT_POLICIES:
            raise UnsupportedInputError(
                f"Unsupported period_alignment_policy '{self.period_alignment_policy}'. "
                f"Supported: {SUPPORTED_ALIGNMENT_POLICIES}.",
                {"period_alignment_policy": self.period_alignment_policy},
            )
        if self.annualization_factor <= 0:
            raise ConfigurationError("annualization_factor must be strictly positive.")
        if self.hac_maxlags < 0 or self.fm_hac_maxlags < 0:
            raise ConfigurationError("HAC maxlags must be non-negative.")
        if self.min_ts_observations < 3:
            raise ConfigurationError("min_ts_observations must be at least 3.")
        if self.min_cs_assets < 2:
            raise ConfigurationError("min_cs_assets must be at least 2.")
        if not self.models:
            raise ConfigurationError("At least one factor model must be configured.")
        names: list[str] = []
        for spec in self.models:
            spec.validate()
            names.append(spec.name)
        if len(set(names)) != len(names):
            raise ConfigurationError(f"Duplicate model names configured: {names}.")
        if self.baseline_model_name not in names:
            raise ConfigurationError(
                f"baseline_model_name '{self.baseline_model_name}' not found in configured models {names}.",
                {"baseline_model_name": self.baseline_model_name, "models": names},
            )
        if self.split is not None:
            self.split.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EngineConfig:
        if not isinstance(data, dict):
            raise ConfigurationError("Configuration payload must be a JSON object.")
        required = (
            "config_version",
            "run_label",
            "asset_returns_path",
            "factor_returns_path",
            "risk_free_path",
            "memberships_path",
            "models",
        )
        missing = [k for k in required if k not in data]
        if missing:
            raise ConfigurationError(
                f"Missing required configuration fields: {missing}.",
                {"missing_fields": missing},
            )
        raw_models = data["models"]
        if not isinstance(raw_models, list) or not raw_models:
            raise ConfigurationError("Configuration field 'models' must be a non-empty list.")
        models: list[FactorModelSpec] = []
        for idx, m in enumerate(raw_models):
            if not isinstance(m, dict) or "name" not in m or "factors" not in m:
                raise ConfigurationError(
                    f"Model entry at index {idx} must have 'name' and 'factors'."
                )
            models.append(
                FactorModelSpec(
                    name=str(m["name"]),
                    factors=tuple(str(f) for f in m["factors"]),
                    include_ts_intercept=bool(m.get("include_ts_intercept", True)),
                    include_cs_intercept=bool(m.get("include_cs_intercept", True)),
                    description=str(m.get("description", "")),
                )
            )
        split_obj: SplitConfig | None = None
        if data.get("split") is not None:
            s = data["split"]
            if not isinstance(s, dict) or "train_start" not in s or "train_end" not in s:
                raise ConfigurationError("Split config requires 'train_start' and 'train_end'.")
            split_obj = SplitConfig(
                train_start=str(s["train_start"]),
                train_end=str(s["train_end"]),
                test_start=str(s["test_start"]) if s.get("test_start") is not None else None,
                test_end=str(s["test_end"]) if s.get("test_end") is not None else None,
            )

        cfg = cls(
            config_version=str(data["config_version"]),
            run_label=str(data["run_label"]),
            asset_returns_path=str(data["asset_returns_path"]),
            factor_returns_path=str(data["factor_returns_path"]),
            risk_free_path=str(data["risk_free_path"]),
            memberships_path=str(data["memberships_path"]),
            models=tuple(models),
            baseline_model_name=str(data.get("baseline_model_name", "CAPM")),
            frequency=str(data.get("frequency", "M")),
            annualization_factor=float(data.get("annualization_factor", 12.0)),
            cov_type=str(data.get("cov_type", "HAC")),
            hac_maxlags=int(data.get("hac_maxlags", 3)),
            fm_hac_maxlags=int(data.get("fm_hac_maxlags", 3)),
            universe_policy=str(data.get("universe_policy", "pit_active")),
            rf_missing_policy=str(data.get("rf_missing_policy", "error")),
            period_alignment_policy=str(data.get("period_alignment_policy", "strict")),
            min_ts_observations=int(data.get("min_ts_observations", 12)),
            min_cs_assets=int(data.get("min_cs_assets", 3)),
            rank_tol=float(data.get("rank_tol", 1e-10)),
            max_condition_number=float(data.get("max_condition_number", 1e8)),
            split=split_obj,
            seed=int(data.get("seed", 42)),
            output_dir=str(data["output_dir"]) if data.get("output_dir") is not None else None,
        )
        cfg.validate()
        return cfg


@dataclass
class DiagnosticCounters:
    """Structured counters and event logs emitted during pipeline and model execution."""

    raw_asset_rows: int = 0
    raw_factor_rows: int = 0
    raw_rf_rows: int = 0
    raw_membership_rows: int = 0
    aligned_rows: int = 0
    aligned_periods: int = 0
    aligned_assets: int = 0
    dropped_inactive_membership_rows: int = 0
    dropped_missing_rf_periods: int = 0
    dropped_unaligned_periods: int = 0
    universe_entries: int = 0
    universe_exits: int = 0
    universe_change_periods: int = 0
    assets_excluded_min_obs: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)

    def record(self, level: str, stage: str, message: str, **fields: Any) -> None:
        self.events.append(
            {
                "level": level,
                "stage": stage,
                "message": message,
                **fields,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AlignedPanel:
    """Validated, point-in-time aligned excess-return and factor panel (NumPy arrays)."""

    dates: tuple[str, ...]
    assets: tuple[str, ...]
    factor_names: tuple[str, ...]
    # Shape: (T, N), NaN where an asset is not active in the universe at period t
    excess_returns: Any  # np.ndarray float64 (T, N)
    # Shape: (T, N), bool mask where asset is active & observed at period t
    active_mask: Any  # np.ndarray bool (T, N)
    # Shape: (T, K), float64 factor returns aligned to dates
    factor_returns: Any  # np.ndarray float64 (T, K)
    # Shape: (T,), float64 risk-free rate per period
    risk_free_rates: Any  # np.ndarray float64 (T,)
    frequency: str
    is_balanced: bool
    universe_changes: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class AssetTimeSeriesResult:
    """Time-series regression summary for a single asset or test portfolio."""

    asset_id: str
    n_obs: int
    alpha: float | None
    alpha_se: float | None
    alpha_tstat: float | None
    alpha_pvalue: float | None
    betas: dict[str, float]
    beta_ses: dict[str, float]
    beta_tstats: dict[str, float]
    beta_pvalues: dict[str, float]
    r_squared: float
    adj_r_squared: float
    residual_std: float
    mean_excess_return: float
    predicted_mean_excess_return: float
    pricing_error: float
    residuals: tuple[float, ...]
    dates: tuple[str, ...]

    def to_dict(self, include_residuals: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if not include_residuals:
            data.pop("residuals", None)
            data.pop("dates", None)
        return data


@dataclass(frozen=True)
class TimeSeriesModelResult:
    """Aggregate time-series CAPM or multi-factor regression results across all assets."""

    model_name: str
    factors: tuple[str, ...]
    cov_type: str
    hac_maxlags: int
    include_intercept: bool
    n_periods: int
    n_assets: int
    date_start: str
    date_end: str
    factor_condition_number: float
    asset_results: tuple[AssetTimeSeriesResult, ...]
    mean_r_squared: float
    mean_adj_r_squared: float
    mean_abs_alpha: float
    rmse_alpha: float
    grs_f_stat: float | None
    grs_pvalue: float | None
    grs_df1: int | None
    grs_df2: int | None
    grs_note: str | None = None

    def to_dict(self, include_residuals: bool = False) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "factors": list(self.factors),
            "cov_type": self.cov_type,
            "hac_maxlags": self.hac_maxlags,
            "include_intercept": self.include_intercept,
            "n_periods": self.n_periods,
            "n_assets": self.n_assets,
            "date_start": self.date_start,
            "date_end": self.date_end,
            "factor_condition_number": self.factor_condition_number,
            "mean_r_squared": self.mean_r_squared,
            "mean_adj_r_squared": self.mean_adj_r_squared,
            "mean_abs_alpha": self.mean_abs_alpha,
            "rmse_alpha": self.rmse_alpha,
            "grs_f_stat": self.grs_f_stat,
            "grs_pvalue": self.grs_pvalue,
            "grs_df1": self.grs_df1,
            "grs_df2": self.grs_df2,
            "grs_note": self.grs_note,
            "asset_results": [
                ar.to_dict(include_residuals=include_residuals) for ar in self.asset_results
            ],
        }


@dataclass(frozen=True)
class FactorPremiumEstimate:
    """Estimated cross-sectional risk premium for one factor (or zero-beta intercept)."""

    name: str
    estimate: float
    annualized_estimate: float
    fm_se: float
    fm_tstat: float
    fm_pvalue: float
    shanken_se: float | None
    shanken_tstat: float | None
    shanken_pvalue: float | None
    hac_se: float
    hac_tstat: float
    hac_pvalue: float


@dataclass(frozen=True)
class PricingErrorMetrics:
    """Cross-sectional pricing error summary metrics."""

    mae: float
    rmse: float
    max_abs_error: float
    mean_error: float
    cross_sectional_r2: float
    adj_cross_sectional_r2: float | None


@dataclass(frozen=True)
class FamaMacBethResult:
    """Two-pass Fama-MacBeth cross-sectional regression outputs with robust SEs."""

    model_name: str
    factors: tuple[str, ...]
    include_intercept: bool
    n_periods: int
    n_assets: int
    date_start: str
    date_end: str
    beta_condition_number: float
    shanken_correction_factor: float | None
    hac_maxlags: int
    premia: tuple[FactorPremiumEstimate, ...]
    pricing_errors_by_asset: dict[str, float]
    mean_actual_excess_by_asset: dict[str, float]
    mean_predicted_excess_by_asset: dict[str, float]
    pricing_metrics: PricingErrorMetrics
    avg_period_cs_r2: float
    lambda_series_dates: tuple[str, ...]
    lambda_series: dict[str, tuple[float, ...]]

    def to_dict(self, include_series: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if not include_series:
            data.pop("lambda_series_dates", None)
            data.pop("lambda_series", None)
        return data


@dataclass(frozen=True)
class OOSEvaluationResult:
    """Out-of-sample evaluation metrics strictly trained on train_window and tested on test_window."""

    model_name: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train_periods: int
    n_test_periods: int
    n_evaluated_assets: int
    # Cross-sectional OOS pricing error using train betas + train FM premia vs realized test means
    oos_cs_pricing_metrics: PricingErrorMetrics
    # Time-series OOS prediction using train betas * realized test factors (+ train alpha=0 pricing)
    oos_ts_r2_zero_alpha: float
    oos_ts_r2_with_train_alpha: float
    oos_ts_rmse_zero_alpha: float
    oos_ts_mae_zero_alpha: float
    # OOS R² relative to naive historical training-mean excess return benchmark
    oos_cs_r2_vs_train_mean: float
    # Per-asset realized vs predicted mean excess return in test window
    oos_pricing_errors_by_asset: dict[str, float]


@dataclass(frozen=True)
class ModelEvaluationResult:
    """Complete in-sample and optional out-of-sample evaluation for one factor model."""

    spec: FactorModelSpec
    time_series_train: TimeSeriesModelResult
    fama_macbeth_train: FamaMacBethResult
    oos_evaluation: OOSEvaluationResult | None = None

    def to_dict(self, include_series: bool = False) -> dict[str, Any]:
        return {
            "spec": asdict(self.spec),
            "time_series_train": self.time_series_train.to_dict(include_residuals=include_series),
            "fama_macbeth_train": self.fama_macbeth_train.to_dict(include_series=include_series),
            "oos_evaluation": asdict(self.oos_evaluation) if self.oos_evaluation else None,
        }


@dataclass(frozen=True)
class ComparisonSummary:
    """Comparison of all configured candidate models against the designated CAPM baseline."""

    baseline_model_name: str
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class RunOutput:
    """Versioned, reproducible top-level engine output contract."""

    schema_version: str
    engine_version: str
    run_id: str
    run_label: str
    config: dict[str, Any]
    input_fingerprints: dict[str, str]
    runtime_environment: dict[str, Any]
    split_boundaries: dict[str, Any]
    diagnostics: dict[str, Any]
    models: tuple[ModelEvaluationResult, ...]
    comparison: ComparisonSummary

    def to_dict(self, include_series: bool = False) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "engine_version": self.engine_version,
            "run_id": self.run_id,
            "run_label": self.run_label,
            "config": self.config,
            "input_fingerprints": self.input_fingerprints,
            "runtime_environment": self.runtime_environment,
            "split_boundaries": self.split_boundaries,
            "diagnostics": self.diagnostics,
            "models": [m.to_dict(include_series=include_series) for m in self.models],
            "comparison": asdict(self.comparison),
        }
