"""Typed configuration schema, YAML loading, validation, and fingerprinting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from volatility_forecasting.errors import ConfigurationError, UnsupportedOptionError

SUPPORTED_SPLIT_SCHEMES = {"expanding", "rolling"}
SUPPORTED_MODEL_TYPES = {
    "naive",
    "rolling_historical",
    "ewma",
    "garch11",
    "ridge_har",
    "hgbr",
}


@dataclass(frozen=True)
class DataConfig:
    returns_path: str
    intraday_bars_path: str | None = None
    labels_path: str | None = None


@dataclass(frozen=True)
class FeatureConfig:
    horizons: tuple[int, ...] = (1, 5, 22)
    primary_horizon: int = 5
    rolling_windows: tuple[int, ...] = (5, 22)
    include_overnight_gap: bool = True
    min_bar_coverage_warn: float = 0.80
    variance_floor: float = 1e-10


@dataclass(frozen=True)
class SplitConfig:
    scheme: str = "expanding"
    n_splits: int = 4
    min_train_size: int = 160
    test_size: int = 60
    embargo_steps: int | None = None
    calibration_fraction: float = 0.25


@dataclass(frozen=True)
class ModelSpec:
    name: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegimeQuantileConfig:
    low_upper: float = 0.33
    normal_upper: float = 0.67
    high_upper: float = 0.95


@dataclass(frozen=True)
class EvaluationConfig:
    variance_floor: float = 1e-10
    interval_levels: tuple[float, ...] = (0.50, 0.80, 0.90)
    regime_quantiles: RegimeQuantileConfig = field(default_factory=RegimeQuantileConfig)
    strict_nonnegative: bool = True


@dataclass(frozen=True)
class OutputConfig:
    reports_dir: str = "reports"
    save_predictions: bool = True


@dataclass(frozen=True)
class EngineConfig:
    version: str
    run_name: str
    seed: int
    annualization_factor: int
    expected_bars_per_session: int
    data: DataConfig
    features: FeatureConfig
    splits: SplitConfig
    models: tuple[ModelSpec, ...]
    evaluation: EvaluationConfig
    output: OutputConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Deterministic SHA-256 hash of canonicalized configuration."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def validate_config(cfg: EngineConfig) -> None:
    """Validate semantic invariants on EngineConfig."""
    if not cfg.run_name.strip():
        raise ConfigurationError("run_name must be a non-empty string.")
    if cfg.annualization_factor <= 0:
        raise ConfigurationError("annualization_factor must be > 0.")
    if cfg.expected_bars_per_session <= 0:
        raise ConfigurationError("expected_bars_per_session must be > 0.")
    if not cfg.data.returns_path:
        raise ConfigurationError("data.returns_path is required.")

    if not cfg.features.horizons or any(h <= 0 for h in cfg.features.horizons):
        raise ConfigurationError("features.horizons must contain positive integers.")
    if cfg.features.primary_horizon not in cfg.features.horizons:
        raise ConfigurationError(
            f"features.primary_horizon ({cfg.features.primary_horizon}) must be in "
            f"features.horizons ({cfg.features.horizons})."
        )
    if any(w <= 0 for w in cfg.features.rolling_windows):
        raise ConfigurationError("features.rolling_windows must contain positive integers.")
    if cfg.features.variance_floor <= 0:
        raise ConfigurationError("features.variance_floor must be > 0.")

    if cfg.splits.scheme not in SUPPORTED_SPLIT_SCHEMES:
        raise UnsupportedOptionError(
            f"Unsupported split scheme '{cfg.splits.scheme}'. "
            f"Supported: {sorted(SUPPORTED_SPLIT_SCHEMES)}."
        )
    if cfg.splits.n_splits <= 0:
        raise ConfigurationError("splits.n_splits must be > 0.")
    if cfg.splits.min_train_size < 10:
        raise ConfigurationError("splits.min_train_size must be >= 10.")
    if cfg.splits.test_size <= 0:
        raise ConfigurationError("splits.test_size must be > 0.")
    if cfg.splits.embargo_steps is not None and cfg.splits.embargo_steps < 0:
        raise ConfigurationError("splits.embargo_steps must be >= 0 when provided.")
    if not (0.0 < cfg.splits.calibration_fraction < 0.8):
        raise ConfigurationError("splits.calibration_fraction must be in (0.0, 0.8).")

    if not cfg.models:
        raise ConfigurationError("At least one model must be configured in models.")
    seen_names: set[str] = set()
    for m in cfg.models:
        if not m.name.strip():
            raise ConfigurationError("Every model entry must have a non-empty name.")
        if m.name in seen_names:
            raise ConfigurationError(f"Duplicate model name '{m.name}'.")
        seen_names.add(m.name)
        if m.type not in SUPPORTED_MODEL_TYPES:
            raise UnsupportedOptionError(
                f"Unsupported model type '{m.type}' for '{m.name}'. "
                f"Supported: {sorted(SUPPORTED_MODEL_TYPES)}."
            )

    if cfg.evaluation.variance_floor <= 0:
        raise ConfigurationError("evaluation.variance_floor must be > 0.")
    if not cfg.evaluation.interval_levels or any(
        not (0.0 < lvl < 1.0) for lvl in cfg.evaluation.interval_levels
    ):
        raise ConfigurationError("evaluation.interval_levels must be probabilities in (0, 1).")

    rq = cfg.evaluation.regime_quantiles
    if not (0.0 < rq.low_upper < rq.normal_upper < rq.high_upper < 1.0):
        raise ConfigurationError(
            "evaluation.regime_quantiles must satisfy "
            "0 < low_upper < normal_upper < high_upper < 1."
        )


def load_config(path: str | Path) -> EngineConfig:
    """Load and validate an EngineConfig from a YAML file."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise ConfigurationError(f"Configuration file not found: {cfg_path}")
    if not cfg_path.is_file():
        raise ConfigurationError(f"Configuration path is not a file: {cfg_path}")

    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {cfg_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigurationError(f"Configuration root in {cfg_path} must be a mapping.")

    try:
        data_raw = raw.get("data") or {}
        features_raw = raw.get("features") or {}
        splits_raw = raw.get("splits") or {}
        models_raw = raw.get("models") or []
        eval_raw = raw.get("evaluation") or {}
        rq_raw = eval_raw.get("regime_quantiles") or {}
        output_raw = raw.get("output") or {}

        if not isinstance(models_raw, list):
            raise ConfigurationError("'models' must be a list of model definitions.")

        cfg = EngineConfig(
            version=str(raw.get("version", "1.0")),
            run_name=str(raw.get("run_name", "unnamed_run")),
            seed=int(raw.get("seed", 42)),
            annualization_factor=int(raw.get("annualization_factor", 252)),
            expected_bars_per_session=int(raw.get("expected_bars_per_session", 78)),
            data=DataConfig(
                returns_path=str(data_raw.get("returns_path", "")),
                intraday_bars_path=(
                    str(data_raw["intraday_bars_path"])
                    if data_raw.get("intraday_bars_path")
                    else None
                ),
                labels_path=(
                    str(data_raw["labels_path"]) if data_raw.get("labels_path") else None
                ),
            ),
            features=FeatureConfig(
                horizons=tuple(int(h) for h in features_raw.get("horizons", [1, 5, 22])),
                primary_horizon=int(features_raw.get("primary_horizon", 5)),
                rolling_windows=tuple(
                    int(w) for w in features_raw.get("rolling_windows", [5, 22])
                ),
                include_overnight_gap=bool(features_raw.get("include_overnight_gap", True)),
                min_bar_coverage_warn=float(features_raw.get("min_bar_coverage_warn", 0.80)),
                variance_floor=float(features_raw.get("variance_floor", 1e-10)),
            ),
            splits=SplitConfig(
                scheme=str(splits_raw.get("scheme", "expanding")),
                n_splits=int(splits_raw.get("n_splits", 4)),
                min_train_size=int(splits_raw.get("min_train_size", 160)),
                test_size=int(splits_raw.get("test_size", 60)),
                embargo_steps=(
                    int(splits_raw["embargo_steps"])
                    if splits_raw.get("embargo_steps") is not None
                    else None
                ),
                calibration_fraction=float(splits_raw.get("calibration_fraction", 0.25)),
            ),
            models=tuple(
                ModelSpec(
                    name=str(m["name"]),
                    type=str(m["type"]),
                    params=dict(m.get("params") or {}),
                )
                for m in models_raw
            ),
            evaluation=EvaluationConfig(
                variance_floor=float(eval_raw.get("variance_floor", 1e-10)),
                interval_levels=tuple(
                    float(x) for x in eval_raw.get("interval_levels", [0.50, 0.80, 0.90])
                ),
                regime_quantiles=RegimeQuantileConfig(
                    low_upper=float(rq_raw.get("low_upper", 0.33)),
                    normal_upper=float(rq_raw.get("normal_upper", 0.67)),
                    high_upper=float(rq_raw.get("high_upper", 0.95)),
                ),
                strict_nonnegative=bool(eval_raw.get("strict_nonnegative", True)),
            ),
            output=OutputConfig(
                reports_dir=str(output_raw.get("reports_dir", "reports")),
                save_predictions=bool(output_raw.get("save_predictions", True)),
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"Malformed configuration field in {cfg_path}: {exc}") from exc

    validate_config(cfg)
    return cfg
