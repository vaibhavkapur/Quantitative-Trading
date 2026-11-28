"""Cross-Asset Factor Investing Engine — public typed library interface."""

from __future__ import annotations

__version__ = "0.1.0"
SCHEMA_VERSION = "1.0.0"

from cross_asset_factor_engine.config import EngineConfig, load_config, validate_config
from cross_asset_factor_engine.contracts import MarketDataset, validate_dataset
from cross_asset_factor_engine.diagnostics import DiagnosticTracker
from cross_asset_factor_engine.errors import (
    ConfigurationError,
    ConstraintViolationError,
    CrossAssetEngineError,
    CurrencyConversionError,
    MissingFactorError,
    RollNormalizationError,
    SchemaValidationError,
    TimestampGateError,
    UnsupportedAssetError,
)
from cross_asset_factor_engine.evaluation import (
    EvaluationOutput,
    WalkForwardFold,
    generate_walk_forward_splits,
    run_walk_forward_evaluation,
)
from cross_asset_factor_engine.factors import (
    FactorScaler,
    build_blended_signals,
    compute_raw_factor_signals,
    verify_signal_provenance,
)
from cross_asset_factor_engine.io import load_dataset_from_dir
from cross_asset_factor_engine.normalization import normalize_market_data
from cross_asset_factor_engine.portfolio import (
    OptimizationResult,
    compute_unconstrained_vol_scaled_weights,
    estimate_covariance,
    optimize_constrained_portfolio,
    verify_portfolio_constraints,
)
from cross_asset_factor_engine.reporting import (
    make_run_id,
    save_evaluation_report,
    verify_replay_equivalence,
)

__all__ = [
    "__version__",
    "SCHEMA_VERSION",
    "EngineConfig",
    "load_config",
    "validate_config",
    "MarketDataset",
    "validate_dataset",
    "load_dataset_from_dir",
    "DiagnosticTracker",
    "CrossAssetEngineError",
    "ConfigurationError",
    "SchemaValidationError",
    "UnsupportedAssetError",
    "TimestampGateError",
    "CurrencyConversionError",
    "RollNormalizationError",
    "MissingFactorError",
    "ConstraintViolationError",
    "normalize_market_data",
    "compute_raw_factor_signals",
    "verify_signal_provenance",
    "FactorScaler",
    "build_blended_signals",
    "estimate_covariance",
    "compute_unconstrained_vol_scaled_weights",
    "optimize_constrained_portfolio",
    "verify_portfolio_constraints",
    "OptimizationResult",
    "WalkForwardFold",
    "generate_walk_forward_splits",
    "run_walk_forward_evaluation",
    "EvaluationOutput",
    "make_run_id",
    "save_evaluation_report",
    "verify_replay_equivalence",
]
