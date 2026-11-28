"""Quantitative Strategy Robustness Lab public typed library interface."""

__version__ = "0.1.0"

from .bootstrap import compute_max_drawdown, run_block_bootstrap
from .costs import CostAdjustedSeries, apply_transaction_costs
from .integrity import audit_trial_integrity
from .models import (
    BootstrapResult,
    ContractValidationError,
    CostModel,
    DSRResult,
    InsufficientSampleError,
    IntegrityAuditResult,
    OutlierDiagnostic,
    ParameterGridSpec,
    PBOResult,
    PerturbationNeighbor,
    PerturbationResult,
    PromotionDecision,
    PromotionGateCheck,
    PromotionVerdict,
    RegistryEntry,
    RegistryIntegrityError,
    ReturnFrequency,
    ReturnSeriesSpec,
    RobustnessLabError,
    TrialRecord,
    TrialStatus,
    UnsupportedConfigurationError,
    WalkForwardFold,
    WalkForwardResult,
)
from .pbo import compute_pbo_cscv
from .perturbation import evaluate_parameter_perturbation
from .pipeline import compare_reports_logical_equality, evaluate_experiment, evaluate_from_directory
from .promotion import evaluate_promotion
from .registry import GENESIS_HASH, ImmutableExperimentRegistry
from .statistics import (
    compute_autocorrelation_vif,
    compute_dsr,
    compute_expected_max_sharpe,
    compute_psr,
    compute_sample_moments,
    compute_sharpe_ratio,
    diagnose_outliers,
    estimate_effective_trials,
)
from .walk_forward import run_walk_forward

__all__ = [
    "__version__",
    "GENESIS_HASH",
    "BootstrapResult",
    "ContractValidationError",
    "CostAdjustedSeries",
    "CostModel",
    "DSRResult",
    "ImmutableExperimentRegistry",
    "InsufficientSampleError",
    "IntegrityAuditResult",
    "OutlierDiagnostic",
    "PBOResult",
    "ParameterGridSpec",
    "PerturbationNeighbor",
    "PerturbationResult",
    "PromotionDecision",
    "PromotionGateCheck",
    "PromotionVerdict",
    "RegistryEntry",
    "RegistryIntegrityError",
    "ReturnFrequency",
    "ReturnSeriesSpec",
    "RobustnessLabError",
    "TrialRecord",
    "TrialStatus",
    "UnsupportedConfigurationError",
    "WalkForwardFold",
    "WalkForwardResult",
    "apply_transaction_costs",
    "audit_trial_integrity",
    "compare_reports_logical_equality",
    "compute_autocorrelation_vif",
    "compute_dsr",
    "compute_expected_max_sharpe",
    "compute_max_drawdown",
    "compute_pbo_cscv",
    "compute_psr",
    "compute_sample_moments",
    "compute_sharpe_ratio",
    "diagnose_outliers",
    "estimate_effective_trials",
    "evaluate_experiment",
    "evaluate_from_directory",
    "evaluate_parameter_perturbation",
    "evaluate_promotion",
    "run_block_bootstrap",
    "run_walk_forward",
]
