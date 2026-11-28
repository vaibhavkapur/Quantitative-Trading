"""Multi-Factor Asset Pricing Engine (mfape).

Educational and research prototype for excess-return alignment, time-series CAPM
and multi-factor regressions, Fama-MacBeth cross-sectional risk premia estimation
with robust standard errors, and out-of-sample pricing error comparison.
"""

from __future__ import annotations

from mfape.engine import AssetPricingEngine
from mfape.errors import (
    ConfigurationError,
    DataContractError,
    DateLeakageError,
    InsufficientObservationsError,
    MFAPEError,
    MissingRiskFreeError,
    PeriodAlignmentError,
    RankDeficiencyError,
    UniverseChangeError,
    UnsupportedInputError,
)
from mfape.schemas import (
    AlignedPanel,
    AssetTimeSeriesResult,
    ComparisonSummary,
    DiagnosticCounters,
    EngineConfig,
    FactorModelSpec,
    FactorPremiumEstimate,
    FamaMacBethResult,
    ModelEvaluationResult,
    OOSEvaluationResult,
    PricingErrorMetrics,
    RunOutput,
    SplitConfig,
    TimeSeriesModelResult,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AssetPricingEngine",
    "MFAPEError",
    "ConfigurationError",
    "DataContractError",
    "DateLeakageError",
    "InsufficientObservationsError",
    "MissingRiskFreeError",
    "PeriodAlignmentError",
    "RankDeficiencyError",
    "UniverseChangeError",
    "UnsupportedInputError",
    "AlignedPanel",
    "AssetTimeSeriesResult",
    "ComparisonSummary",
    "DiagnosticCounters",
    "EngineConfig",
    "FactorModelSpec",
    "FactorPremiumEstimate",
    "FamaMacBethResult",
    "ModelEvaluationResult",
    "OOSEvaluationResult",
    "PricingErrorMetrics",
    "RunOutput",
    "SplitConfig",
    "TimeSeriesModelResult",
]
