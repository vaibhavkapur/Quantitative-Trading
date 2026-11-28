"""Typed domain and data-contract exception hierarchy for mfape."""

from __future__ import annotations

from typing import Any


class MFAPEError(Exception):
    """Base error for all Multi-Factor Asset Pricing Engine failures."""

    code: str = "MFAPE_ERROR"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.code,
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class ConfigurationError(MFAPEError):
    """Raised when engine or model configuration is missing, malformed, or invalid."""

    code = "CONFIGURATION_ERROR"


class UnsupportedInputError(MFAPEError):
    """Raised when an input format, covariance estimator, or mode is unsupported."""

    code = "UNSUPPORTED_INPUT"


class DataContractError(MFAPEError):
    """Raised when input data violates schema, type, uniqueness, or value contracts."""

    code = "DATA_CONTRACT_ERROR"


class MissingRiskFreeError(DataContractError):
    """Raised when risk-free observations are missing for required return periods."""

    code = "MISSING_RISK_FREE"


class PeriodAlignmentError(DataContractError):
    """Raised when date grids, period frequencies, or asset-factor dates fail alignment."""

    code = "PERIOD_ALIGNMENT_ERROR"


class UniverseChangeError(DataContractError):
    """Raised when point-in-time portfolio membership violates universe policies."""

    code = "UNIVERSE_CHANGE_ERROR"


class DateLeakageError(MFAPEError):
    """Raised when training and evaluation windows overlap or leak future data."""

    code = "DATE_LEAKAGE_ERROR"


class RankDeficiencyError(MFAPEError):
    """Raised when factor or beta design matrices are rank-deficient or ill-conditioned."""

    code = "RANK_DEFICIENCY_ERROR"


class InsufficientObservationsError(MFAPEError):
    """Raised when too few time-series or cross-sectional observations exist to fit a model."""

    code = "INSUFFICIENT_OBSERVATIONS"
