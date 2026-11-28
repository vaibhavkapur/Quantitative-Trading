"""Typed exception hierarchy for the Cross-Asset Factor Investing Engine."""

from __future__ import annotations

from typing import Any


class CrossAssetEngineError(Exception):
    """Base error with machine-readable code and structured diagnostics."""

    code: str = "ENGINE_ERROR"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": type(self).__name__,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class ConfigurationError(CrossAssetEngineError):
    """Raised when configuration is missing, malformed, or internally inconsistent."""

    code = "CONFIGURATION_ERROR"


class SchemaValidationError(CrossAssetEngineError):
    """Raised when an input or output dataset violates the documented contract."""

    code = "SCHEMA_VALIDATION_ERROR"


class UnsupportedAssetError(CrossAssetEngineError):
    """Raised when an unsupported asset class, instrument, or carry convention is supplied."""

    code = "UNSUPPORTED_ASSET_ERROR"


class TimestampGateError(CrossAssetEngineError):
    """Raised when a look-ahead or timestamp ordering invariant is violated."""

    code = "TIMESTAMP_GATE_ERROR"


class CurrencyConversionError(CrossAssetEngineError):
    """Raised when FX conversion cannot be completed due to missing or invalid rates."""

    code = "CURRENCY_CONVERSION_ERROR"


class RollNormalizationError(CrossAssetEngineError):
    """Raised when a futures contract roll transition cannot be normalized cleanly."""

    code = "ROLL_NORMALIZATION_ERROR"


class MissingFactorError(CrossAssetEngineError):
    """Raised in strict mode when required factor inputs are unavailable."""

    code = "MISSING_FACTOR_ERROR"


class ConstraintViolationError(CrossAssetEngineError):
    """Raised when portfolio constraints are infeasible or violated beyond tolerance."""

    code = "CONSTRAINT_VIOLATION_ERROR"
