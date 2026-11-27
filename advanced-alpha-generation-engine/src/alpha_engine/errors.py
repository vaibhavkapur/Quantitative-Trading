"""Structured domain errors for the Advanced Alpha Generation Engine."""

from __future__ import annotations

from typing import Any


class AlphaEngineError(Exception):
    """Base structured exception for all engine errors."""

    error_code: str = "ALPHA_ENGINE_ERROR"

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        if error_code is not None:
            self.error_code = error_code

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.error_code,
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class DataValidationError(AlphaEngineError):
    error_code = "DATA_VALIDATION_ERROR"


class LookAheadViolationError(AlphaEngineError):
    error_code = "LOOK_AHEAD_VIOLATION"


class SurvivorshipViolationError(AlphaEngineError):
    error_code = "SURVIVORSHIP_VIOLATION"


class CorporateActionIntegrityError(AlphaEngineError):
    error_code = "CORPORATE_ACTION_INTEGRITY_ERROR"


class MissingSymbolError(AlphaEngineError):
    error_code = "MISSING_SYMBOL_ERROR"


class FactorDuplicationError(AlphaEngineError):
    error_code = "FACTOR_DUPLICATION_ERROR"


class HighTurnoverError(AlphaEngineError):
    error_code = "HIGH_TURNOVER_ERROR"


class UnstableCoefficientsError(AlphaEngineError):
    error_code = "UNSTABLE_COEFFICIENTS_ERROR"


class PromotionGateError(AlphaEngineError):
    error_code = "PROMOTION_GATE_ERROR"


class ExecutionDisabledError(AlphaEngineError):
    error_code = "EXECUTION_DISABLED_ERROR"


class OptionalDependencyError(AlphaEngineError):
    error_code = "OPTIONAL_DEPENDENCY_ERROR"
