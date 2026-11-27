"""Structured domain and operational errors for the Performance Attribution Engine."""

from __future__ import annotations

from typing import Any, Dict, Optional


class PAEError(Exception):
    """Base structured error with machine-readable code and context."""

    code: str = "PAE_ERROR"
    http_status: int = 400

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        http_status: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if http_status is not None:
            self.http_status = http_status
        self.details: Dict[str, Any] = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.code,
            "message": self.message,
            "details": self.details,
        }


class ValidationError(PAEError):
    code = "VALIDATION_ERROR"
    http_status = 422


class ConfigurationError(PAEError):
    code = "CONFIGURATION_ERROR"
    http_status = 400


class OrderExecutionProhibitedError(PAEError):
    code = "ORDER_EXECUTION_PROHIBITED"
    http_status = 403


class ZeroNavError(PAEError):
    code = "ZERO_NAV"
    http_status = 422


class CurrencyConversionError(PAEError):
    code = "CURRENCY_CONVERSION_ERROR"
    http_status = 422


class BenchmarkAlignmentError(PAEError):
    code = "BENCHMARK_ALIGNMENT_ERROR"
    http_status = 422


class ReconciliationError(PAEError):
    code = "RECONCILIATION_ERROR"
    http_status = 422


class AuthorizationError(PAEError):
    code = "AUTHORIZATION_ERROR"
    http_status = 403


class WorkflowViolationError(PAEError):
    code = "WORKFLOW_VIOLATION"
    http_status = 409


class NotFoundError(PAEError):
    code = "NOT_FOUND"
    http_status = 404


class IntegrityError(PAEError):
    code = "INTEGRITY_ERROR"
    http_status = 500
