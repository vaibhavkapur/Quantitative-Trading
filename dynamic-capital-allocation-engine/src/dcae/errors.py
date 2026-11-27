"""Structured domain errors for the Dynamic Capital Allocation Engine."""

from __future__ import annotations

from typing import Any, Dict, Optional


class DCAEError(Exception):
    """Base structured error with machine-readable code and context."""

    code: str = "DCAE_ERROR"
    http_status: int = 400

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        http_status: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code
        if http_status is not None:
            self.http_status = http_status
        self.message = message
        self.context: Dict[str, Any] = dict(context or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.code,
            "message": self.message,
            "context": self.context,
        }


class ValidationError(DCAEError):
    code = "VALIDATION_ERROR"
    http_status = 422


class PointInTimeViolationError(DCAEError):
    code = "POINT_IN_TIME_VIOLATION"
    http_status = 422


class StaleDataError(DCAEError):
    code = "STALE_DATA"
    http_status = 409


class InsufficientHistoryError(DCAEError):
    code = "INSUFFICIENT_HISTORY"
    http_status = 422


class ZeroVolatilityError(DCAEError):
    code = "ZERO_VOLATILITY"
    http_status = 422


class NonPSDCovarianceError(DCAEError):
    code = "NON_PSD_COVARIANCE"
    http_status = 422


class InfeasibleConstraintsError(DCAEError):
    code = "INFEASIBLE_CONSTRAINTS"
    http_status = 409


class ApprovalRequiredError(DCAEError):
    code = "APPROVAL_REQUIRED"
    http_status = 403


class DualControlViolationError(DCAEError):
    code = "DUAL_CONTROL_VIOLATION"
    http_status = 403


class LiveExecutionProhibitedError(DCAEError):
    code = "LIVE_EXECUTION_PROHIBITED"
    http_status = 403


class ProposalStateError(DCAEError):
    code = "INVALID_PROPOSAL_STATE"
    http_status = 409


class AuditIntegrityError(DCAEError):
    code = "AUDIT_INTEGRITY_ERROR"
    http_status = 500
