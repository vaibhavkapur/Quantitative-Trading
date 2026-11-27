"""Structured domain errors for deterministic failure reporting."""

from __future__ import annotations

from typing import Any


class PMSError(Exception):
    """Base structured error with machine-readable code and context."""

    code: str = "PMS_ERROR"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.code,
            "message": self.message,
            "details": self.details,
        }


class ValidationError(PMSError):
    code = "VALIDATION_ERROR"


class MasterDataError(PMSError):
    code = "MASTER_DATA_ERROR"


class DuplicateEventError(PMSError):
    code = "DUPLICATE_EVENT"


class ImmutabilityViolationError(PMSError):
    code = "IMMUTABILITY_VIOLATION"


class InvariantViolationError(PMSError):
    code = "INVARIANT_VIOLATION"


class AuthorizationError(PMSError):
    code = "AUTHORIZATION_ERROR"


class ControlViolationError(PMSError):
    code = "CONTROL_VIOLATION"
