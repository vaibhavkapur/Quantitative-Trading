"""Typed exceptions for the Statistical Arbitrage Research Platform."""

from __future__ import annotations


class StatArbError(Exception):
    """Base exception for all statistical arbitrage platform errors."""


class ConfigurationError(StatArbError):
    """Raised when configuration is missing, malformed, or violates domain invariants."""


class ContractValidationError(StatArbError):
    """Raised when input or output data violates documented schema or value contracts."""


class MissingDataError(ContractValidationError):
    """Raised when required price, borrow, or history data is missing."""


class BorrowUnavailableError(StatArbError):
    """Raised in strict mode when a required short locate is unavailable or exceeded."""


class LeakageViolationError(StatArbError):
    """Raised when training or selection routines receive timestamps beyond the split boundary."""


class ReconciliationError(StatArbError):
    """Raised when executed trades do not reconcile exactly to positions or cash/equity ledgers."""


class UnsupportedInputError(StatArbError):
    """Raised when an unsupported file format, estimator, or parameter combination is requested."""
