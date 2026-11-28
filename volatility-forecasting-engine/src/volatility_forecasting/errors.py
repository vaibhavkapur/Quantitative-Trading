"""Domain-specific exception hierarchy for the Volatility Forecasting Engine."""

from __future__ import annotations


class VolatilityEngineError(Exception):
    """Base exception for all volatility forecasting engine errors."""


class ConfigurationError(VolatilityEngineError):
    """Raised when configuration is missing, malformed, or semantically invalid."""


class DataContractError(VolatilityEngineError):
    """Raised when input or output datasets violate documented data contracts."""


class FutureLeakageError(VolatilityEngineError):
    """Raised when chronological ordering or train/test purge invariants are violated."""


class NegativeForecastError(VolatilityEngineError):
    """Raised when any model emits negative, NaN, or infinite variance/volatility forecasts."""


class ModelFitError(VolatilityEngineError):
    """Raised when a model cannot be fitted cleanly on the provided training window."""


class UnsupportedOptionError(VolatilityEngineError):
    """Raised when an unsupported model type, backend, horizon, or split scheme is requested."""
