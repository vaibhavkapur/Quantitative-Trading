"""Canonical domain models and validation for the Performance Attribution Engine."""

from pae.domain.models import (
    AttributionInputBatch,
    BenchmarkObservation,
    CashFlow,
    DailyNavSnapshot,
    DataQualityException,
    FactorObservation,
    FeeAdjustment,
    FxRateObservation,
    PositionSnapshot,
    TradeExecution,
)
from pae.domain.validation import validate_input_batch

__all__ = [
    "AttributionInputBatch",
    "BenchmarkObservation",
    "CashFlow",
    "DailyNavSnapshot",
    "DataQualityException",
    "FactorObservation",
    "FeeAdjustment",
    "FxRateObservation",
    "PositionSnapshot",
    "TradeExecution",
    "validate_input_batch",
]
