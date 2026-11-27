"""Unit tests for M2 causal factor library and feature matrix builder."""

from __future__ import annotations

import polars as pl
import pytest

from alpha_engine.config import UniversePolicy
from alpha_engine.contracts.validation import PointInTimeValidator
from alpha_engine.data.loader import ResearchDataset
from alpha_engine.errors import DataValidationError
from alpha_engine.factors import (
    EarningsYieldFactor,
    FeatureMatrixBuilder,
    MomentumFactor,
    default_factor_suite,
)


def test_default_factor_suite_computes_causal_utc_signals(
    research_dataset: ResearchDataset,
    universe_policy: UniversePolicy,
) -> None:
    validator = PointInTimeValidator(universe_policy)
    factors = default_factor_suite()
    assert len(factors) == 6

    for factor in factors:
        sig = factor.compute(research_dataset.bars, fundamentals_df=research_dataset.fundamentals)
        assert sig.height > 1000
        validator.validate_signal_causality(
            sig, research_dataset.bars, fundamentals_df=research_dataset.fundamentals
        )
        assert sig.filter(~pl.col("raw_value").is_finite()).height == 0
        # Determinism check
        sig2 = factor.compute(research_dataset.bars, fundamentals_df=research_dataset.fundamentals)
        assert sig.equals(sig2)


def test_earnings_yield_requires_fundamentals(research_dataset: ResearchDataset) -> None:
    with pytest.raises(DataValidationError, match="requires fundamentals_df"):
        EarningsYieldFactor().compute(research_dataset.bars, fundamentals_df=None)


def test_feature_matrix_builder_rejects_duplicate_ids_and_aligns_availability(
    research_dataset: ResearchDataset,
) -> None:
    with pytest.raises(DataValidationError, match="Duplicate signal_id"):
        FeatureMatrixBuilder([MomentumFactor(), MomentumFactor()])

    factors = default_factor_suite()
    wide, frames = FeatureMatrixBuilder(factors).build(
        research_dataset.bars, fundamentals_df=research_dataset.fundamentals
    )
    assert len(frames) == len(factors)
    for f in factors:
        assert f.definition.signal_id in wide.columns
    assert wide.filter(pl.col("available_at_utc") < pl.col("event_ts_utc")).height == 0
