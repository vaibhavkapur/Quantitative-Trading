"""Unit tests for domain contracts, config guards, and point-in-time validation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from alpha_engine.config import EngineConfig, UniversePolicy
from alpha_engine.contracts.models import (
    FundamentalRecord,
    MarketBar,
    SignalDefinition,
    SignalRecord,
)
from alpha_engine.contracts.validation import PointInTimeValidator
from alpha_engine.data.loader import PointInTimeDataLoader, ResearchDataset
from alpha_engine.errors import DataValidationError, ExecutionDisabledError, LookAheadViolationError


def test_engine_config_forbids_live_execution() -> None:
    with pytest.raises(ExecutionDisabledError):
        EngineConfig(config_version="1.0.0", execution_enabled=True)


def test_market_bar_rejects_naive_or_non_causal_timestamps() -> None:
    t0 = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
    with pytest.raises(ValueError):
        MarketBar(
            symbol="SYN001",
            event_ts_utc=datetime(2024, 1, 2, 21, 0),  # naive
            available_at_utc=t0,
            open=10.0,
            high=11.0,
            low=9.5,
            close=10.5,
            adjusted_close=10.5,
            volume=1000.0,
        )
    with pytest.raises(LookAheadViolationError):
        MarketBar(
            symbol="SYN001",
            event_ts_utc=t0,
            available_at_utc=t0 - timedelta(minutes=1),
            open=10.0,
            high=11.0,
            low=9.5,
            close=10.5,
            adjusted_close=10.5,
            volume=1000.0,
        )


def test_fundamental_and_signal_record_causality() -> None:
    period_end = datetime(2024, 3, 31, 21, 0, tzinfo=UTC)
    with pytest.raises(LookAheadViolationError):
        FundamentalRecord(
            symbol="SYN001",
            period_end_utc=period_end,
            event_ts_utc=period_end,
            available_at_utc=period_end - timedelta(days=1),
            earnings_ttm_usd=1e7,
            book_value_usd=1e8,
            market_cap_usd=2e8,
            roe=0.10,
        )
    with pytest.raises(LookAheadViolationError):
        SignalRecord(
            signal_id="mom_20_5",
            signal_version="1.0.0",
            symbol="SYN001",
            event_ts_utc=period_end,
            available_at_utc=period_end - timedelta(seconds=1),
            raw_value=0.5,
        )


def test_signal_definition_version_and_hash_change() -> None:
    d1 = SignalDefinition(
        signal_id="mom_20_5",
        name="Momentum",
        version="1.0.0",
        family="momentum",
        description="test",
        parameters={"lookback_days": 20},
        lookback_bars=20,
    )
    d2 = d1.model_copy(update={"version": "1.1.0"})
    assert d1.definition_hash != d2.definition_hash
    assert len(d1.definition_hash) == 16


def test_loader_verifies_checksums_and_as_of_cutoff(
    saved_fixtures_dir: Path,
    universe_policy: UniversePolicy,
    tmp_path: Path,
) -> None:
    loader = PointInTimeDataLoader(saved_fixtures_dir, universe_policy)
    full = loader.load()
    assert full.validation.passed

    # Cutoff midway through dataset must exclude future bars and fundamentals
    dates = full.bars.select("event_ts_utc").unique().sort("event_ts_utc")["event_ts_utc"].to_list()
    cutoff = dates[250] + timedelta(minutes=10)
    partial = loader.load(as_of_utc=cutoff)
    assert partial.bars.select("available_at_utc").max().item() <= cutoff
    assert partial.fundamentals.select("available_at_utc").max().item() <= cutoff

    # Tampered fixture must fail checksum validation
    tampered_dir = tmp_path / "tampered"
    tampered_dir.mkdir()
    for p in saved_fixtures_dir.iterdir():
        (tampered_dir / p.name).write_bytes(p.read_bytes())
    (tampered_dir / "market_bars.parquet").write_bytes(b"corrupted")
    with pytest.raises(DataValidationError, match="Checksum mismatch"):
        PointInTimeDataLoader(tampered_dir, universe_policy).load()


def test_pit_fundamental_asof_join_never_uses_unfiled_data(research_dataset: ResearchDataset) -> None:
    joined = PointInTimeDataLoader.attach_pit_fundamentals(
        research_dataset.bars, research_dataset.fundamentals
    )
    non_null = joined.filter(pl.col("filing_available_at_utc").is_not_null())
    assert non_null.height > 0
    leaks = non_null.filter(pl.col("filing_available_at_utc") > pl.col("available_at_utc"))
    assert leaks.height == 0


def test_validator_rejects_naive_datetime_column(
    research_dataset: ResearchDataset,
    universe_policy: UniversePolicy,
) -> None:
    validator = PointInTimeValidator(universe_policy)
    bad = research_dataset.bars.with_columns(pl.col("event_ts_utc").dt.replace_time_zone(None))
    with pytest.raises(DataValidationError, match="UTC timezone"):
        validator.validate_timestamps(bad, dataset_name="market_bars")
