"""Tests for canonical data contracts, schema validation, and unsupported/invalid inputs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from cross_asset_factor_engine import (
    CurrencyConversionError,
    MarketDataset,
    RollNormalizationError,
    SchemaValidationError,
    TimestampGateError,
    UnsupportedAssetError,
    load_dataset_from_dir,
    validate_dataset,
)
from cross_asset_factor_engine.contracts import validate_prices
from cross_asset_factor_engine.io import _read_csv_with_schema
from cross_asset_factor_engine.contracts import PRICES_SCHEMA


def test_valid_fixtures_pass_contract_validation(small_dataset: MarketDataset, multi_year_dataset: MarketDataset) -> None:
    assert small_dataset.contract_metadata.height == 4
    assert small_dataset.dataset_hash
    assert multi_year_dataset.contract_metadata.height == 9
    assert multi_year_dataset.prices.height == 9 * 1008


def test_missing_columns_fixture_raises_schema_error(fixtures_dir: Path) -> None:
    with pytest.raises(SchemaValidationError) as exc_info:
        _read_csv_with_schema(
            fixtures_dir / "invalid" / "missing_columns_prices.csv",
            PRICES_SCHEMA,
            "prices",
        )
    assert "publication_ts" in str(exc_info.value)


def test_negative_price_fixture_raises_schema_error(
    small_dataset: MarketDataset, fixtures_dir: Path
) -> None:
    bad_px = _read_csv_with_schema(
        fixtures_dir / "invalid" / "negative_price.csv",
        PRICES_SCHEMA,
        "prices",
    )
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_prices(bad_px, small_dataset.contract_metadata)
    assert "non-positive" in str(exc_info.value)


def test_invalid_roll_transition_fixture_raises_roll_error(
    small_dataset: MarketDataset, fixtures_dir: Path
) -> None:
    bad_roll_px = _read_csv_with_schema(
        fixtures_dir / "invalid" / "invalid_roll_transition.csv",
        PRICES_SCHEMA,
        "prices",
    )
    with pytest.raises(RollNormalizationError):
        validate_prices(bad_roll_px, small_dataset.contract_metadata)


def test_missing_fx_currency_raises_currency_error(small_dataset: MarketDataset) -> None:
    fx_no_eur = small_dataset.fx_rates.filter(pl.col("currency") != "EUR")
    with pytest.raises(CurrencyConversionError) as exc_info:
        validate_dataset(
            small_dataset.contract_metadata,
            small_dataset.prices,
            fx_no_eur,
            small_dataset.factor_inputs,
            base_currency="USD",
        )
    assert "EUR" in str(exc_info.value)


def test_unsupported_asset_class_raises_error(small_dataset: MarketDataset) -> None:
    bad_meta = small_dataset.contract_metadata.with_columns(
        pl.when(pl.col("asset_id") == "EQ_SPX")
        .then(pl.lit("crypto"))
        .otherwise(pl.col("asset_class"))
        .alias("asset_class")
    )
    with pytest.raises(UnsupportedAssetError):
        validate_dataset(
            bad_meta,
            small_dataset.prices,
            small_dataset.fx_rates,
            small_dataset.factor_inputs,
        )


def test_publication_before_as_of_raises_timestamp_gate_error(small_dataset: MarketDataset) -> None:
    bad_px = small_dataset.prices.with_columns(
        pl.lit(datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc)).alias("publication_ts")
    )
    with pytest.raises(TimestampGateError):
        validate_prices(bad_px, small_dataset.contract_metadata)


def test_missing_dataset_directory_raises_schema_error(tmp_path: Path) -> None:
    with pytest.raises(SchemaValidationError):
        load_dataset_from_dir(tmp_path / "does_not_exist")
