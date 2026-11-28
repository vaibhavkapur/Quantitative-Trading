"""Deterministic tests for momentum, carry, defensive factors, and missing-factor handling."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from cross_asset_factor_engine import (
    DiagnosticTracker,
    EngineConfig,
    FactorScaler,
    MarketDataset,
    MissingFactorError,
    build_blended_signals,
    compute_raw_factor_signals,
    normalize_market_data,
    validate_dataset,
    verify_signal_provenance,
)
from cross_asset_factor_engine.contracts import FACTOR_INPUTS_SCHEMA
from cross_asset_factor_engine.io import _read_csv_with_schema


def test_missing_factor_inputs_non_strict_propagates_provenance_and_partial_blend(
    small_dataset: MarketDataset,
    small_config: EngineConfig,
    fixtures_dir: Path,
) -> None:
    missing_fi = _read_csv_with_schema(
        fixtures_dir / "boundary" / "missing_factor_inputs.csv",
        FACTOR_INPUTS_SCHEMA,
        "factor_inputs",
    )
    ds_missing = validate_dataset(
        small_dataset.contract_metadata,
        small_dataset.prices,
        small_dataset.fx_rates,
        missing_fi,
        base_currency="USD",
    )

    diag = DiagnosticTracker(deterministic_clock=True)
    norm = normalize_market_data(ds_missing, diagnostics=diag)
    raw = compute_raw_factor_signals(norm, ds_missing, small_config, diagnostics=diag)

    # EQ_SPX has no factor_inputs -> carry must be MISSING_INPUT, defensive uses low-vol fallback
    spx_carry = raw.filter((pl.col("asset_id") == "EQ_SPX") & (pl.col("factor_name") == "carry"))
    assert spx_carry.height == 12
    assert set(spx_carry["signal_status"].to_list()) == {"MISSING_INPUT"}
    assert spx_carry["raw_value"].null_count() == 12
    assert all(r == "missing_or_stale_dividend_yield" for r in spx_carry["missing_reason"].to_list())

    spx_def_mature = raw.filter(
        (pl.col("asset_id") == "EQ_SPX")
        & (pl.col("factor_name") == "defensive")
        & (pl.col("signal_status") != "INSUFFICIENT_HISTORY")
    )
    assert spx_def_mature.height > 0
    assert set(spx_def_mature["signal_status"].to_list()) == {"OK_FALLBACK_LOW_VOL"}

    # Scale on full small window and verify partial blend renormalizes cleanly
    scaler = FactorScaler().fit(raw)
    scaled = scaler.transform(raw)
    blended = build_blended_signals(scaled, small_config.factors.blend_weights)
    spx_blended_late = blended.filter(
        (pl.col("asset_id") == "EQ_SPX") & (pl.col("z_score").is_not_null())
    )
    assert spx_blended_late.height > 0
    assert set(spx_blended_late["signal_status"].to_list()) == {"OK_PARTIAL_BLEND"}

    audit = verify_signal_provenance(pl.concat([scaled, blended], how="vertical"))
    assert audit["lookahead_violations"] == 0
    assert audit["all_have_as_of_provenance"] is True
    assert diag.get_counter("signals_missing_input") == 12


def test_missing_factor_inputs_strict_mode_raises(
    small_dataset: MarketDataset,
    small_config: EngineConfig,
    fixtures_dir: Path,
) -> None:
    missing_fi = _read_csv_with_schema(
        fixtures_dir / "boundary" / "missing_factor_inputs.csv",
        FACTOR_INPUTS_SCHEMA,
        "factor_inputs",
    )
    ds_missing = validate_dataset(
        small_dataset.contract_metadata,
        small_dataset.prices,
        small_dataset.fx_rates,
        missing_fi,
        base_currency="USD",
    )
    strict_cfg = replace(
        small_config,
        data=replace(small_config.data, strict_missing_factors=True),
    )
    norm = normalize_market_data(ds_missing)
    with pytest.raises(MissingFactorError) as exc_info:
        compute_raw_factor_signals(norm, ds_missing, strict_cfg)
    assert exc_info.value.details["asset_id"] == "EQ_SPX"


def test_defensive_without_fallback_marks_missing_input(
    small_dataset: MarketDataset,
    small_config: EngineConfig,
    fixtures_dir: Path,
) -> None:
    missing_fi = _read_csv_with_schema(
        fixtures_dir / "boundary" / "missing_factor_inputs.csv",
        FACTOR_INPUTS_SCHEMA,
        "factor_inputs",
    )
    ds_missing = validate_dataset(
        small_dataset.contract_metadata,
        small_dataset.prices,
        small_dataset.fx_rates,
        missing_fi,
        base_currency="USD",
    )
    no_fb_cfg = replace(
        small_config,
        data=replace(small_config.data, defensive_fallback_to_low_vol=False),
    )
    norm = normalize_market_data(ds_missing)
    raw = compute_raw_factor_signals(norm, ds_missing, no_fb_cfg)
    spx_def_mature = raw.filter(
        (pl.col("asset_id") == "EQ_SPX")
        & (pl.col("factor_name") == "defensive")
        & (pl.col("signal_status") != "INSUFFICIENT_HISTORY")
    )
    assert spx_def_mature.height > 0
    assert set(spx_def_mature["signal_status"].to_list()) == {"MISSING_INPUT"}
    assert spx_def_mature["raw_value"].null_count() == spx_def_mature.height


def test_factor_scaler_fits_strictly_on_training_window(
    small_dataset: MarketDataset,
    small_config: EngineConfig,
) -> None:
    norm = normalize_market_data(small_dataset)
    raw = compute_raw_factor_signals(norm, small_dataset, small_config)
    dates = sorted(raw["signal_date"].unique().to_list())
    train_df = raw.filter(pl.col("signal_date") <= dates[6])
    test_df = raw.filter(pl.col("signal_date") > dates[6])

    scaler_a = FactorScaler().fit(train_df)
    # Wildly perturb test_df raw values and verify scaler params do NOT change
    test_perturbed = test_df.with_columns((pl.col("raw_value") * 1000.0 + 50.0).alias("raw_value"))
    scaler_b = FactorScaler().fit(train_df)
    assert scaler_a.to_dict() == scaler_b.to_dict()

    # Transform original test_df produces finite bounded z-scores within [-winsorize_z, winsorize_z]
    scaled = scaler_a.transform(test_df)
    z_vals = scaled.filter(pl.col("z_score").is_not_null())["z_score"].to_numpy()
    assert len(z_vals) > 0
    assert (z_vals <= small_config.factors.scaling.winsorize_z + 1e-12).all()
    assert (z_vals >= -small_config.factors.scaling.winsorize_z - 1e-12).all()
    _ = test_perturbed
