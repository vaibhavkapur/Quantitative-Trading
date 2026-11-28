"""Tests for data contracts, schema validation, and excess-return alignment pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from mfape.errors import (
    DataContractError,
    DateLeakageError,
    InsufficientObservationsError,
    MissingRiskFreeError,
    PeriodAlignmentError,
    UniverseChangeError,
)
from mfape.io import load_table
from mfape.pipeline import build_aligned_panel, split_panel
from mfape.schemas import DiagnosticCounters, SplitConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REF_DIR = PROJECT_ROOT / "data" / "fixtures" / "reference_hand_calculated"


def _load_ref_tables():
    return (
        load_table(REF_DIR / "asset_returns.csv"),
        load_table(REF_DIR / "factor_returns.csv"),
        load_table(REF_DIR / "risk_free.csv"),
        load_table(REF_DIR / "portfolio_memberships.csv"),
    )


def test_reference_alignment_computes_exact_excess_returns() -> None:
    assets_df, factors_df, rf_df, mem_df = _load_ref_tables()
    oracle = json.loads((REF_DIR / "expected_oracle.json").read_text(encoding="utf-8"))
    diag = DiagnosticCounters()

    panel = build_aligned_panel(
        asset_returns_df=assets_df,
        factor_returns_df=factors_df,
        risk_free_df=rf_df,
        memberships_df=mem_df,
        required_factors=["MKT_RF", "VAL"],
        frequency="M",
        universe_policy="strict_balanced",
        rf_missing_policy="error",
        period_alignment_policy="strict",
        diagnostics=diag,
    )

    assert panel.dates == tuple(oracle["dates"])
    assert panel.assets == ("P1", "P2", "P3", "P4")
    assert panel.is_balanced is True
    assert diag.aligned_periods == 6
    assert diag.aligned_assets == 4
    assert diag.aligned_rows == 24
    assert np.allclose(panel.risk_free_rates, oracle["rf"])

    # Verify excess_return = beta_mkt * MKT_RF + beta_val * VAL (exact zero noise)
    for j, aid in enumerate(panel.assets):
        b = oracle["true_betas"][aid]
        expected_exc = (
            b["MKT_RF"] * panel.factor_returns[:, 0] + b["VAL"] * panel.factor_returns[:, 1]
        )
        assert np.allclose(panel.excess_returns[:, j], expected_exc, atol=1e-9)


def test_missing_required_column_raises_data_contract_error() -> None:
    assets_df, factors_df, rf_df, mem_df = _load_ref_tables()
    bad_assets = assets_df.drop("total_return")
    with pytest.raises(DataContractError, match="missing required column"):
        build_aligned_panel(bad_assets, factors_df, rf_df, mem_df, ["MKT_RF"])


def test_duplicate_asset_date_keys_raise_data_contract_error() -> None:
    assets_df, factors_df, rf_df, mem_df = _load_ref_tables()
    dup_assets = pl.concat([assets_df, assets_df.head(1)])
    with pytest.raises(DataContractError, match="duplicate key"):
        build_aligned_panel(dup_assets, factors_df, rf_df, mem_df, ["MKT_RF"])


def test_null_or_nonfinite_factor_raises_data_contract_error() -> None:
    assets_df, factors_df, rf_df, mem_df = _load_ref_tables()
    bad_factors = factors_df.with_columns(
        pl.when(pl.col("date") == "2020-03-31")
        .then(None)
        .otherwise(pl.col("MKT_RF"))
        .alias("MKT_RF")
    )
    with pytest.raises(DataContractError, match="null/NaN/infinite"):
        build_aligned_panel(assets_df, bad_factors, rf_df, mem_df, ["MKT_RF"])


def test_zero_active_memberships_raises_insufficient_observations() -> None:
    assets_df, factors_df, rf_df, mem_df = _load_ref_tables()
    inactive_mem = mem_df.with_columns(pl.lit(False).alias("in_universe"))
    with pytest.raises(InsufficientObservationsError, match="Zero active asset"):
        build_aligned_panel(assets_df, factors_df, rf_df, inactive_mem, ["MKT_RF"])


def test_split_panel_enforces_strict_no_date_leakage() -> None:
    assets_df, factors_df, rf_df, mem_df = _load_ref_tables()
    panel = build_aligned_panel(assets_df, factors_df, rf_df, mem_df, ["MKT_RF", "VAL"])

    # Overlapping boundary: test_start == train_end -> DateLeakageError
    overlap_split = SplitConfig(
        train_start="2020-01-31",
        train_end="2020-03-31",
        test_start="2020-03-31",
        test_end="2020-06-30",
    )
    with pytest.raises(DateLeakageError, match="Date leakage"):
        split_panel(panel, overlap_split)

    # Reversed boundary: test_start < train_end -> DateLeakageError
    reversed_split = SplitConfig(
        train_start="2020-01-31",
        train_end="2020-04-30",
        test_start="2020-02-29",
        test_end="2020-06-30",
    )
    with pytest.raises(DateLeakageError, match="Date leakage"):
        split_panel(panel, reversed_split)

    # Valid strictly chronological split
    valid_split = SplitConfig(
        train_start="2020-01-31",
        train_end="2020-03-31",
        test_start="2020-04-30",
        test_end="2020-06-30",
    )
    train_p, test_p = split_panel(panel, valid_split)
    assert test_p is not None
    assert train_p.dates[-1] < test_p.dates[0]
    assert len(train_p.dates) == 3
    assert len(test_p.dates) == 3


def test_period_cadence_violation_raises_period_alignment_error() -> None:
    assets_df, factors_df, rf_df, mem_df = _load_ref_tables()
    # Replace 2020-02-29 with 2020-02-15 and keep 2020-01-31 -> gap is 15 days (< 27 for M)
    bad_dates_assets = assets_df.with_columns(
        pl.when(pl.col("date") == "2020-02-29").then(pl.lit("2020-02-15")).otherwise(pl.col("date")).alias("date")
    )
    bad_dates_factors = factors_df.with_columns(
        pl.when(pl.col("date") == "2020-02-29").then(pl.lit("2020-02-15")).otherwise(pl.col("date")).alias("date")
    )
    bad_dates_rf = rf_df.with_columns(
        pl.when(pl.col("date") == "2020-02-29").then(pl.lit("2020-02-15")).otherwise(pl.col("date")).alias("date")
    )
    bad_dates_mem = mem_df.with_columns(
        pl.when(pl.col("date") == "2020-02-29").then(pl.lit("2020-02-15")).otherwise(pl.col("date")).alias("date")
    )
    with pytest.raises(PeriodAlignmentError, match="period cadence"):
        build_aligned_panel(
            bad_dates_assets, bad_dates_factors, bad_dates_rf, bad_dates_mem, ["MKT_RF"], frequency="M"
        )
