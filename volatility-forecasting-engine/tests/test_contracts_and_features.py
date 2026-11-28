"""Unit tests for data contracts, hand-calculated reference fixture, missing bars, overnight gaps, and constant returns."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from volatility_forecasting.contracts import (
    read_table,
    validate_intraday_bars_df,
    validate_labels_df,
    validate_returns_df,
)
from volatility_forecasting.errors import DataContractError
from volatility_forecasting.evaluation import qlike_loss
from volatility_forecasting.features import (
    CAUSAL_FEATURE_COLUMNS,
    build_causal_dataset,
    compute_daily_realized_variance,
)
from volatility_forecasting.fixtures import (
    generate_constant_returns_fixture,
    generate_hand_calculated_reference_fixture,
)


def test_hand_calculated_reference_fixture_exact_match() -> None:
    ret_df, bar_df, expected = generate_hand_calculated_reference_fixture()
    dataset, diag = build_causal_dataset(
        ret_df,
        bar_df,
        horizons=(1, 2),
        annualization_factor=252,
        expected_bars_per_session=2,
        include_overnight_gap=True,
    )

    assert diag.n_rows == 6
    assert diag.n_sessions_with_intraday == 6
    assert diag.n_sessions_fallback_daily == 0

    np.testing.assert_allclose(dataset["daily_rv"].to_numpy(), expected["daily_rv"], rtol=1e-9, atol=1e-12)

    h1_vals = dataset["target_rv_h1"].to_list()
    assert h1_vals[-1] is None
    np.testing.assert_allclose(h1_vals[:-1], expected["target_rv_h1"][:-1], rtol=1e-9, atol=1e-12)

    h2_vals = dataset["target_rv_h2"].to_list()
    assert h2_vals[-2:] == [None, None]
    np.testing.assert_allclose(h2_vals[:-2], expected["target_rv_h2"][:-2], rtol=1e-9, atol=1e-12)

    assert dataset["rv_5d"][4] == pytest.approx(expected["rv_5d_day4"], rel=1e-9, abs=1e-12)
    for col in CAUSAL_FEATURE_COLUMNS:
        assert col in dataset.columns


def test_constant_returns_zero_variance_clean() -> None:
    const_df = generate_constant_returns_fixture(n_days=50, constant_return=0.0)
    dataset, diag = build_causal_dataset(const_df, horizons=(1, 5))

    assert diag.n_rows == 50
    np.testing.assert_array_equal(dataset["daily_rv"].to_numpy(), np.zeros(50))
    np.testing.assert_array_equal(dataset["rv_5d"].to_numpy(), np.zeros(50))
    np.testing.assert_array_equal(dataset["vol_1d"].to_numpy(), np.zeros(50))

    for col in CAUSAL_FEATURE_COLUMNS:
        vals = dataset[col].to_numpy()
        assert np.all(np.isfinite(vals)), f"Feature {col} had non-finite values on constant returns"

    valid_h1 = dataset.filter(pl.col("target_rv_h1").is_not_null())
    y_true = valid_h1["target_rv_h1"].to_numpy()
    assert qlike_loss(y_true, np.zeros_like(y_true)) == pytest.approx(0.0, abs=1e-12)


def test_missing_bars_partial_and_full_session_fallback() -> None:
    ret_df, bar_df, expected = generate_hand_calculated_reference_fixture()
    # Drop bar 0 of day 1 (partial missing session) and ALL bars of day 3 (full missing session)
    dates = ret_df["session_date"].to_list()
    day1_str = dates[1]
    day3_str = dates[3]

    filtered_bars = bar_df.filter(
        ~(
            ((pl.col("session_date") == day1_str) & (pl.col("timestamp").str.contains("11:00:00")))
            | (pl.col("session_date") == day3_str)
        )
    )

    daily_df, diag = compute_daily_realized_variance(
        ret_df,
        filtered_bars,
        expected_bars_per_session=2,
        include_overnight_gap=True,
        min_bar_coverage_warn=0.80,
    )

    assert diag.n_sessions_with_intraday == 5
    assert diag.n_sessions_fallback_daily == 1
    assert diag.n_sessions_low_coverage == 2  # day 1 (0.5 coverage) and day 3 (0.0 coverage)

    assert daily_df["intraday_bar_coverage"][1] == pytest.approx(0.5)
    assert daily_df["used_fallback_daily_rv"][1] is False
    assert daily_df["intraday_bar_coverage"][3] == pytest.approx(0.0)
    assert daily_df["used_fallback_daily_rv"][3] is True

    # Day 3 fallback equals daily close-to-close squared log_return (day 3 had +0.02 and -0.02 => total log_ret = 0.0)
    assert daily_df["daily_rv"][3] == pytest.approx(float(ret_df["log_return"][3]) ** 2, abs=1e-12)


def test_overnight_gap_inclusion_vs_exclusion() -> None:
    ret_df, bar_df, _ = generate_hand_calculated_reference_fixture()

    with_gap, diag_with = compute_daily_realized_variance(
        ret_df, bar_df, expected_bars_per_session=2, include_overnight_gap=True
    )
    without_gap, _ = compute_daily_realized_variance(
        ret_df, bar_df, expected_bars_per_session=2, include_overnight_gap=False
    )

    # Day 1 has overnight return = +0.02 => gap_sq = 4e-4
    assert diag_with.n_sessions_with_overnight_gap == 2  # day 1 and day 4
    assert with_gap["overnight_gap_sq"][1] == pytest.approx(4e-4, rel=1e-9)
    assert with_gap["daily_rv"][1] - without_gap["daily_rv"][1] == pytest.approx(4e-4, rel=1e-9)

    # Also verify overnight gap inference when overnight_return column is omitted
    ret_no_ov = ret_df.drop("overnight_return")
    inferred, _ = compute_daily_realized_variance(
        ret_no_ov, bar_df, expected_bars_per_session=2, include_overnight_gap=True
    )
    assert inferred["overnight_gap_sq"][1] == pytest.approx(4e-4, rel=1e-6)
    assert inferred["daily_rv"][1] == pytest.approx(6e-4, rel=1e-6)


def test_data_contract_validation_errors(tmp_path: Path) -> None:
    # Missing required column
    with pytest.raises(DataContractError, match="missing required columns"):
        validate_returns_df(pl.DataFrame({"asset_id": ["A"], "timestamp": ["2024-01-02T16:00:00"]}))

    # Duplicate timestamps
    dup = pl.DataFrame(
        {
            "asset_id": ["A", "A"],
            "timestamp": ["2024-01-02T16:00:00", "2024-01-02T16:00:00"],
            "log_return": [0.01, 0.02],
        }
    )
    with pytest.raises(DataContractError, match="duplicate"):
        validate_returns_df(dup)

    # NaN log_return
    nan_df = pl.DataFrame(
        {"asset_id": ["A"], "timestamp": ["2024-01-02T16:00:00"], "log_return": [float("nan")]}
    )
    with pytest.raises(DataContractError, match="NaN or infinite"):
        validate_returns_df(nan_df)

    # OHLC ordering violation
    bad_ohlc = pl.DataFrame(
        {
            "asset_id": ["A"],
            "timestamp": ["2024-01-02T10:00:00"],
            "session_date": ["2024-01-02"],
            "open": [100.0],
            "high": [95.0],
            "low": [98.0],
            "close": [99.0],
            "volume": [100.0],
        }
    )
    with pytest.raises(DataContractError, match="OHLC ordering"):
        validate_intraday_bars_df(bad_ohlc)

    # Negative realized variance in precomputed labels
    bad_lbl = pl.DataFrame(
        {
            "asset_id": ["A"],
            "timestamp": ["2024-01-02T16:00:00"],
            "horizon_days": [5],
            "realized_variance": [-0.001],
            "realized_vol": [0.1],
        }
    )
    with pytest.raises(DataContractError, match="finite and nonnegative"):
        validate_labels_df(bad_lbl)

    # Unsupported file extension
    bad_ext = tmp_path / "data.txt"
    bad_ext.write_text("hello", encoding="utf-8")
    with pytest.raises(DataContractError, match="Unsupported input file extension"):
        read_table(bad_ext)
