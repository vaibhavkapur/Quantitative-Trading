"""Integration tests for future label leakage prevention, chronological splits, and reproducibility."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from volatility_forecasting.config import (
    DataConfig,
    EngineConfig,
    EvaluationConfig,
    FeatureConfig,
    ModelSpec,
    OutputConfig,
    SplitConfig,
)
from volatility_forecasting.errors import FutureLeakageError
from volatility_forecasting.features import CAUSAL_FEATURE_COLUMNS, build_causal_dataset
from volatility_forecasting.fixtures import generate_multi_regime_fixture, write_all_fixtures
from volatility_forecasting.pipeline import evaluate_models_chronologically, run_pipeline
from volatility_forecasting.splits import assert_no_future_leakage, generate_chronological_splits


def test_future_label_leakage_detected_and_blocked() -> None:
    ret_df, bar_df = generate_multi_regime_fixture(n_days=200, bars_per_day=13, seed=19)
    dataset, _ = build_causal_dataset(ret_df, bar_df, horizons=(5,), expected_bars_per_session=13)

    # Case 1: embargo_steps < horizon_days must raise FutureLeakageError
    bad_split_cfg = SplitConfig(
        scheme="expanding",
        n_splits=2,
        min_train_size=100,
        test_size=30,
        embargo_steps=2,  # < horizon_days=5 -> leaks 3 future days of targets!
    )
    with pytest.raises(FutureLeakageError, match="embargo_steps"):
        generate_chronological_splits(dataset, bad_split_cfg, horizon_days=5)

    # Case 2: Direct unpurged indices where last training row's h=5 label extends into test period
    train_idx = list(range(0, 100))
    test_idx_unpurged = list(range(102, 132))  # row 99 has label_end_ts at index 104 >= 102!
    with pytest.raises(FutureLeakageError, match="Future label leakage detected"):
        assert_no_future_leakage(dataset, train_idx, test_idx_unpurged, horizon_days=5)

    # Case 3: Overlapping indices
    with pytest.raises(FutureLeakageError, match="overlap"):
        assert_no_future_leakage(dataset, list(range(0, 100)), list(range(95, 120)), horizon_days=5)

    # Case 4: Valid embargo >= horizon passes cleanly
    valid_split_cfg = SplitConfig(
        scheme="expanding", n_splits=2, min_train_size=100, test_size=30, embargo_steps=5
    )
    splits = generate_chronological_splits(dataset, valid_split_cfg, horizon_days=5)
    assert len(splits) == 2
    for sp in splits:
        assert sp.train_end_ts < sp.test_start_ts


def test_causal_invariance_under_future_perturbation() -> None:
    """Perturbing future returns after cutoff T must NOT change any feature or forecast at <= T."""
    ret_df, bar_df = generate_multi_regime_fixture(n_days=240, bars_per_day=13, seed=21)
    cutoff_idx = 180
    cutoff_date = ret_df["session_date"][cutoff_idx]

    # Extreme 20x perturbation strictly AFTER cutoff_date
    ret_perturbed = ret_df.with_columns(
        pl.when(pl.col("session_date") > cutoff_date)
        .then(pl.col("log_return") * 20.0)
        .otherwise(pl.col("log_return"))
        .alias("log_return")
    )

    ds_orig, _ = build_causal_dataset(ret_df, horizons=(5,))
    ds_pert, _ = build_causal_dataset(ret_perturbed, horizons=(5,))

    # 1. All causal features up to and including cutoff_idx must be bit-identical
    for col in CAUSAL_FEATURE_COLUMNS:
        np.testing.assert_array_equal(
            ds_orig[col][: cutoff_idx + 1].to_numpy(),
            ds_pert[col][: cutoff_idx + 1].to_numpy(),
        )

    # 2. Predictions on split whose test window ends BEFORE cutoff_idx must be bit-identical
    cfg = EngineConfig(
        version="1.0",
        run_name="causality_check",
        seed=42,
        annualization_factor=252,
        expected_bars_per_session=13,
        data=DataConfig(returns_path="unused.csv"),
        features=FeatureConfig(horizons=(5,), primary_horizon=5),
        splits=SplitConfig(
            scheme="expanding", n_splits=1, min_train_size=120, test_size=40, embargo_steps=5
        ),
        models=(
            ModelSpec(name="naive", type="naive"),
            ModelSpec(name="ewma", type="ewma", params={"decay": 0.94}),
            ModelSpec(name="garch11", type="garch11", params={"backend": "scipy"}),
            ModelSpec(name="ridge_har", type="ridge_har", params={"alpha": 1.0}),
            ModelSpec(name="hgbr", type="hgbr", params={"max_iter": 40}),
        ),
        evaluation=EvaluationConfig(strict_nonnegative=True),
        output=OutputConfig(reports_dir="reports"),
    )
    # Test ends at index 120 + 5 + 40 = 165, whose target labels end at 170 < cutoff_idx (180)
    _, comp1, int1, _, _, _ = evaluate_models_chronologically(ds_orig, cfg, horizon_days=5)
    _, comp2, int2, _, _, _ = evaluate_models_chronologically(ds_pert, cfg, horizon_days=5)

    assert comp1.equals(comp2)
    assert int1.equals(int2)


def test_fixed_splits_exact_reproducibility_and_baseline_comparison(tmp_path: Path) -> None:
    fixtures_dir = tmp_path / "fixtures"
    written = write_all_fixtures(fixtures_dir)
    reports_a = tmp_path / "reports_a"
    reports_b = tmp_path / "reports_b"

    def make_cfg(rep_dir: Path) -> EngineConfig:
        return EngineConfig(
            version="1.0",
            run_name="repro_test",
            seed=42,
            annualization_factor=252,
            expected_bars_per_session=13,
            data=DataConfig(
                returns_path=written["multi_regime_returns"],
                intraday_bars_path=written["multi_regime_intraday"],
            ),
            features=FeatureConfig(horizons=(1, 5), primary_horizon=5),
            splits=SplitConfig(scheme="rolling", n_splits=3, min_train_size=150, test_size=50),
            models=(
                ModelSpec(name="naive_persistence", type="naive", params={"lookback": 1}),
                ModelSpec(name="rolling_22d", type="rolling_historical", params={"window": 22}),
                ModelSpec(name="ewma_094", type="ewma", params={"decay": 0.94}),
                ModelSpec(name="garch_11", type="garch11", params={"backend": "scipy"}),
                ModelSpec(name="ridge_har", type="ridge_har", params={"alpha": 1.0}),
                ModelSpec(name="hgbr", type="hgbr", params={"max_iter": 60, "max_depth": 4}),
            ),
            evaluation=EvaluationConfig(strict_nonnegative=True),
            output=OutputConfig(reports_dir=str(rep_dir), save_predictions=True),
        )

    res_a = run_pipeline(make_cfg(reports_a), export_reports=True)
    res_b = run_pipeline(make_cfg(reports_b), export_reports=True)

    # Exact metric reproducibility across runs with fixed seed & inputs
    assert res_a.model_comparison.equals(res_b.model_comparison)
    assert res_a.forecast_intervals.equals(res_b.forecast_intervals)
    assert res_a.regime_diagnostics.equals(res_b.regime_diagnostics)

    # All forecasts nonnegative
    assert bool((res_a.model_comparison["min_pred_var"] >= 0.0).all())
    assert bool((res_a.forecast_intervals["pred_var"] >= 0.0).all())
    assert bool((res_a.forecast_intervals["lower_vol_80"] >= 0.0).all())
    assert bool(
        (
            res_a.forecast_intervals["upper_vol_80"] >= res_a.forecast_intervals["lower_vol_80"]
        ).all()
    )

    # Reports exported and naive baseline comparison present
    assert "qlike_vs_baseline_pct" in res_a.model_comparison.columns
    assert "mae_vol_vs_baseline_pct" in res_a.model_comparison.columns
    for artifact in ("model_comparison", "regime_diagnostics", "splits", "run_manifest", "run_summary", "forecast_intervals"):
        assert Path(res_a.output_paths[artifact]).exists()
