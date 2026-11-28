"""Integration and boundary tests covering missing RF, rank deficiency, period alignment,
universe changes, zero date leakage, and CAPM baseline OOS comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from mfape.engine import AssetPricingEngine
from mfape.errors import (
    MissingRiskFreeError,
    PeriodAlignmentError,
    RankDeficiencyError,
    UniverseChangeError,
)
from mfape.io import load_config, load_table
from mfape.schemas import EngineConfig, FactorModelSpec, SplitConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INVALID_DIR = PROJECT_ROOT / "data" / "fixtures" / "invalid"
UNIVERSE_DIR = PROJECT_ROOT / "data" / "fixtures" / "universe_change"


def _make_config(
    fixture_dir: Path,
    models: tuple[FactorModelSpec, ...],
    baseline: str = "CAPM",
    **overrides,
) -> EngineConfig:
    params = dict(
        config_version="1.0.0",
        run_label="integration_test",
        asset_returns_path=str(fixture_dir / "asset_returns.csv"),
        factor_returns_path=str(fixture_dir / "factor_returns.csv"),
        risk_free_path=str(fixture_dir / "risk_free.csv"),
        memberships_path=str(fixture_dir / "portfolio_memberships.csv"),
        models=models,
        baseline_model_name=baseline,
        cov_type="OLS",
        hac_maxlags=0,
        fm_hac_maxlags=0,
        min_ts_observations=4,
        min_cs_assets=4,
    )
    params.update(overrides)
    return EngineConfig(**params)


def test_missing_risk_free_raises_under_error_policy_and_recovers_under_drop_policy() -> None:
    engine = AssetPricingEngine()
    miss_dir = INVALID_DIR / "missing_rf"
    models = (FactorModelSpec(name="CAPM", factors=("MKT_RF",)),)

    cfg_strict = _make_config(miss_dir, models, rf_missing_policy="error")
    with pytest.raises(MissingRiskFreeError, match="Missing or non-finite risk-free") as exc_info:
        engine.run_from_config(cfg_strict, write_outputs=False)
    assert exc_info.value.details["missing_count"] == 2
    assert set(exc_info.value.details["missing_rf_dates"]) == {"2020-03-31", "2020-05-31"}

    cfg_drop = _make_config(miss_dir, models, rf_missing_policy="drop_period")
    out, _ = engine.run_from_config(cfg_drop, write_outputs=False)
    assert out.diagnostics["dropped_missing_rf_periods"] == 2
    assert out.diagnostics["aligned_periods"] == 4


def test_rank_deficient_factors_integration_raises_explicit_error() -> None:
    engine = AssetPricingEngine()
    rank_dir = INVALID_DIR / "rank_deficient_factors"
    models = (
        FactorModelSpec(name="CAPM", factors=("MKT_RF",)),
        FactorModelSpec(name="COLLINEAR", factors=("MKT_RF", "VAL", "FAC_COLLINEAR")),
    )
    cfg = _make_config(rank_dir, models)
    with pytest.raises(RankDeficiencyError, match="rank-deficient"):
        engine.run_from_config(cfg, write_outputs=False)


def test_misaligned_periods_raise_strict_and_align_inner() -> None:
    engine = AssetPricingEngine()
    mis_dir = INVALID_DIR / "misaligned_periods"
    models = (FactorModelSpec(name="CAPM", factors=("MKT_RF",)),)

    cfg_strict = _make_config(mis_dir, models, period_alignment_policy="strict")
    with pytest.raises(PeriodAlignmentError, match="Period alignment mismatch"):
        engine.run_from_config(cfg_strict, write_outputs=False)

    cfg_inner = _make_config(mis_dir, models, period_alignment_policy="inner")
    out, _ = engine.run_from_config(cfg_inner, write_outputs=False)
    assert out.diagnostics["dropped_unaligned_periods"] == 2
    assert out.diagnostics["aligned_periods"] == 5


def test_universe_changes_handled_across_all_three_policies() -> None:
    engine = AssetPricingEngine()
    models = (
        FactorModelSpec(name="CAPM", factors=("MKT_RF",)),
        FactorModelSpec(name="TWO_FACTOR", factors=("MKT_RF", "HML")),
    )

    # 1. strict_balanced rejects mid-sample exit/entry
    cfg_strict = _make_config(
        UNIVERSE_DIR,
        models,
        universe_policy="strict_balanced",
        min_ts_observations=12,
        min_cs_assets=4,
    )
    with pytest.raises(UniverseChangeError, match="Universe membership changed"):
        engine.run_from_config(cfg_strict, write_outputs=False)

    # 2. pit_active preserves point-in-time membership and records exact entries/exits
    cfg_pit = _make_config(
        UNIVERSE_DIR,
        models,
        universe_policy="pit_active",
        min_ts_observations=12,
        min_cs_assets=4,
    )
    out_pit, _ = engine.run_from_config(cfg_pit, write_outputs=False)
    assert out_pit.diagnostics["universe_change_periods"] == 1
    assert out_pit.diagnostics["universe_entries"] == 1
    assert out_pit.diagnostics["universe_exits"] == 1
    assert out_pit.diagnostics["dropped_inactive_membership_rows"] == 48  # 24 EXIT + 24 ENTRY
    assert out_pit.models[1].time_series_train.n_assets == 7
    assert "GRS test requires a balanced panel" in (out_pit.models[1].time_series_train.grs_note or "")

    # 3. min_obs_balanced keeps only the 5 core assets present in all 48 periods
    cfg_bal = _make_config(
        UNIVERSE_DIR,
        models,
        universe_policy="min_obs_balanced",
        min_ts_observations=12,
        min_cs_assets=4,
    )
    out_bal, _ = engine.run_from_config(cfg_bal, write_outputs=False)
    assert out_bal.diagnostics["assets_excluded_min_obs"] == 2
    assert out_bal.models[1].time_series_train.n_assets == 5
    assert out_bal.models[1].time_series_train.grs_f_stat is not None


def test_zero_date_leakage_perturbing_test_window_does_not_change_train_estimates() -> None:
    """Hard empirical invariance test: arbitrarily corrupting test-window returns MUST NOT
    change any training-window time-series beta, alpha, or Fama-MacBeth premium.
    """
    cfg = load_config(PROJECT_ROOT / "config" / "default_config.json")
    engine = AssetPricingEngine()

    assets_df = load_table(PROJECT_ROOT / cfg.asset_returns_path)
    factors_df = load_table(PROJECT_ROOT / cfg.factor_returns_path)
    rf_df = load_table(PROJECT_ROOT / cfg.risk_free_path)
    mem_df = load_table(PROJECT_ROOT / cfg.memberships_path)

    out_clean = engine.run_on_frames(assets_df, factors_df, rf_df, mem_df, cfg)

    # Heavily perturb all test-window (>= 2020-01-31) asset returns and factor returns
    perturbed_assets = assets_df.with_columns(
        pl.when(pl.col("date") >= "2020-01-31")
        .then(pl.col("total_return") + 0.50)
        .otherwise(pl.col("total_return"))
        .alias("total_return")
    )
    perturbed_factors = factors_df.with_columns(
        pl.when(pl.col("date") >= "2020-01-31")
        .then(pl.col("MKT_RF") - 0.25)
        .otherwise(pl.col("MKT_RF"))
        .alias("MKT_RF")
    )

    out_perturbed = engine.run_on_frames(perturbed_assets, perturbed_factors, rf_df, mem_df, cfg)

    # Every single training-window time-series and Fama-MacBeth output must match bit-for-bit!
    for m_clean, m_pert in zip(out_clean.models, out_perturbed.models):
        assert m_clean.time_series_train == m_pert.time_series_train
        assert m_clean.fama_macbeth_train == m_pert.fama_macbeth_train
        # Meanwhile OOS evaluation MUST change since test data was perturbed
        assert m_clean.oos_evaluation != m_pert.oos_evaluation


def test_end_to_end_demo_ff3_outperforms_capm_baseline_in_and_out_of_sample() -> None:
    engine = AssetPricingEngine()
    out, _ = engine.run_from_config_file(
        PROJECT_ROOT / "config" / "default_config.json",
        base_dir=PROJECT_ROOT,
        write_outputs=False,
    )

    assert out.comparison.baseline_model_name == "CAPM"
    by_name = {r["model_name"]: r for r in out.comparison.rows}
    capm = by_name["CAPM"]
    ff3 = by_name["FF3"]

    assert ff3["ts_mean_r2"] > capm["ts_mean_r2"]
    assert ff3["ts_mean_abs_alpha"] < capm["ts_mean_abs_alpha"]
    assert ff3["fm_cs_r2"] > capm["fm_cs_r2"]
    assert ff3["fm_pricing_rmse"] < capm["fm_pricing_rmse"]
    assert ff3["oos_cs_rmse"] < capm["oos_cs_rmse"]
    assert ff3["oos_ts_r2_zero_alpha"] > capm["oos_ts_r2_zero_alpha"]
    assert ff3["delta_oos_cs_rmse_vs_baseline"] < 0.0
    assert ff3["delta_oos_ts_r2_vs_baseline"] > 0.0
