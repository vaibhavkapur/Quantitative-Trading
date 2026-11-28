"""Domain correctness tests for Fama-MacBeth cross-sectional premia and robust SEs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mfape.errors import RankDeficiencyError
from mfape.fama_macbeth import estimate_fama_macbeth
from mfape.io import load_table
from mfape.pipeline import build_aligned_panel
from mfape.schemas import AlignedPanel, AssetTimeSeriesResult, FactorModelSpec, TimeSeriesModelResult
from mfape.time_series import fit_time_series_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REF_DIR = PROJECT_ROOT / "data" / "fixtures" / "reference_hand_calculated"
SYN_DIR = PROJECT_ROOT / "data" / "fixtures" / "synthetic_ff3"


def _load_panel(directory: Path, factors: list[str]):
    return build_aligned_panel(
        asset_returns_df=load_table(directory / "asset_returns.csv"),
        factor_returns_df=load_table(directory / "factor_returns.csv"),
        risk_free_df=load_table(directory / "risk_free.csv"),
        memberships_df=load_table(directory / "portfolio_memberships.csv"),
        required_factors=factors,
        frequency="M",
        universe_policy="strict_balanced",
    )


def test_fama_macbeth_exact_hand_calculated_oracle() -> None:
    panel = _load_panel(REF_DIR, ["MKT_RF", "VAL"])
    oracle = json.loads((REF_DIR / "expected_oracle.json").read_text(encoding="utf-8"))
    exp = oracle["expected_two_factor"]

    spec = FactorModelSpec(name="TWO_FACTOR", factors=("MKT_RF", "VAL"))
    ts_res = fit_time_series_model(panel, spec, cov_type="OLS", min_ts_observations=6)
    fm_res = estimate_fama_macbeth(
        panel=panel,
        ts_result=ts_res,
        spec=spec,
        annualization_factor=12.0,
        fm_hac_maxlags=0,
        min_cs_assets=4,
    )

    premia = {p.name: p for p in fm_res.premia}
    assert np.isclose(premia["gamma_0"].estimate, exp["fm_gamma_0"], atol=1e-10)
    assert np.isclose(premia["MKT_RF"].estimate, exp["fm_lambda_MKT_RF"], atol=1e-10)
    assert np.isclose(premia["VAL"].estimate, exp["fm_lambda_VAL"], atol=1e-10)
    assert np.isclose(premia["MKT_RF"].fm_se, exp["fm_se_MKT_RF"], atol=1e-10)
    assert np.isclose(premia["VAL"].fm_se, exp["fm_se_VAL"], atol=1e-10)
    assert np.isclose(fm_res.pricing_metrics.cross_sectional_r2, exp["fm_cs_r2"], atol=1e-10)
    assert np.isclose(fm_res.pricing_metrics.rmse, exp["fm_pricing_rmse"], atol=1e-10)


def test_fama_macbeth_recovers_synthetic_ff3_premia_with_robust_errors() -> None:
    panel = _load_panel(SYN_DIR, ["MKT_RF", "SMB", "HML"])
    gt = json.loads((SYN_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    realized_means = gt["realized_factor_means_full_sample"]

    spec_ff3 = FactorModelSpec(name="FF3", factors=("MKT_RF", "SMB", "HML"))
    ts_ff3 = fit_time_series_model(panel, spec_ff3, cov_type="HAC", hac_maxlags=3, min_ts_observations=24)
    fm_ff3 = estimate_fama_macbeth(
        panel=panel,
        ts_result=ts_ff3,
        spec=spec_ff3,
        annualization_factor=12.0,
        fm_hac_maxlags=3,
        min_cs_assets=5,
    )

    premia = {p.name: p for p in fm_ff3.premia}

    # Cross-sectional factor premia should closely recover the realized factor sample means
    for f in ("MKT_RF", "SMB", "HML"):
        est = premia[f].estimate
        assert abs(est - realized_means[f]) < 0.0025, f"Premium mismatch for {f}: {est} vs {realized_means[f]}"
        assert premia[f].fm_se > 0.0
        assert premia[f].shanken_se is not None and premia[f].shanken_se > 0.0
        assert premia[f].hac_se > 0.0
        # Shanken correction factor c = lambda' Sigma_f^-1 lambda must be positive
        assert fm_ff3.shanken_correction_factor is not None and fm_ff3.shanken_correction_factor > 0.0

    # Zero-beta intercept should be close to 0 (not significantly far from 0)
    assert abs(premia["gamma_0"].estimate) < 0.003

    # High cross-sectional R^2 and much lower pricing error than CAPM
    spec_capm = FactorModelSpec(name="CAPM", factors=("MKT_RF",))
    ts_capm = fit_time_series_model(panel, spec_capm, cov_type="HAC", hac_maxlags=3, min_ts_observations=24)
    fm_capm = estimate_fama_macbeth(panel, ts_capm, spec_capm, fm_hac_maxlags=3, min_cs_assets=5)

    assert fm_ff3.pricing_metrics.cross_sectional_r2 > 0.85
    assert fm_ff3.pricing_metrics.cross_sectional_r2 > fm_capm.pricing_metrics.cross_sectional_r2 + 0.30
    assert fm_ff3.pricing_metrics.rmse < fm_capm.pricing_metrics.rmse * 0.5


def test_fama_macbeth_detects_rank_deficient_beta_matrix() -> None:
    # Construct a synthetic beta matrix where every asset has identical beta_MKT == 1.0 -> collinear with intercept!
    rng = np.random.default_rng(7)
    t_obs, n_assets = 24, 5
    dates = tuple(f"2020-{m:02d}-28" for m in range(1, 13)) + tuple(f"2021-{m:02d}-28" for m in range(1, 13))
    assets = tuple(f"A{i}" for i in range(n_assets))
    f_mat = rng.normal(0.01, 0.03, size=(t_obs, 1))
    exc = rng.normal(0.01, 0.02, size=(t_obs, n_assets))

    panel = AlignedPanel(
        dates=dates,
        assets=assets,
        factor_names=("MKT_RF",),
        excess_returns=exc,
        active_mask=np.ones((t_obs, n_assets), dtype=bool),
        factor_returns=f_mat,
        risk_free_rates=np.zeros(t_obs, dtype=np.float64),
        frequency="M",
        is_balanced=True,
        universe_changes=(),
    )

    spec = FactorModelSpec(name="CAPM", factors=("MKT_RF",), include_cs_intercept=True)
    # Force constant betas across all assets (exact collinearity with gamma_0 column of 1s)
    fake_asset_results = tuple(
        AssetTimeSeriesResult(
            asset_id=a,
            n_obs=t_obs,
            alpha=0.0,
            alpha_se=0.001,
            alpha_tstat=0.0,
            alpha_pvalue=1.0,
            betas={"MKT_RF": 1.0},
            beta_ses={"MKT_RF": 0.05},
            beta_tstats={"MKT_RF": 20.0},
            beta_pvalues={"MKT_RF": 0.0},
            r_squared=0.5,
            adj_r_squared=0.48,
            residual_std=0.01,
            mean_excess_return=0.01,
            predicted_mean_excess_return=0.01,
            pricing_error=0.0,
            residuals=tuple(0.0 for _ in range(t_obs)),
            dates=dates,
        )
        for a in assets
    )
    fake_ts = TimeSeriesModelResult(
        model_name="CAPM",
        factors=("MKT_RF",),
        cov_type="OLS",
        hac_maxlags=0,
        include_intercept=True,
        n_periods=t_obs,
        n_assets=n_assets,
        date_start=dates[0],
        date_end=dates[-1],
        factor_condition_number=1.0,
        asset_results=fake_asset_results,
        mean_r_squared=0.5,
        mean_adj_r_squared=0.48,
        mean_abs_alpha=0.0,
        rmse_alpha=0.0,
        grs_f_stat=None,
        grs_pvalue=None,
        grs_df1=None,
        grs_df2=None,
    )

    with pytest.raises(RankDeficiencyError, match="rank-deficient"):
        estimate_fama_macbeth(panel, fake_ts, spec, min_cs_assets=3)
