"""Domain correctness tests for time-series CAPM and multi-factor regressions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import statsmodels.api as sm

from mfape.errors import RankDeficiencyError
from mfape.io import load_table
from mfape.pipeline import build_aligned_panel
from mfape.schemas import FactorModelSpec
from mfape.time_series import fit_time_series_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REF_DIR = PROJECT_ROOT / "data" / "fixtures" / "reference_hand_calculated"
SYN_DIR = PROJECT_ROOT / "data" / "fixtures" / "synthetic_ff3"
RANK_DIR = PROJECT_ROOT / "data" / "fixtures" / "invalid" / "rank_deficient_factors"


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


def test_time_series_matches_hand_calculated_analytical_oracle() -> None:
    panel = _load_panel(REF_DIR, ["MKT_RF", "VAL"])
    oracle = json.loads((REF_DIR / "expected_oracle.json").read_text(encoding="utf-8"))

    # 1. Exact two-factor recovery
    spec_2f = FactorModelSpec(name="TWO_FACTOR", factors=("MKT_RF", "VAL"))
    res_2f = fit_time_series_model(panel, spec_2f, cov_type="OLS", min_ts_observations=6)

    assert res_2f.n_assets == 4
    assert res_2f.n_periods == 6
    assert np.isclose(res_2f.mean_r_squared, 1.0, atol=1e-10)
    assert np.isclose(res_2f.mean_abs_alpha, 0.0, atol=1e-10)
    for ar in res_2f.asset_results:
        true_b = oracle["true_betas"][ar.asset_id]
        assert np.isclose(ar.alpha, 0.0, atol=1e-10)
        assert np.isclose(ar.betas["MKT_RF"], true_b["MKT_RF"], atol=1e-10)
        assert np.isclose(ar.betas["VAL"], true_b["VAL"], atol=1e-10)

    # 2. Orthogonal CAPM exact alpha = beta_VAL * mean(VAL)
    spec_capm = FactorModelSpec(name="CAPM", factors=("MKT_RF",))
    res_capm = fit_time_series_model(panel, spec_capm, cov_type="OLS", min_ts_observations=6)
    for ar in res_capm.asset_results:
        expected_alpha = oracle["expected_capm"]["alphas"][ar.asset_id]
        expected_beta = oracle["expected_capm"]["betas_MKT_RF"][ar.asset_id]
        assert np.isclose(ar.alpha, expected_alpha, atol=1e-10)
        assert np.isclose(ar.betas["MKT_RF"], expected_beta, atol=1e-10)


def test_time_series_recovers_synthetic_ff3_betas_and_matches_statsmodels_oracle() -> None:
    panel = _load_panel(SYN_DIR, ["MKT_RF", "SMB", "HML", "MOM"])
    gt = json.loads((SYN_DIR / "ground_truth.json").read_text(encoding="utf-8"))

    spec_ff3 = FactorModelSpec(name="FF3", factors=("MKT_RF", "SMB", "HML"))
    res_ff3 = fit_time_series_model(panel, spec_ff3, cov_type="HAC", hac_maxlags=3, min_ts_observations=24)

    # Cross-check first asset directly against raw statsmodels HAC OLS oracle
    first_asset = res_ff3.asset_results[0]
    j = panel.assets.index(first_asset.asset_id)
    y = panel.excess_returns[:, j]
    x = sm.add_constant(panel.factor_returns[:, :3])
    sm_fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    assert np.isclose(first_asset.alpha, float(sm_fit.params[0]), atol=1e-12)
    assert np.isclose(first_asset.alpha_se, float(sm_fit.bse[0]), atol=1e-12)
    assert np.isclose(first_asset.betas["MKT_RF"], float(sm_fit.params[1]), atol=1e-12)
    assert np.isclose(first_asset.beta_ses["MKT_RF"], float(sm_fit.bse[1]), atol=1e-12)

    # Verify synthetic true beta recovery across all 25 portfolios (MAE < 0.04)
    beta_errors = []
    for ar in res_ff3.asset_results:
        tb = gt["true_betas"][ar.asset_id]
        for f in ("MKT_RF", "SMB", "HML"):
            beta_errors.append(abs(ar.betas[f] - tb[f]))
    assert float(np.mean(beta_errors)) < 0.04
    assert float(np.max(beta_errors)) < 0.15

    # True DGP is FF3 with zero alpha: GRS should NOT reject H0 (p > 0.05)
    assert res_ff3.grs_f_stat is not None
    assert res_ff3.grs_pvalue is not None
    assert res_ff3.grs_pvalue > 0.05

    # Compare against CAPM baseline: FF3 should have dramatically higher R^2 and lower |alpha|
    spec_capm = FactorModelSpec(name="CAPM", factors=("MKT_RF",))
    res_capm = fit_time_series_model(panel, spec_capm, cov_type="HAC", hac_maxlags=3, min_ts_observations=24)
    assert res_ff3.mean_r_squared > res_capm.mean_r_squared + 0.08
    assert res_ff3.mean_abs_alpha < res_capm.mean_abs_alpha * 0.5
    assert res_capm.grs_pvalue is not None and res_capm.grs_pvalue < 0.05
    assert res_capm.grs_f_stat is not None and res_capm.grs_f_stat > res_ff3.grs_f_stat


def test_rank_deficient_factors_raise_rank_deficiency_error() -> None:
    panel = _load_panel(RANK_DIR, ["MKT_RF", "VAL", "FAC_COLLINEAR"])
    spec = FactorModelSpec(name="COLLINEAR", factors=("MKT_RF", "VAL", "FAC_COLLINEAR"))
    with pytest.raises(RankDeficiencyError, match="rank-deficient"):
        fit_time_series_model(panel, spec, cov_type="OLS", min_ts_observations=6)
