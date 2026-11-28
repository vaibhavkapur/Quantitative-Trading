"""Unit and reference-oracle tests for cointegration, hedge ratios, and multiple testing."""

from __future__ import annotations

import math

import numpy as np
import pytest

from statarb.cointegration import (
    adjust_pvalues,
    compute_adf_test,
    estimate_half_life,
    estimate_johansen_basket,
    estimate_ols_hedge_ratio,
    estimate_tls_hedge_ratio,
    test_engle_granger_pair,
)
from statarb.fixtures import (
    generate_reference_pair_fixture,
    generate_synthetic_universe_fixture,
)


def test_hand_calculated_ols_and_tls_oracle() -> None:
    """Verify exact recovery on a noiseless linear relationship y = 3.0 + 2.5 * x."""
    x = np.linspace(10.0, 50.0, 100)
    y = 3.0 + 2.5 * x

    alpha_ols, beta_ols = estimate_ols_hedge_ratio(y, x)
    alpha_tls, beta_tls = estimate_tls_hedge_ratio(y, x)

    assert alpha_ols == pytest.approx(3.0, abs=1e-10)
    assert beta_ols == pytest.approx(2.5, abs=1e-10)
    assert alpha_tls == pytest.approx(3.0, abs=1e-10)
    assert beta_tls == pytest.approx(2.5, abs=1e-10)


def test_hand_calculated_ar1_half_life_oracle() -> None:
    """Verify exact half-life recovery on deterministic geometric decay s_t = 0.8^t."""
    phi = 0.80
    expected_hl = -math.log(2.0) / math.log(phi)  # ~3.10628 days
    s = np.array([10.0 * (phi**t) for t in range(80)], dtype=np.float64)

    hl = estimate_half_life(s)
    assert hl == pytest.approx(expected_hl, rel=1e-6)


def test_non_mean_reverting_series_returns_infinite_half_life() -> None:
    """Explosive or drifting series must yield infinite half-life."""
    explosive = np.array([1.02**t for t in range(60)], dtype=np.float64)
    assert math.isinf(estimate_half_life(explosive))


def test_reference_fixture_cointegrated_vs_independent_series() -> None:
    """Required scenario: known cointegrated pair passes; independent random walks fail."""
    bundle = generate_reference_pair_fixture(seed=7, n_days=260)
    piv = bundle.prices.pivot(on="symbol", index="date", values="close_adj").sort("date")

    y = piv["REF_Y"].to_numpy()
    x = piv["REF_X"].to_numpy()
    ind_a = piv["REF_IND_A"].to_numpy()
    ind_b = piv["REF_IND_B"].to_numpy()

    gt = bundle.ground_truth["cointegrated_pair"]
    coint_res = test_engle_granger_pair(y, x, estimator="ols")
    tls_res = test_engle_granger_pair(y, x, estimator="tls")
    ind_res = test_engle_granger_pair(ind_a, ind_b, estimator="ols")

    # Cointegrated pair recovers true parameters within documented finite-sample tolerances
    assert coint_res["beta"] == pytest.approx(gt["true_beta"], abs=0.05)
    assert tls_res["beta"] == pytest.approx(gt["true_beta"], abs=0.05)
    assert coint_res["alpha"] == pytest.approx(gt["true_alpha"], abs=2.5)
    assert coint_res["adf_pvalue"] < 0.01
    assert tls_res["adf_pvalue"] < 0.01
    assert 2.0 <= coint_res["half_life_days"] <= 10.0

    # Independent random walks must NOT appear strongly cointegrated
    assert ind_res["adf_pvalue"] > 0.10


def test_johansen_basket_vs_independent_random_walks() -> None:
    """3-asset planted basket is detected by Johansen; 3 independent RWs are rejected."""
    prices, _, _ = generate_synthetic_universe_fixture(seed=42, n_days=400)
    piv = prices.pivot(on="symbol", index="date", values="close_adj").sort("date")

    basket_mat = np.log(piv.select(["FIN_A", "FIN_B", "FIN_C"]).to_numpy())
    rw_mat = np.log(piv.select(["IND_RW1", "IND_RW2", "IND_RW3"]).to_numpy())

    b_res = estimate_johansen_basket(basket_mat, ("FIN_A", "FIN_B", "FIN_C"))
    rw_res = estimate_johansen_basket(rw_mat, ("IND_RW1", "IND_RW2", "IND_RW3"))

    assert b_res is not None
    assert b_res["johansen_significant_95"] is True
    assert b_res["adf_pvalue"] < 0.02
    assert 2.0 <= b_res["half_life_days"] <= 25.0
    # True weights are approximately (1.0, -0.65, -0.45)
    assert b_res["weights"][0] == pytest.approx(1.0, abs=1e-8)
    assert b_res["weights"][1] == pytest.approx(-0.65, abs=0.15)
    assert b_res["weights"][2] == pytest.approx(-0.45, abs=0.15)

    # Independent random walks fail Johansen 95% trace threshold or stationarity
    assert rw_res is None or (not rw_res["johansen_significant_95"]) or rw_res["adf_pvalue"] > 0.05


def test_multiple_testing_adjust_pvalues_exact_oracle() -> None:
    """Verify Benjamini-Hochberg and Bonferroni against hand-calculated reference values."""
    raw = [0.005, 0.04, 0.02, 0.50]

    # Bonferroni: m=4 -> [0.02, 0.16, 0.08, 1.0]
    bonf = adjust_pvalues(raw, method="bonferroni")
    assert bonf == pytest.approx([0.02, 0.16, 0.08, 1.0], abs=1e-12)

    # BH sorted: p=(0.005, 0.02, 0.04, 0.50), ranks=(1,2,3,4)
    # raw*m/r = (0.02, 0.04, 0.053333..., 0.50) -> monotonic -> map back to original order
    bh = adjust_pvalues(raw, method="bh")
    assert bh == pytest.approx([0.02, 0.04 * 4 / 3, 0.04, 0.50], abs=1e-12)

    # None pass-through
    none_adj = adjust_pvalues(raw, method="none")
    assert none_adj == pytest.approx(raw, abs=1e-12)


def test_adf_constant_series_safe_fallback() -> None:
    """Zero-variance constant series should not crash and must return p=1.0."""
    stat, pval = compute_adf_test(np.ones(50))
    assert stat == 0.0
    assert pval == 1.0
