"""Domain correctness tests for DSR, PSR, effective trials, and CSCV PBO.

Covers required scenarios:
  - Null strategies (multiple-testing false discovery deflation)
  - Correlated trials (effective trial count reduction)
  - Known signal (high DSR, low PBO)
  - Autocorrelation variance inflation (Lo 2002)
"""

from __future__ import annotations

import math
import numpy as np
from scipy import stats

from robustness_lab.pbo import compute_pbo_cscv
from robustness_lab.statistics import (
    compute_autocorrelation_vif,
    compute_dsr,
    compute_expected_max_sharpe,
    compute_psr,
    estimate_effective_trials,
)


def test_psr_matches_analytical_mertens_normal_formula() -> None:
    # Construct near-Gaussian zero-skew sample and compare IID PSR to exact closed form
    rng = np.random.default_rng(101)
    r = rng.normal(loc=0.001, scale=0.01, size=500)
    psr, sr_p, skew, ex_kurt, vif = compute_psr(r, benchmark_sr_annual=0.0, use_autocorr_adjustment=False)
    assert vif == 1.0

    pearson_k = ex_kurt + 3.0
    se = math.sqrt((1.0 - skew * sr_p + ((pearson_k - 1.0) / 4.0) * sr_p ** 2) / (500 - 1))
    expected_psr = float(stats.norm.cdf(sr_p / se))
    assert abs(psr - expected_psr) < 1e-12


def test_autocorrelation_vif_inflates_for_ar1_process() -> None:
    rng = np.random.default_rng(202)
    n = 600
    eps = rng.normal(0.0, 0.01, size=n)
    ar1 = np.zeros(n)
    phi = 0.55
    for t in range(1, n):
        ar1[t] = phi * ar1[t - 1] + eps[t]

    vif_ar1 = compute_autocorrelation_vif(ar1, max_lags=5)
    vif_iid = compute_autocorrelation_vif(eps, max_lags=5)
    assert vif_ar1 > 1.8
    assert 0.7 < vif_iid < 1.3


def test_null_strategies_yield_low_dsr_and_high_pbo() -> None:
    rng = np.random.default_rng(303)
    t_obs, n_trials = 400, 24
    null_mat = rng.normal(loc=0.0, scale=0.01, size=(t_obs, n_trials))
    # Pick the luckiest in-sample null strategy (cherry-picking under null)
    srs = np.mean(null_mat, axis=0) / np.std(null_mat, axis=0, ddof=1)
    best_idx = int(np.argmax(srs))

    dsr_res = compute_dsr(
        candidate_trial_id=f"null_{best_idx}",
        candidate_returns=null_mat[:, best_idx],
        all_trial_returns=null_mat,
        annualization_factor=252,
    )
    pbo_res = compute_pbo_cscv(null_mat, n_partitions=8, annualization_factor=252)

    assert dsr_res.raw_trial_count == n_trials
    assert dsr_res.effective_trial_count > 16.0  # mostly independent
    assert dsr_res.expected_max_sharpe_annual > 0.5
    assert dsr_res.deflated_sharpe_ratio < 0.85  # fails strict 0.95 promotion bar
    assert pbo_res.pbo > 0.35  # overfitting probability is elevated under pure null


def test_correlated_trials_reduce_effective_trial_count_and_penalty() -> None:
    rng = np.random.default_rng(404)
    t_obs, n_trials = 400, 12
    # Independent trials
    indep_mat = rng.normal(loc=0.0, scale=0.01, size=(t_obs, n_trials))
    n_eff_indep, corr_indep = estimate_effective_trials(indep_mat)

    # Highly correlated trials (common factor + tiny idiosyncratic noise)
    common = rng.normal(loc=0.0, scale=0.01, size=(t_obs, 1))
    corr_mat = common + rng.normal(loc=0.0, scale=0.001, size=(t_obs, n_trials))
    n_eff_corr, corr_high = estimate_effective_trials(corr_mat)

    assert n_eff_indep > 9.5
    assert abs(corr_indep) < 0.15
    assert n_eff_corr < 1.5
    assert corr_high > 0.95

    # Same trial variance -> fewer effective trials means smaller E[max SR] threshold
    sr_star_indep = compute_expected_max_sharpe(0.0025, n_eff_indep)
    sr_star_corr = compute_expected_max_sharpe(0.0025, n_eff_corr)
    assert sr_star_corr < sr_star_indep


def test_known_signal_achieves_high_dsr_and_low_pbo() -> None:
    rng = np.random.default_rng(505)
    t_obs, n_trials = 400, 10
    # 9 weak/zero-drift decoys + 1 genuine high-SR strategy
    mat = rng.normal(loc=0.0, scale=0.01, size=(t_obs, n_trials))
    mat[:, 0] = rng.normal(loc=0.0018, scale=0.008, size=t_obs)  # true annualized SR ~ 3.5

    dsr_res = compute_dsr(
        candidate_trial_id="known_alpha",
        candidate_returns=mat[:, 0],
        all_trial_returns=mat,
        annualization_factor=252,
    )
    pbo_res = compute_pbo_cscv(mat, n_partitions=8, annualization_factor=252)

    assert dsr_res.deflated_sharpe_ratio > 0.98
    assert dsr_res.psr_vs_benchmark > 0.99
    assert pbo_res.pbo < 0.10
    assert pbo_res.prob_oos_loss < 0.05
    assert pbo_res.exact_enumeration is True
    assert pbo_res.n_combinations_evaluated == 70
