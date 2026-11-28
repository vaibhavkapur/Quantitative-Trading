"""Core multiple-testing statistical diagnostics: Sharpe, PSR, DSR, and outlier analysis.

Pure domain module (no file I/O). Implements:
  - Sample Sharpe ratio (per-period and annualized)
  - Lo (2002) / Newey-West Bartlett autocorrelation variance inflation factor
  - Probabilistic Sharpe Ratio (Bailey & Lopez de Prado, 2012)
  - Effective number of independent trials via correlation matrix eigenvalue participation ratio
  - Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
  - Extreme-outlier & tail-impact diagnostics
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy import stats

from .models import (
    ContractValidationError,
    DSRResult,
    InsufficientSampleError,
    OutlierDiagnostic,
)

EULER_MASCHERONI = 0.5772156649015329


def _validate_return_vector(returns: np.ndarray, min_len: int = 2, label: str = "returns") -> np.ndarray:
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim != 1:
        raise ContractValidationError(f"{label} must be 1-dimensional, got shape {arr.shape}")
    if arr.size < min_len:
        raise InsufficientSampleError(f"{label} has {arr.size} observations; at least {min_len} required")
    if not np.all(np.isfinite(arr)):
        raise ContractValidationError(f"{label} contains NaN or infinite values")
    return arr


def compute_sharpe_ratio(
    returns: np.ndarray,
    annualization_factor: int = 252,
    ddof: int = 1,
) -> tuple[float, float]:
    """Return (per_period_sharpe, annualized_sharpe)."""
    arr = _validate_return_vector(returns, min_len=max(2, ddof + 1), label="returns")
    if annualization_factor <= 0:
        raise ContractValidationError(f"annualization_factor must be > 0, got {annualization_factor}")
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=ddof))
    if std <= 1e-15:
        if abs(mean) <= 1e-15:
            return 0.0, 0.0
        raise ContractValidationError(
            "Constant non-zero return series has zero volatility and undefined Sharpe ratio"
        )
    sr_period = mean / std
    sr_annual = sr_period * math.sqrt(float(annualization_factor))
    return sr_period, sr_annual


def compute_sample_moments(returns: np.ndarray) -> tuple[float, float, float, float]:
    """Return (mean, sample_std, skewness, excess_kurtosis)."""
    arr = _validate_return_vector(returns, min_len=4, label="returns")
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    if std <= 1e-15:
        return mean, std, 0.0, 0.0
    skew = float(stats.skew(arr, bias=False))
    ex_kurt = float(stats.kurtosis(arr, fisher=True, bias=False))
    return mean, std, skew, ex_kurt


def compute_autocorrelation_vif(returns: np.ndarray, max_lags: int = 5) -> float:
    """Estimate variance inflation factor (VIF) from sample autocorrelation.

    Uses Newey-West Bartlett weights:
        VIF = 1 + 2 * sum_{k=1..q} (1 - k/(q+1)) * rho_k
    Floored at 0.25 to avoid degenerate negative variance estimates in small samples.
    """
    arr = _validate_return_vector(returns, min_len=4, label="returns")
    if max_lags <= 0:
        return 1.0
    centered = arr - np.mean(arr)
    var = float(np.dot(centered, centered))
    if var <= 1e-24:
        return 1.0

    q = min(max_lags, arr.size // 4)
    if q < 1:
        return 1.0

    vif = 1.0
    for k in range(1, q + 1):
        rho_k = float(np.dot(centered[k:], centered[:-k])) / var
        weight = 1.0 - (k / float(q + 1))
        vif += 2.0 * weight * rho_k

    return max(0.25, vif)


def compute_psr(
    returns: np.ndarray,
    benchmark_sr_annual: float = 0.0,
    annualization_factor: int = 252,
    use_autocorr_adjustment: bool = True,
    max_autocorr_lags: int = 5,
) -> tuple[float, float, float, float, float]:
    """Compute Probabilistic Sharpe Ratio (PSR) against an annualized benchmark SR.

    Uses Bailey & Lopez de Prado (2012) / Mertens (2002) standard error:
      se(SR_hat) = sqrt(VIF * (1 - gamma_3 * SR_hat + ((gamma_4 - 1) / 4) * SR_hat^2) / (T - 1))
    where gamma_4 is Pearson (non-excess) kurtosis = excess_kurtosis + 3.

    Returns:
      (psr, sr_period, skewness, excess_kurtosis, autocorr_vif)
    """
    arr = _validate_return_vector(returns, min_len=4, label="returns")
    t_obs = arr.size
    sr_period, _ = compute_sharpe_ratio(arr, annualization_factor=annualization_factor)
    _, _, skew, ex_kurt = compute_sample_moments(arr)
    vif = compute_autocorrelation_vif(arr, max_lags=max_autocorr_lags) if use_autocorr_adjustment else 1.0

    sr0_period = benchmark_sr_annual / math.sqrt(float(annualization_factor))
    pearson_kurt = ex_kurt + 3.0
    inner = 1.0 - skew * sr_period + ((pearson_kurt - 1.0) / 4.0) * (sr_period ** 2)
    # Guard numerical floor in finite samples with bias-corrected moments
    inner = max(inner, 1e-12)
    se_sr = math.sqrt((vif * inner) / float(t_obs - 1))

    z_stat = (sr_period - sr0_period) / se_sr
    psr = float(stats.norm.cdf(z_stat))
    return psr, sr_period, skew, ex_kurt, vif


def estimate_effective_trials(trial_returns_matrix: np.ndarray) -> tuple[float, float]:
    """Estimate effective number of independent trials N_eff from T x N return matrix.

    Uses the eigenvalue participation ratio of the N x N correlation matrix C:
        N_eff = (sum_i lambda_i)^2 / sum_i lambda_i^2 = N^2 / tr(C^2)

    Properties:
      - Uncorrelated trials (C = I_N) -> N_eff = N
      - Identical trials (rho = 1) -> N_eff = 1.0
      - K independent clusters of identical trials -> N_eff = K

    Returns (effective_trials, mean_off_diagonal_corr).
    """
    mat = np.asarray(trial_returns_matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ContractValidationError(f"trial_returns_matrix must be 2D (T x N), got shape {mat.shape}")
    t_obs, n_trials = mat.shape
    if n_trials < 1:
        raise InsufficientSampleError("At least 1 trial is required")
    if t_obs < 4:
        raise InsufficientSampleError(f"At least 4 observations required, got {t_obs}")
    if not np.all(np.isfinite(mat)):
        raise ContractValidationError("trial_returns_matrix contains NaN or infinite values")

    if n_trials == 1:
        return 1.0, 0.0

    stds = np.std(mat, axis=0, ddof=1)
    # Handle zero-variance columns cleanly by treating them as uncorrelated degenerate columns
    valid_mask = stds > 1e-15
    if np.sum(valid_mask) <= 1:
        return float(n_trials), 0.0

    corr = np.eye(n_trials, dtype=np.float64)
    valid_idx = np.where(valid_mask)[0]
    sub_corr = np.corrcoef(mat[:, valid_idx], rowvar=False)
    sub_corr = np.nan_to_num(sub_corr, nan=0.0)
    sub_corr = np.clip(sub_corr, -1.0, 1.0)
    np.fill_diagonal(sub_corr, 1.0)
    corr[np.ix_(valid_idx, valid_idx)] = sub_corr

    # Participation ratio of positive eigenvalues
    eigvals = np.linalg.eigvalsh(corr)
    eigvals = np.maximum(eigvals, 0.0)
    denom = float(np.sum(eigvals ** 2))
    n_eff = (float(np.sum(eigvals)) ** 2) / denom if denom > 1e-15 else float(n_trials)
    n_eff = float(np.clip(n_eff, 1.0, float(n_trials)))

    off_diag_mask = ~np.eye(n_trials, dtype=bool)
    mean_off_diag = float(np.mean(corr[off_diag_mask]))
    return n_eff, mean_off_diag


def compute_expected_max_sharpe(
    trial_sharpe_variance_period: float,
    effective_trials: float,
) -> float:
    """Compute E[max SR] threshold SR* under null hypothesis (Bailey & Lopez de Prado 2014).

    SR* = sqrt(V[{SR_n}]) * ((1 - gamma) * Phi^-1(1 - 1/N_eff) + gamma * Phi^-1(1 - 1/(N_eff * e)))
    Returns per-period SR*.
    """
    if trial_sharpe_variance_period < 0.0 or not math.isfinite(trial_sharpe_variance_period):
        raise ContractValidationError(
            f"trial_sharpe_variance_period must be finite and >= 0, got {trial_sharpe_variance_period!r}"
        )
    if effective_trials < 1.0 or not math.isfinite(effective_trials):
        raise ContractValidationError(f"effective_trials must be >= 1, got {effective_trials!r}")

    if effective_trials <= 1.0 + 1e-9 or trial_sharpe_variance_period <= 1e-18:
        return 0.0

    term1 = (1.0 - EULER_MASCHERONI) * float(stats.norm.ppf(1.0 - 1.0 / effective_trials))
    term2 = EULER_MASCHERONI * float(stats.norm.ppf(1.0 - 1.0 / (effective_trials * math.e)))
    sr_star = math.sqrt(trial_sharpe_variance_period) * (term1 + term2)
    return max(0.0, sr_star)


def compute_dsr(
    candidate_trial_id: str,
    candidate_returns: np.ndarray,
    all_trial_returns: np.ndarray,
    annualization_factor: int = 252,
    benchmark_sr_annual: float = 0.0,
    min_observations: int = 30,
    use_autocorr_adjustment: bool = True,
    max_autocorr_lags: int = 5,
    extra_unobserved_failed_trials: int = 0,
) -> DSRResult:
    """Compute Deflated Sharpe Ratio (DSR) and supporting diagnostics.

    Parameters:
      candidate_trial_id: Identifier of the selected strategy trial.
      candidate_returns: 1D array of net returns for the candidate strategy (length T).
      all_trial_returns: 2D array (T x N_completed) of net returns across all completed trials.
      extra_unobserved_failed_trials: Count of failed/aborted trials without return series,
        added conservatively to raw trial count and scaled into effective trials.
    """
    cand = _validate_return_vector(candidate_returns, min_len=min_observations, label="candidate_returns")
    mat = np.asarray(all_trial_returns, dtype=np.float64)
    if mat.ndim != 2:
        raise ContractValidationError(f"all_trial_returns must be 2D (T x N), got shape {mat.shape}")
    if mat.shape[0] != cand.size:
        raise ContractValidationError(
            f"Candidate length ({cand.size}) does not match trial matrix rows ({mat.shape[0]})"
        )
    if extra_unobserved_failed_trials < 0:
        raise ContractValidationError("extra_unobserved_failed_trials must be >= 0")

    t_obs, n_completed = mat.shape
    raw_trial_count = n_completed + extra_unobserved_failed_trials

    psr_bench, sr_period, skew, ex_kurt, vif = compute_psr(
        cand,
        benchmark_sr_annual=benchmark_sr_annual,
        annualization_factor=annualization_factor,
        use_autocorr_adjustment=use_autocorr_adjustment,
        max_autocorr_lags=max_autocorr_lags,
    )
    sr_annual = sr_period * math.sqrt(float(annualization_factor))

    # Per-period Sharpe ratio of every completed trial
    trial_srs: list[float] = []
    for col in range(n_completed):
        col_arr = mat[:, col]
        col_std = float(np.std(col_arr, ddof=1))
        col_mean = float(np.mean(col_arr))
        if col_std <= 1e-15:
            trial_srs.append(0.0)
        else:
            trial_srs.append(col_mean / col_std)

    n_eff_completed, mean_corr = estimate_effective_trials(mat)
    # Conservatively scale effective trial count if failed/aborted trials were also attempted
    eff_ratio = n_eff_completed / float(n_completed) if n_completed > 0 else 1.0
    n_eff_total = max(1.0, n_eff_completed + extra_unobserved_failed_trials * eff_ratio)

    if len(trial_srs) >= 2:
        trial_sr_var = float(np.var(trial_srs, ddof=1))
    else:
        trial_sr_var = 0.0

    # If multiple trials exist but cross-sectional variance is 0, fallback to theoretical null variance 1/(T-1)
    warnings: list[str] = []
    if n_eff_total > 1.0 and trial_sr_var <= 1e-12:
        trial_sr_var = 1.0 / float(t_obs - 1)
        warnings.append(
            "Cross-sectional Sharpe variance across trials was near zero; fell back to null variance 1/(T-1)."
        )

    sr_star_period = compute_expected_max_sharpe(trial_sr_var, n_eff_total)
    sr_star_annual = sr_star_period * math.sqrt(float(annualization_factor))

    # DSR is PSR evaluated against SR*
    dsr, _, _, _, _ = compute_psr(
        cand,
        benchmark_sr_annual=sr_star_annual,
        annualization_factor=annualization_factor,
        use_autocorr_adjustment=use_autocorr_adjustment,
        max_autocorr_lags=max_autocorr_lags,
    )

    if raw_trial_count < 2:
        warnings.append("Only 1 trial logged; multiple-testing deflation is inactive (SR* = 0).")
    if vif > 1.25:
        warnings.append(
            f"Positive return autocorrelation detected (VIF={vif:.2f}); Sharpe standard error was inflated."
        )
    if ex_kurt > 10.0:
        warnings.append(
            f"Heavy tails detected (excess kurtosis={ex_kurt:.2f}); asymptotic normal PSR/DSR approximation has higher uncertainty."
        )
    if mean_corr > 0.70 and n_completed > 1:
        warnings.append(
            f"High average pairwise trial correlation ({mean_corr:.2f}); effective trials reduced from {raw_trial_count} to {n_eff_total:.2f}."
        )

    assumptions = (
        "Stationary return distribution over the evaluation window.",
        "Sharpe ratio estimator standard error adjusts for sample skewness and Pearson kurtosis (Mertens 2002).",
        (
            f"Autocorrelation adjusted via Bartlett-weighted VIF up to {max_autocorr_lags} lags (Lo 2002)."
            if use_autocorr_adjustment
            else "Returns assumed serially uncorrelated (IID)."
        ),
        "Effective independent trial count estimated via correlation matrix eigenvalue participation ratio.",
        "Expected maximum Sharpe under null uses Bailey & Lopez de Prado (2014) extreme-value approximation; does not prove future alpha.",
    )

    return DSRResult(
        candidate_trial_id=candidate_trial_id,
        n_observations=t_obs,
        annualization_factor=annualization_factor,
        observed_sharpe_period=sr_period,
        observed_sharpe_annual=sr_annual,
        benchmark_sharpe_annual=benchmark_sr_annual,
        skewness=skew,
        excess_kurtosis=ex_kurt,
        autocorr_variance_inflation=vif,
        psr_vs_benchmark=psr_bench,
        raw_trial_count=raw_trial_count,
        effective_trial_count=n_eff_total,
        trial_sharpe_variance_period=trial_sr_var,
        expected_max_sharpe_period=sr_star_period,
        expected_max_sharpe_annual=sr_star_annual,
        deflated_sharpe_ratio=dsr,
        assumptions=assumptions,
        warnings=tuple(warnings),
    )


def diagnose_outliers(
    returns: np.ndarray,
    annualization_factor: int = 252,
    zscore_threshold: float = 8.0,
    kurtosis_threshold: float = 20.0,
    winsor_limits: tuple[float, float] = (0.01, 0.01),
) -> OutlierDiagnostic:
    """Assess whether candidate performance is driven by extreme outliers or fat tails."""
    arr = _validate_return_vector(returns, min_len=10, label="returns")
    _, raw_sr_annual = compute_sharpe_ratio(arr, annualization_factor=annualization_factor)
    _, std, skew, ex_kurt = compute_sample_moments(arr)

    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    robust_scale = 1.4826 * mad if mad > 1e-12 else std
    if robust_scale > 1e-15:
        zscores = np.abs(arr - median) / robust_scale
    else:
        zscores = np.zeros_like(arr)

    max_abs_ret = float(np.max(np.abs(arr)))
    max_abs_z = float(np.max(zscores))
    n_extreme = int(np.sum(zscores > zscore_threshold))

    lo_q = float(np.quantile(arr, winsor_limits[0]))
    hi_q = float(np.quantile(arr, 1.0 - winsor_limits[1]))
    winsorized = np.clip(arr, lo_q, hi_q)
    if float(np.std(winsorized, ddof=1)) > 1e-15:
        _, win_sr_annual = compute_sharpe_ratio(winsorized, annualization_factor=annualization_factor)
    else:
        win_sr_annual = 0.0

    denom = max(abs(raw_sr_annual), 1e-6)
    impact_pct = (raw_sr_annual - win_sr_annual) / denom

    warnings: list[str] = []
    if n_extreme > 0:
        warnings.append(
            f"Detected {n_extreme} extreme outlier observation(s) with robust |z| > {zscore_threshold:.1f} "
            f"(max |z|={max_abs_z:.2f}, max |r|={max_abs_ret:.4f})."
        )
    if ex_kurt > kurtosis_threshold:
        warnings.append(
            f"Excess kurtosis ({ex_kurt:.2f}) exceeds warning threshold ({kurtosis_threshold:.1f})."
        )
    if abs(impact_pct) > 0.35 and abs(raw_sr_annual) > 0.2:
        warnings.append(
            f"1% winsorization changes annualized Sharpe from {raw_sr_annual:.2f} to {win_sr_annual:.2f} "
            f"({impact_pct * 100:.1f}% sensitivity)."
        )

    return OutlierDiagnostic(
        max_abs_return=max_abs_ret,
        max_abs_zscore=max_abs_z,
        n_extreme_observations=n_extreme,
        excess_kurtosis=ex_kurt,
        skewness=skew,
        winsorized_sharpe_annual=win_sr_annual,
        raw_sharpe_annual=raw_sr_annual,
        outlier_impact_pct=impact_pct,
        warnings=tuple(warnings),
    )
