"""Pure domain module for two-pass Fama-MacBeth cross-sectional regressions.

Isolated from file I/O. Computes:
  - Period-by-period cross-sectional OLS slopes lambda_t
  - Classical Fama-MacBeth (1973) standard errors
  - Shanken (1992) errors-in-variables (EIV) corrected standard errors
  - Newey-West HAC standard errors on the time series of lambda_t
  - Cross-sectional pricing errors, MAE, RMSE, and cross-sectional R^2
"""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from scipy import stats

from mfape.errors import InsufficientObservationsError, RankDeficiencyError
from mfape.schemas import (
    AlignedPanel,
    FactorModelSpec,
    FactorPremiumEstimate,
    FamaMacBethResult,
    PricingErrorMetrics,
    TimeSeriesModelResult,
)
from mfape.time_series import check_design_matrix_rank


def compute_pricing_error_metrics(
    actual_means: np.ndarray,
    predicted_means: np.ndarray,
    n_params: int,
) -> PricingErrorMetrics:
    """Compute cross-sectional pricing error statistics from realized vs predicted mean returns."""
    errors = actual_means - predicted_means
    n_assets = len(actual_means)
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    max_abs = float(np.max(np.abs(errors)))
    mean_err = float(np.mean(errors))

    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((actual_means - np.mean(actual_means)) ** 2))
    if ss_tot <= 1e-18:
        cs_r2 = 1.0 if ss_res <= 1e-18 else 0.0
    else:
        cs_r2 = 1.0 - (ss_res / ss_tot)

    # Adjusted R^2 requires N > p (where p includes intercept if present)
    k_predictors = max(1, n_params - 1)
    if n_assets > k_predictors + 1:
        adj_r2: float | None = 1.0 - (1.0 - cs_r2) * (n_assets - 1) / float(
            n_assets - k_predictors - 1
        )
    else:
        adj_r2 = None

    return PricingErrorMetrics(
        mae=mae,
        rmse=rmse,
        max_abs_error=max_abs,
        mean_error=mean_err,
        cross_sectional_r2=float(cs_r2),
        adj_cross_sectional_r2=float(adj_r2) if adj_r2 is not None else None,
    )


def _two_sided_t_pvalue(t_stat: float, df: int) -> float:
    if not np.isfinite(t_stat):
        return 0.0
    if df <= 0:
        return float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat))))
    return float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df)))


def _compute_hac_se(series: np.ndarray, maxlags: int) -> float:
    n_obs = len(series)
    if n_obs < 2:
        return 0.0
    if float(np.std(series, ddof=1)) <= 1e-15:
        return 0.0
    effective_lags = min(maxlags, max(0, n_obs - 2))
    ones = np.ones((n_obs, 1), dtype=np.float64)
    fit = sm.OLS(series, ones).fit(cov_type="HAC", cov_kwds={"maxlags": effective_lags})
    return float(fit.bse[0])


def estimate_fama_macbeth(
    panel: AlignedPanel,
    ts_result: TimeSeriesModelResult,
    spec: FactorModelSpec,
    annualization_factor: float = 12.0,
    fm_hac_maxlags: int = 3,
    min_cs_assets: int = 3,
    rank_tol: float = 1e-10,
    max_condition_number: float = 1e8,
) -> FamaMacBethResult:
    """Run second-pass Fama-MacBeth cross-sectional regressions using first-pass betas.

    For each period t = 1..T with sufficient active assets:
        R^e_{i,t} = gamma_{0,t} + beta_i' lambda_t + e_{i,t}
    """
    spec.validate()
    factors = spec.factors
    k_factors = len(factors)
    include_intercept = spec.include_cs_intercept
    param_names = (["gamma_0", *factors]) if include_intercept else list(factors)
    p_params = len(param_names)

    # Align first-pass betas with assets present in panel
    asset_to_ts = {ar.asset_id: ar for ar in ts_result.asset_results}
    eligible_indices: list[int] = []
    eligible_assets: list[str] = []
    beta_rows: list[list[float]] = []

    for j, asset_id in enumerate(panel.assets):
        if asset_id in asset_to_ts:
            ar = asset_to_ts[asset_id]
            eligible_indices.append(j)
            eligible_assets.append(asset_id)
            beta_rows.append([ar.betas[f] for f in factors])

    n_assets = len(eligible_assets)
    min_needed = max(min_cs_assets, p_params + 1)
    if n_assets < min_needed:
        raise InsufficientObservationsError(
            f"Fama-MacBeth({spec.name}): need at least {min_needed} assets for {p_params} "
            f"cross-sectional parameters, found {n_assets}.",
            {"model": spec.name, "min_needed": min_needed, "n_assets": n_assets},
        )

    beta_matrix = np.array(beta_rows, dtype=np.float64)  # (N, K)
    if include_intercept:
        full_x = np.column_stack([np.ones(n_assets, dtype=np.float64), beta_matrix])
    else:
        full_x = beta_matrix

    # Verify full-cross-section beta design matrix is full rank and well-conditioned
    beta_cond = check_design_matrix_rank(
        full_x,
        column_names=param_names,
        context=f"FamaMacBeth({spec.name}) beta design matrix",
        rank_tol=rank_tol,
        max_condition_number=max_condition_number,
    )

    idx_arr = np.array(eligible_indices, dtype=np.int64)
    sub_excess = panel.excess_returns[:, idx_arr]  # (T, N)
    sub_mask = panel.active_mask[:, idx_arr]       # (T, N)
    n_periods_total = len(panel.dates)

    lambda_rows: list[np.ndarray] = []
    used_dates: list[str] = []
    period_r2s: list[float] = []

    for t in range(n_periods_total):
        active_t = sub_mask[t, :]
        n_active_t = int(np.sum(active_t))
        if n_active_t < min_needed:
            continue

        y_t = sub_excess[t, active_t]
        x_t = full_x[active_t, :]

        # Verify rank on unbalanced period cross-section
        if n_active_t < n_assets:
            check_design_matrix_rank(
                x_t,
                column_names=param_names,
                context=f"FamaMacBeth({spec.name}) cross-section at {panel.dates[t]}",
                rank_tol=rank_tol,
                max_condition_number=max_condition_number,
            )

        # Cross-sectional OLS via least squares
        lam_t, _, _, _ = np.linalg.lstsq(x_t, y_t, rcond=None)
        resid_t = y_t - x_t @ lam_t
        ss_res_t = float(np.sum(resid_t ** 2))
        ss_tot_t = float(np.sum((y_t - np.mean(y_t)) ** 2))
        if ss_tot_t > 1e-18:
            period_r2s.append(1.0 - ss_res_t / ss_tot_t)
        else:
            period_r2s.append(1.0 if ss_res_t <= 1e-18 else 0.0)

        lambda_rows.append(lam_t)
        used_dates.append(panel.dates[t])

    t_cs = len(lambda_rows)
    if t_cs < 2:
        raise InsufficientObservationsError(
            f"Fama-MacBeth({spec.name}): at least 2 valid cross-sectional periods are required, got {t_cs}.",
            {"model": spec.name, "valid_periods": t_cs},
        )

    lambda_mat = np.vstack(lambda_rows)  # (T_cs, p)
    lambda_hat = np.mean(lambda_mat, axis=0)  # (p,)

    # 1. Classical Fama-MacBeth standard errors
    if t_cs > 1:
        fm_cov = np.cov(lambda_mat, rowvar=False, ddof=1) / float(t_cs)
        fm_cov = np.atleast_2d(fm_cov)
        fm_ses = np.sqrt(np.maximum(0.0, np.diag(fm_cov)))
    else:
        fm_ses = np.zeros(p_params, dtype=np.float64)

    # 2. Shanken (1992) EIV correction
    # c = lambda_f' * Sigma_f^{-1} * lambda_f
    factor_offset = 1 if include_intercept else 0
    lambda_f = lambda_hat[factor_offset:]  # (K,)
    name_to_fidx = {name: i for i, name in enumerate(panel.factor_names)}
    f_indices = [name_to_fidx[f] for f in factors]
    date_to_tidx = {d: i for i, d in enumerate(panel.dates)}
    used_t_indices = np.array([date_to_tidx[d] for d in used_dates], dtype=np.int64)
    factor_used = panel.factor_returns[used_t_indices, :][:, f_indices]  # (T_cs, K)

    shanken_c: float | None = None
    shanken_ses: np.ndarray | None = None

    if t_cs > k_factors + 1:
        sigma_f = np.atleast_2d(np.cov(factor_used, rowvar=False, ddof=1))
        try:
            sigma_f_inv = np.linalg.inv(sigma_f)
            c_val = float(lambda_f.T @ sigma_f_inv @ lambda_f)
            if np.isfinite(c_val) and c_val >= 0.0:
                shanken_c = c_val
                # Exact Shanken (1992) covariance if balanced panel with full-sample TS residuals:
                # V_Shanken = (1/T) * [ (1 + c) * (A Sigma_eps A') + Sigma_f_pad ]
                # where A = (X'X)^{-1} X'
                is_balanced_cs = bool(sub_mask[used_t_indices, :].all()) and all(
                    asset_to_ts[a].n_obs == n_periods_total for a in eligible_assets
                )
                if is_balanced_cs and n_periods_total > k_factors + 1:
                    resid_cols = [
                        np.array(asset_to_ts[a].residuals, dtype=np.float64)[used_t_indices]
                        for a in eligible_assets
                    ]
                    resid_mat = np.column_stack(resid_cols)  # (T_cs, N)
                    sigma_eps = np.einsum("ti,tj->ij", resid_mat, resid_mat) / float(
                        t_cs - k_factors - 1
                    )
                    xtx = np.einsum("ni,nj->ij", full_x, full_x)
                    a_mat = np.linalg.inv(xtx) @ full_x.T  # (p, N)
                    sigma_f_pad = np.zeros((p_params, p_params), dtype=np.float64)
                    sigma_f_pad[factor_offset:, factor_offset:] = sigma_f
                    a_sigma_at = np.einsum("pi,ij,qj->pq", a_mat, sigma_eps, a_mat)
                    v_shanken = ((1.0 + shanken_c) * a_sigma_at + sigma_f_pad) / float(t_cs)
                    shanken_ses = np.sqrt(np.maximum(0.0, np.diag(v_shanken)))
                else:
                    # Conservative Shanken scaling for unbalanced PIT panels: SE * sqrt(1 + c)
                    shanken_ses = fm_ses * np.sqrt(1.0 + shanken_c)
        except np.linalg.LinAlgError:
            shanken_c = None
            shanken_ses = None

    # 3. Newey-West HAC standard errors on lambda_t
    hac_ses = np.array(
        [_compute_hac_se(lambda_mat[:, p], maxlags=fm_hac_maxlags) for p in range(p_params)],
        dtype=np.float64,
    )

    df_t = max(1, t_cs - 1)
    premia_list: list[FactorPremiumEstimate] = []
    for p, pname in enumerate(param_names):
        est = float(lambda_hat[p])
        fm_se = float(fm_ses[p])
        fm_t = (est / fm_se) if fm_se > 1e-15 else (0.0 if abs(est) <= 1e-15 else float("inf") * np.sign(est))
        fm_p = _two_sided_t_pvalue(fm_t, df_t)

        if shanken_ses is not None:
            sh_se: float | None = float(shanken_ses[p])
            sh_t: float | None = (
                (est / sh_se)
                if sh_se > 1e-15
                else (0.0 if abs(est) <= 1e-15 else float("inf") * np.sign(est))
            )
            sh_p: float | None = _two_sided_t_pvalue(sh_t, df_t)
        else:
            sh_se = None
            sh_t = None
            sh_p = None

        hac_se = float(hac_ses[p])
        hac_t = (est / hac_se) if hac_se > 1e-15 else (0.0 if abs(est) <= 1e-15 else float("inf") * np.sign(est))
        hac_p = _two_sided_t_pvalue(hac_t, df_t)

        premia_list.append(
            FactorPremiumEstimate(
                name=pname,
                estimate=est,
                annualized_estimate=float(est * annualization_factor),
                fm_se=fm_se,
                fm_tstat=float(fm_t) if np.isfinite(fm_t) else 0.0,
                fm_pvalue=fm_p,
                shanken_se=sh_se,
                shanken_tstat=float(sh_t) if (sh_t is not None and np.isfinite(sh_t)) else (0.0 if sh_t is not None else None),
                shanken_pvalue=sh_p,
                hac_se=hac_se,
                hac_tstat=float(hac_t) if np.isfinite(hac_t) else 0.0,
                hac_pvalue=hac_p,
            )
        )

    # Compute cross-sectional pricing errors across evaluated assets
    actual_means = np.array(
        [float(np.nanmean(sub_excess[used_t_indices, j])) for j in range(n_assets)],
        dtype=np.float64,
    )
    predicted_means = full_x @ lambda_hat
    pricing_errors = actual_means - predicted_means
    pricing_metrics = compute_pricing_error_metrics(
        actual_means=actual_means,
        predicted_means=predicted_means,
        n_params=p_params,
    )

    pricing_errors_by_asset = {
        eligible_assets[j]: float(pricing_errors[j]) for j in range(n_assets)
    }
    mean_actual_by_asset = {
        eligible_assets[j]: float(actual_means[j]) for j in range(n_assets)
    }
    mean_pred_by_asset = {
        eligible_assets[j]: float(predicted_means[j]) for j in range(n_assets)
    }
    lambda_series = {
        pname: tuple(float(v) for v in lambda_mat[:, p])
        for p, pname in enumerate(param_names)
    }

    return FamaMacBethResult(
        model_name=spec.name,
        factors=factors,
        include_intercept=include_intercept,
        n_periods=t_cs,
        n_assets=n_assets,
        date_start=used_dates[0],
        date_end=used_dates[-1],
        beta_condition_number=beta_cond,
        shanken_correction_factor=shanken_c,
        hac_maxlags=fm_hac_maxlags,
        premia=tuple(premia_list),
        pricing_errors_by_asset=pricing_errors_by_asset,
        mean_actual_excess_by_asset=mean_actual_by_asset,
        mean_predicted_excess_by_asset=mean_pred_by_asset,
        pricing_metrics=pricing_metrics,
        avg_period_cs_r2=float(np.mean(period_r2s)),
        lambda_series_dates=tuple(used_dates),
        lambda_series=lambda_series,
    )
