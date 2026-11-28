"""Pure domain module for time-series CAPM and multi-factor regressions.

Isolated from file I/O. Uses NumPy, SciPy, and statsmodels OLS with OLS, HC3,
or Newey-West HAC standard errors and computes the Gibbons-Ross-Shanken (GRS)
joint zero-alpha test statistic.
"""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from scipy import stats

from mfape.errors import (
    InsufficientObservationsError,
    RankDeficiencyError,
    UnsupportedInputError,
)
from mfape.schemas import (
    AlignedPanel,
    AssetTimeSeriesResult,
    FactorModelSpec,
    TimeSeriesModelResult,
)


def check_design_matrix_rank(
    matrix: np.ndarray,
    column_names: list[str],
    context: str,
    rank_tol: float = 1e-10,
    max_condition_number: float = 1e8,
) -> float:
    """Validate full column rank and numerical conditioning via SVD.

    Returns the 2-norm condition number if full rank and well-conditioned;
    otherwise raises RankDeficiencyError.
    """
    if matrix.ndim != 2:
        raise RankDeficiencyError(f"{context}: design matrix must be 2-D, got {matrix.ndim}-D.")
    n_rows, n_cols = matrix.shape
    if n_rows < n_cols:
        raise RankDeficiencyError(
            f"{context}: fewer rows ({n_rows}) than columns ({n_cols}).",
            {"context": context, "n_rows": n_rows, "n_cols": n_cols, "columns": column_names},
        )
    if not np.all(np.isfinite(matrix)):
        raise RankDeficiencyError(
            f"{context}: design matrix contains non-finite values.",
            {"context": context, "columns": column_names},
        )

    singular_values = np.linalg.svd(matrix, compute_uv=False)
    s_max = float(singular_values[0]) if singular_values.size > 0 else 0.0
    s_min = float(singular_values[-1]) if singular_values.size > 0 else 0.0

    # Relative tolerance consistent with np.linalg.matrix_rank
    threshold = max(n_rows, n_cols) * rank_tol * max(s_max, 1.0)
    rank = int(np.sum(singular_values > threshold))
    cond = float("inf") if s_min <= 0.0 else s_max / s_min

    if rank < n_cols:
        raise RankDeficiencyError(
            f"{context}: rank-deficient design matrix (rank={rank} < cols={n_cols}, "
            f"min_singular_value={s_min:.3e}, columns={column_names}).",
            {
                "context": context,
                "rank": rank,
                "expected_rank": n_cols,
                "columns": column_names,
                "singular_values": [float(s) for s in singular_values],
            },
        )
    if cond > max_condition_number:
        raise RankDeficiencyError(
            f"{context}: ill-conditioned design matrix (condition_number={cond:.3e} > "
            f"max={max_condition_number:.3e}, columns={column_names}).",
            {
                "context": context,
                "condition_number": cond,
                "max_condition_number": max_condition_number,
                "columns": column_names,
            },
        )
    return cond


def _extract_factor_slice(panel: AlignedPanel, factors: tuple[str, ...]) -> np.ndarray:
    name_to_idx = {name: i for i, name in enumerate(panel.factor_names)}
    missing = [f for f in factors if f not in name_to_idx]
    if missing:
        raise UnsupportedInputError(
            f"Requested factor(s) {missing} not present in aligned panel factors {list(panel.factor_names)}.",
            {"missing_factors": missing, "available_factors": list(panel.factor_names)},
        )
    indices = [name_to_idx[f] for f in factors]
    return panel.factor_returns[:, indices]


def _fit_ols_with_cov(
    y: np.ndarray,
    x: np.ndarray,
    cov_type: str,
    hac_maxlags: int,
):
    model = sm.OLS(y, x)
    if cov_type == "OLS":
        return model.fit()
    if cov_type == "HC3":
        return model.fit(cov_type="HC3")
    if cov_type == "HAC":
        # Cap lag length cleanly below sample size - 1
        effective_lags = min(hac_maxlags, max(0, len(y) - 2))
        return model.fit(cov_type="HAC", cov_kwds={"maxlags": effective_lags})
    raise UnsupportedInputError(f"Unsupported covariance estimator '{cov_type}'.")


def compute_grs_test(
    alphas: np.ndarray,
    residuals: np.ndarray,
    factor_matrix: np.ndarray,
) -> tuple[float | None, float | None, int | None, int | None, str | None]:
    """Compute the Gibbons-Ross-Shanken (1989) F-test for H0: all alphas == 0.

    Parameters
    ----------
    alphas: shape (N,)
    residuals: shape (T, N), balanced residual matrix
    factor_matrix: shape (T, K)

    Returns
    -------
    (grs_f_stat, grs_pvalue, df1, df2, note)
    """
    t_obs, n_assets = residuals.shape
    k_factors = factor_matrix.shape[1]
    df1 = n_assets
    df2 = t_obs - n_assets - k_factors

    if df2 <= 0:
        return (
            None,
            None,
            df1,
            df2,
            f"GRS test undefined: requires T - N - K > 0 (got T={t_obs}, N={n_assets}, K={k_factors}).",
        )

    # Unbiased residual covariance matrix Sigma_hat = (E'E) / (T - K - 1)
    denom_res = t_obs - k_factors - 1
    if denom_res <= 0:
        return (None, None, df1, df2, "GRS test undefined: T - K - 1 <= 0.")

    f_bar = np.mean(factor_matrix, axis=0)  # (K,)
    f_centered = factor_matrix - f_bar

    # Standard GRS (1989) F-statistic using MLE (1/T) covariance matrices:
    # GRS = ((T - N - K) / N) * (1 / (1 + f_bar' Omega_mle^{-1} f_bar)) * (alpha' Sigma_mle^{-1} alpha)
    sigma_mle = np.einsum("ti,tj->ij", residuals, residuals) / float(t_obs)
    omega_mle = np.einsum("ti,tj->ij", f_centered, f_centered) / float(t_obs)

    # Guard against exact zero-noise fixtures where Sigma_mle is near-zero / singular
    if np.linalg.cond(sigma_mle) > 1e12 or float(np.max(np.abs(sigma_mle))) <= 1e-18:
        return (
            None,
            None,
            df1,
            df2,
            "GRS test unavailable: singular or near-zero residual covariance matrix (possible zero-noise fixture).",
        )

    try:
        sigma_mle_inv = np.linalg.inv(sigma_mle)
        omega_mle_inv = np.linalg.inv(omega_mle)
    except np.linalg.LinAlgError:
        return (None, None, df1, df2, "GRS test unavailable: singular covariance matrix.")

    sharpe_sq = float(np.einsum("i,ij,j->", f_bar, omega_mle_inv, f_bar))
    quad_alpha = float(np.einsum("i,ij,j->", alphas, sigma_mle_inv, alphas))
    grs_f = float((df2 / float(n_assets)) * (quad_alpha / (1.0 + sharpe_sq)))
    if not np.isfinite(grs_f) or grs_f < 0.0:
        return (None, None, df1, df2, "GRS test produced non-finite value.")
    p_val = float(1.0 - stats.f.cdf(grs_f, df1, df2))
    return (grs_f, p_val, df1, df2, None)


def fit_time_series_model(
    panel: AlignedPanel,
    spec: FactorModelSpec,
    cov_type: str = "HAC",
    hac_maxlags: int = 3,
    min_ts_observations: int = 12,
    rank_tol: float = 1e-10,
    max_condition_number: float = 1e8,
) -> TimeSeriesModelResult:
    """Fit time-series CAPM or multi-factor OLS regressions for every eligible asset.

    Model per asset i:
        R^e_{i,t} = alpha_i + sum_k beta_{i,k} * F_{k,t} + eps_{i,t}
    """
    spec.validate()
    factor_mat = _extract_factor_slice(panel, spec.factors)
    n_periods, k_factors = factor_mat.shape

    # Check global factor design matrix rank/conditioning
    if spec.include_ts_intercept:
        global_x = np.column_stack([np.ones(n_periods, dtype=np.float64), factor_mat])
        global_cols = ["intercept", *spec.factors]
    else:
        global_x = factor_mat
        global_cols = list(spec.factors)

    factor_cond = check_design_matrix_rank(
        global_x,
        column_names=global_cols,
        context=f"TimeSeries({spec.name}) global factor matrix",
        rank_tol=rank_tol,
        max_condition_number=max_condition_number,
    )

    p_params = global_x.shape[1]
    min_required = max(min_ts_observations, p_params + 1)

    asset_results: list[AssetTimeSeriesResult] = []
    balanced_residuals_list: list[np.ndarray] = []
    balanced_alphas_list: list[float] = []
    all_assets_full_sample = True

    for j, asset_id in enumerate(panel.assets):
        mask = panel.active_mask[:, j]
        n_obs = int(np.sum(mask))
        if n_obs < min_required:
            all_assets_full_sample = False
            continue
        if n_obs < n_periods:
            all_assets_full_sample = False

        y_i = panel.excess_returns[mask, j]
        f_i = factor_mat[mask, :]
        dates_i = tuple(panel.dates[t] for t in range(n_periods) if mask[t])

        if spec.include_ts_intercept:
            x_i = np.column_stack([np.ones(n_obs, dtype=np.float64), f_i])
        else:
            x_i = f_i

        # Check asset-specific active subsample rank when panel is unbalanced
        if n_obs < n_periods:
            check_design_matrix_rank(
                x_i,
                column_names=global_cols,
                context=f"TimeSeries({spec.name}) asset '{asset_id}'",
                rank_tol=rank_tol,
                max_condition_number=max_condition_number,
            )

        res = _fit_ols_with_cov(y_i, x_i, cov_type=cov_type, hac_maxlags=hac_maxlags)
        params = np.asarray(res.params, dtype=np.float64)
        bse = np.asarray(res.bse, dtype=np.float64)
        tvalues = np.asarray(res.tvalues, dtype=np.float64)
        pvalues = np.asarray(res.pvalues, dtype=np.float64)
        resid = np.asarray(res.resid, dtype=np.float64)

        offset = 1 if spec.include_ts_intercept else 0
        if spec.include_ts_intercept:
            alpha = float(params[0])
            alpha_se = float(bse[0])
            alpha_t = float(tvalues[0]) if np.isfinite(tvalues[0]) else 0.0
            alpha_p = float(pvalues[0]) if np.isfinite(pvalues[0]) else 1.0
        else:
            alpha = None
            alpha_se = None
            alpha_t = None
            alpha_p = None

        betas = {f: float(params[offset + k]) for k, f in enumerate(spec.factors)}
        beta_ses = {f: float(bse[offset + k]) for k, f in enumerate(spec.factors)}
        beta_tstats = {
            f: float(tvalues[offset + k]) if np.isfinite(tvalues[offset + k]) else 0.0
            for k, f in enumerate(spec.factors)
        }
        beta_pvalues = {
            f: float(pvalues[offset + k]) if np.isfinite(pvalues[offset + k]) else 1.0
            for k, f in enumerate(spec.factors)
        }

        mean_exc = float(np.mean(y_i))
        mean_f = np.mean(f_i, axis=0)
        beta_vec = np.array([betas[f] for f in spec.factors], dtype=np.float64)
        # Factor-implied expected excess return under zero-alpha pricing restriction: beta_i' * E[F]
        predicted_mean = float(beta_vec @ mean_f)
        pricing_err = mean_exc - predicted_mean

        dof = max(1, n_obs - p_params)
        resid_std = float(np.sqrt(np.sum(resid ** 2) / float(dof)))

        asset_results.append(
            AssetTimeSeriesResult(
                asset_id=asset_id,
                n_obs=n_obs,
                alpha=alpha,
                alpha_se=alpha_se,
                alpha_tstat=alpha_t,
                alpha_pvalue=alpha_p,
                betas=betas,
                beta_ses=beta_ses,
                beta_tstats=beta_tstats,
                beta_pvalues=beta_pvalues,
                r_squared=float(res.rsquared),
                adj_r_squared=float(res.rsquared_adj),
                residual_std=resid_std,
                mean_excess_return=mean_exc,
                predicted_mean_excess_return=predicted_mean,
                pricing_error=pricing_err,
                residuals=tuple(float(r) for r in resid),
                dates=dates_i,
            )
        )

        if n_obs == n_periods and alpha is not None:
            balanced_residuals_list.append(resid)
            balanced_alphas_list.append(alpha)

    if not asset_results:
        raise InsufficientObservationsError(
            f"No assets met minimum time-series observations ({min_required}) for model '{spec.name}'.",
            {"model": spec.name, "min_required": min_required, "panel_periods": n_periods},
        )

    r2_vals = np.array([ar.r_squared for ar in asset_results], dtype=np.float64)
    adj_r2_vals = np.array([ar.adj_r_squared for ar in asset_results], dtype=np.float64)
    pricing_errs = np.array(
        [ar.alpha if ar.alpha is not None else ar.pricing_error for ar in asset_results],
        dtype=np.float64,
    )

    grs_f: float | None = None
    grs_p: float | None = None
    grs_df1: int | None = None
    grs_df2: int | None = None
    grs_note: str | None = None

    if not spec.include_ts_intercept:
        grs_note = "GRS test not applicable when time-series intercept is omitted."
    elif not all_assets_full_sample or len(balanced_residuals_list) != len(asset_results):
        grs_note = "GRS test requires a balanced panel across all evaluated assets; skipped for unbalanced PIT panel."
    else:
        resid_mat = np.column_stack(balanced_residuals_list)  # (T, N)
        alpha_vec = np.array(balanced_alphas_list, dtype=np.float64)
        grs_f, grs_p, grs_df1, grs_df2, grs_note = compute_grs_test(
            alphas=alpha_vec,
            residuals=resid_mat,
            factor_matrix=factor_mat,
        )

    return TimeSeriesModelResult(
        model_name=spec.name,
        factors=spec.factors,
        cov_type=cov_type,
        hac_maxlags=hac_maxlags if cov_type == "HAC" else 0,
        include_intercept=spec.include_ts_intercept,
        n_periods=n_periods,
        n_assets=len(asset_results),
        date_start=panel.dates[0],
        date_end=panel.dates[-1],
        factor_condition_number=factor_cond,
        asset_results=tuple(asset_results),
        mean_r_squared=float(np.mean(r2_vals)),
        mean_adj_r_squared=float(np.mean(adj_r2_vals)),
        mean_abs_alpha=float(np.mean(np.abs(pricing_errs))),
        rmse_alpha=float(np.sqrt(np.mean(pricing_errs ** 2))),
        grs_f_stat=grs_f,
        grs_pvalue=grs_p,
        grs_df1=grs_df1,
        grs_df2=grs_df2,
        grs_note=grs_note,
    )
