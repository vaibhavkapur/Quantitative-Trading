"""Multi-factor statistical return attribution with explicit non-causation disclosure."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from pae.config import AttributionConfig
from pae.domain.models import AttributionInputBatch, DataQualityException
from pae.engine.returns import ReturnSummary


@dataclass(frozen=True)
class FactorExposureResult:
    """Estimated exposure and Carino-linked return contribution for a single risk factor."""

    factor_name: str
    beta: float
    std_error: Optional[float]
    t_stat: Optional[float]
    linked_contribution: float
    mean_daily_factor_return: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FactorAttributionReport:
    """Multi-factor OLS regression attribution with mandatory non-causation disclaimer."""

    status: str  # "AVAILABLE", "INSUFFICIENT_DATA", "DISABLED"
    disclaimer: str
    sample_size: int
    factor_names: List[str]
    r_squared: Optional[float]
    adjusted_r_squared: Optional[float]
    daily_alpha: Optional[float]
    linked_risk_free_contribution: float
    linked_factor_total_contribution: float
    linked_specific_return: float
    explained_plus_specific_return: float
    portfolio_gross_return_sample: float
    residual_check: float
    factors: List[FactorExposureResult] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _solve_linear_system(a: List[List[float]], b: List[float]) -> List[float]:
    """Solve A x = b using Gaussian elimination with partial pivoting (pure Python)."""
    n = len(b)
    mat = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(mat[r][col]))
        if abs(mat[pivot_row][col]) < 1e-15:
            raise ValueError("Singular matrix in factor regression")
        if pivot_row != col:
            mat[col], mat[pivot_row] = mat[pivot_row], mat[col]
        pivot = mat[col][col]
        for j in range(col, n + 1):
            mat[col][j] /= pivot
        for r in range(n):
            if r != col:
                factor = mat[r][col]
                if factor != 0.0:
                    for j in range(col, n + 1):
                        mat[r][j] -= factor * mat[col][j]
    return [mat[i][n] for i in range(n)]


def _invert_matrix(a: List[List[float]]) -> Optional[List[List[float]]]:
    """Invert a small square matrix using Gauss-Jordan elimination."""
    n = len(a)
    aug = [a[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-15:
            return None
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pivot
        for r in range(n):
            if r != col:
                factor = aug[r][col]
                for j in range(2 * n):
                    aug[r][j] -= factor * aug[col][j]
    return [row[n:] for row in aug]


def compute_factor_attribution(
    batch: AttributionInputBatch,
    config: AttributionConfig,
    returns: ReturnSummary,
    exceptions: List[DataQualityException],
) -> FactorAttributionReport:
    """Estimate statistical risk-factor exposures and Carino-linked factor contributions."""
    if not config.factor_enabled:
        return FactorAttributionReport(
            status="DISABLED",
            disclaimer=config.non_causation_disclaimer,
            sample_size=0,
            factor_names=[],
            r_squared=None,
            adjusted_r_squared=None,
            daily_alpha=None,
            linked_risk_free_contribution=0.0,
            linked_factor_total_contribution=0.0,
            linked_specific_return=0.0,
            explained_plus_specific_return=0.0,
            portfolio_gross_return_sample=0.0,
            residual_check=0.0,
            notes=["Factor attribution disabled by configuration."],
        )

    fac_by_date = {f.trade_date: f for f in batch.factors}
    aligned = []
    for dret in returns.daily_returns:
        if dret.is_zero_nav_day:
            continue
        fobs = fac_by_date.get(dret.trade_date)
        if fobs is not None and fobs.factor_returns:
            aligned.append((dret, fobs))

    if len(aligned) < config.factor_min_observations:
        msg = (
            f"Factor attribution requires at least {config.factor_min_observations} aligned "
            f"observations; found {len(aligned)}."
        )
        if batch.factors:
            exceptions.append(
                DataQualityException(
                    code="INSUFFICIENT_FACTOR_OBSERVATIONS",
                    severity="INFO",
                    message=msg,
                    details={"aligned_observations": len(aligned), "min_required": config.factor_min_observations},
                )
            )
        return FactorAttributionReport(
            status="INSUFFICIENT_DATA",
            disclaimer=config.non_causation_disclaimer,
            sample_size=len(aligned),
            factor_names=[],
            r_squared=None,
            adjusted_r_squared=None,
            daily_alpha=None,
            linked_risk_free_contribution=0.0,
            linked_factor_total_contribution=0.0,
            linked_specific_return=0.0,
            explained_plus_specific_return=0.0,
            portfolio_gross_return_sample=0.0,
            residual_check=0.0,
            notes=[msg],
        )

    # Common factor names present across all aligned dates
    common_factors = sorted(
        set.intersection(*(set(fobs.factor_returns.keys()) for _, fobs in aligned))
    )
    if not common_factors:
        return FactorAttributionReport(
            status="INSUFFICIENT_DATA",
            disclaimer=config.non_causation_disclaimer,
            sample_size=len(aligned),
            factor_names=[],
            r_squared=None,
            adjusted_r_squared=None,
            daily_alpha=None,
            linked_risk_free_contribution=0.0,
            linked_factor_total_contribution=0.0,
            linked_specific_return=0.0,
            explained_plus_specific_return=0.0,
            portfolio_gross_return_sample=0.0,
            residual_check=0.0,
            notes=["No common factor series across aligned dates."],
        )

    n = len(aligned)
    k = len(common_factors)
    p = k + 1  # intercept + k factors

    # Design matrix X (n x p) and response y = r_gross - r_f
    x_rows: List[List[float]] = []
    y_vals: List[float] = []
    for dret, fobs in aligned:
        x_rows.append([1.0] + [fobs.factor_returns[fn] for fn in common_factors])
        y_vals.append(dret.gross_return - fobs.risk_free_rate)

    # Compute X^T X + ridge * I (only regularizing slopes, not intercept, if needed)
    xtx = [[0.0] * p for _ in range(p)]
    xty = [0.0] * p
    for row, y in zip(x_rows, y_vals):
        for i in range(p):
            xty[i] += row[i] * y
            for j in range(p):
                xtx[i][j] += row[i] * row[j]

    reg = config.factor_ridge_regularization
    for i in range(1, p):
        xtx[i][i] += reg

    try:
        coeffs = _solve_linear_system(xtx, xty)
    except ValueError:
        for i in range(p):
            xtx[i][i] += 1e-6
        coeffs = _solve_linear_system(xtx, xty)

    alpha = coeffs[0]
    betas = coeffs[1:]

    # Fit diagnostics: R^2, Adjusted R^2, standard errors & t-stats
    y_mean = sum(y_vals) / n
    ss_tot = sum((y - y_mean) ** 2 for y in y_vals)
    residuals = []
    for row, y in zip(x_rows, y_vals):
        y_hat = sum(c * xv for c, xv in zip(coeffs, row))
        residuals.append(y - y_hat)
    ss_res = sum(e * e for e in residuals)
    r_sq = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-18 else 0.0
    adj_r_sq = (
        1.0 - (1.0 - r_sq) * (n - 1) / (n - p)
        if n > p
        else None
    )

    xtx_inv = _invert_matrix(xtx)
    se_list: List[Optional[float]] = [None] * k
    t_list: List[Optional[float]] = [None] * k
    if xtx_inv is not None and n > p:
        sigma2 = ss_res / (n - p)
        for idx in range(k):
            var_b = sigma2 * xtx_inv[idx + 1][idx + 1]
            if var_b > 0:
                se = math.sqrt(var_b)
                se_list[idx] = se
                t_list[idx] = betas[idx] / se if se > 1e-15 else None

    # Link daily factor contributions using Carino gross weights on the aligned sample
    from pae.engine.returns import compute_carino_weights

    sample_pairs = [(dret.trade_date, dret.gross_return) for dret, _ in aligned]
    w_sample = compute_carino_weights(sample_pairs)
    sample_cum = 1.0
    for _, r in sample_pairs:
        sample_cum *= 1.0 + r
    sample_cum -= 1.0

    linked_rf = 0.0
    linked_specific = 0.0
    linked_by_factor = {fn: 0.0 for fn in common_factors}

    for idx, (dret, fobs) in enumerate(aligned):
        w = w_sample[dret.trade_date]
        linked_rf += w * fobs.risk_free_rate
        factor_sum_t = 0.0
        for fi, fn in enumerate(common_factors):
            c_ft = betas[fi] * fobs.factor_returns[fn]
            linked_by_factor[fn] += w * c_ft
            factor_sum_t += c_ft
        specific_t = dret.gross_return - fobs.risk_free_rate - factor_sum_t
        linked_specific += w * specific_t

    linked_factor_total = sum(linked_by_factor.values())
    total_reconstructed = linked_rf + linked_factor_total + linked_specific

    factor_results = []
    for fi, fn in enumerate(common_factors):
        mean_f = sum(fobs.factor_returns[fn] for _, fobs in aligned) / n
        factor_results.append(
            FactorExposureResult(
                factor_name=fn,
                beta=betas[fi],
                std_error=se_list[fi],
                t_stat=t_list[fi],
                linked_contribution=linked_by_factor[fn],
                mean_daily_factor_return=mean_f,
            )
        )

    return FactorAttributionReport(
        status="AVAILABLE",
        disclaimer=config.non_causation_disclaimer,
        sample_size=n,
        factor_names=common_factors,
        r_squared=r_sq,
        adjusted_r_squared=adj_r_sq,
        daily_alpha=alpha,
        linked_risk_free_contribution=linked_rf,
        linked_factor_total_contribution=linked_factor_total,
        linked_specific_return=linked_specific,
        explained_plus_specific_return=total_reconstructed,
        portfolio_gross_return_sample=sample_cum,
        residual_check=sample_cum - total_reconstructed,
        factors=factor_results,
        notes=[
            f"OLS regression estimated over {n} aligned trading days with {k} factors.",
            config.non_causation_disclaimer,
        ],
    )
