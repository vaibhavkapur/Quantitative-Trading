"""Volatility evaluation metrics (QLIKE, MAE, RMSE, calibration), conformal intervals, and regime diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from volatility_forecasting.config import RegimeQuantileConfig
from volatility_forecasting.contracts import validate_nonnegative_forecasts

REGIME_ORDER: tuple[str, ...] = ("low", "normal", "high", "shock")


def variance_to_vol(variance: np.ndarray, annualization_factor: int = 252) -> np.ndarray:
    """Convert daily realized variance to annualized volatility."""
    arr = np.maximum(0.0, np.asarray(variance, dtype=np.float64))
    return np.sqrt(float(annualization_factor) * arr)


def qlike_loss(
    y_true_var: np.ndarray,
    y_pred_var: np.ndarray,
    *,
    variance_floor: float = 1e-10,
) -> float:
    """Quasi-Likelihood (QLIKE) loss on variance scale.

    QLIKE(v_hat, v) = (v / v_hat) - log(v / v_hat) - 1 >= 0.
    Both inputs are floored at `variance_floor` so constant-return zero-variance
    cases evaluate cleanly to 0.0 without division-by-zero or log(0).
    """
    y_true = np.maximum(np.asarray(y_true_var, dtype=np.float64), variance_floor)
    y_pred = np.maximum(
        validate_nonnegative_forecasts(y_pred_var, model_name="qlike_input", quantity="variance"),
        variance_floor,
    )
    ratio = y_true / y_pred
    losses = ratio - np.log(ratio) - 1.0
    return float(np.mean(np.maximum(0.0, losses)))


def mae_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64))))


def rmse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def mincer_zarnowitz_regression(
    y_true_var: np.ndarray,
    y_pred_var: np.ndarray,
) -> dict[str, float]:
    """Mincer-Zarnowitz calibration regression: y_true = intercept + slope * y_pred + eps.

    Well-calibrated forecasts have intercept ≈ 0 and slope ≈ 1.
    """
    y = np.asarray(y_true_var, dtype=np.float64)
    x = np.asarray(y_pred_var, dtype=np.float64)
    if len(y) < 2 or float(np.var(x)) <= 1e-24:
        return {
            "mz_intercept": float(np.mean(y) - np.mean(x)) if len(y) > 0 else 0.0,
            "mz_slope": 1.0 if (len(y) > 0 and float(np.var(y)) <= 1e-24) else 0.0,
            "mz_r2": 1.0 if (len(y) > 0 and float(np.var(y)) <= 1e-24 and np.allclose(y, x)) else 0.0,
        }

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    cov_xy = float(np.mean((x - x_mean) * (y - y_mean)))
    var_x = float(np.mean((x - x_mean) ** 2))
    slope = cov_xy / var_x
    intercept = y_mean - slope * x_mean
    y_hat = intercept + slope * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-24 else 1.0
    return {
        "mz_intercept": float(intercept),
        "mz_slope": float(slope),
        "mz_r2": float(r2),
    }


@dataclass(frozen=True)
class ConformalMultiplierTable:
    """Multiplicative residual quantiles fit strictly on calibration window."""

    levels: tuple[float, ...]
    multipliers: dict[float, tuple[float, float]]  # level -> (low_mult, high_mult)
    n_cal: int


def fit_conformal_multipliers(
    y_cal_var: np.ndarray,
    y_cal_pred_var: np.ndarray,
    levels: tuple[float, ...] = (0.50, 0.80, 0.90),
    *,
    variance_floor: float = 1e-10,
) -> ConformalMultiplierTable:
    """Fit nonnegative multiplicative conformal interval multipliers on calibration residuals.

    Ratio score: r_i = max(y_cal_i, floor) / max(y_pred_i, floor).
    For nominal coverage (1 - alpha), uses finite-sample conformal quantiles
    at alpha/2 and 1 - alpha/2 so lower bounds are guaranteed >= 0.
    """
    y_true = np.maximum(np.asarray(y_cal_var, dtype=np.float64), 0.0)
    y_pred = np.maximum(np.asarray(y_cal_pred_var, dtype=np.float64), 0.0)
    n = len(y_true)
    if n == 0:
        return ConformalMultiplierTable(
            levels=levels,
            multipliers={lvl: (0.5, 1.5) for lvl in levels},
            n_cal=0,
        )

    # Constant-zero calibration edge case
    if float(np.max(y_true)) <= 1e-16 and float(np.max(y_pred)) <= 1e-16:
        return ConformalMultiplierTable(
            levels=levels,
            multipliers={lvl: (1.0, 1.0) for lvl in levels},
            n_cal=n,
        )

    ratios = np.maximum(y_true, variance_floor) / np.maximum(y_pred, variance_floor)
    ratios = np.clip(ratios, 0.0, 25.0)

    mults: dict[float, tuple[float, float]] = {}
    for lvl in levels:
        alpha = 1.0 - float(lvl)
        # Finite-sample conservative conformal quantile adjustment
        q_low = max(0.0, (alpha / 2.0) * (1.0 - 1.0 / (n + 1.0)))
        q_high = min(1.0, (1.0 - alpha / 2.0) * (1.0 + 1.0 / (n + 1.0)))
        m_low = float(max(0.0, np.quantile(ratios, q_low)))
        m_high = float(max(m_low, np.quantile(ratios, q_high)))
        mults[float(lvl)] = (m_low, m_high)

    return ConformalMultiplierTable(levels=tuple(float(x) for x in levels), multipliers=mults, n_cal=n)


def apply_forecast_intervals(
    y_pred_var: np.ndarray,
    table: ConformalMultiplierTable,
    *,
    annualization_factor: int = 252,
) -> dict[float, dict[str, np.ndarray]]:
    """Apply conformal multipliers to point variance forecasts, returning variance & vol intervals."""
    point_var = validate_nonnegative_forecasts(
        y_pred_var, model_name="interval_point", quantity="variance"
    )
    intervals: dict[float, dict[str, np.ndarray]] = {}
    for lvl, (m_low, m_high) in table.multipliers.items():
        low_var = np.maximum(0.0, point_var * m_low)
        high_var = np.maximum(low_var, point_var * m_high)
        intervals[lvl] = {
            "lower_var": low_var,
            "upper_var": high_var,
            "lower_vol": variance_to_vol(low_var, annualization_factor),
            "upper_vol": variance_to_vol(high_var, annualization_factor),
        }
    return intervals


def interval_calibration_metrics(
    y_true_vol: np.ndarray,
    lower_vol: np.ndarray,
    upper_vol: np.ndarray,
    nominal_level: float,
) -> dict[str, float]:
    """Compute empirical coverage, coverage error, average width, and Winkler score on vol scale."""
    y = np.asarray(y_true_vol, dtype=np.float64)
    lo = np.asarray(lower_vol, dtype=np.float64)
    hi = np.asarray(upper_vol, dtype=np.float64)
    alpha = 1.0 - float(nominal_level)

    covered = (y >= lo - 1e-12) & (y <= hi + 1e-12)
    emp_cov = float(np.mean(covered)) if len(y) > 0 else 0.0
    width = hi - lo
    penalty = np.where(y < lo, (2.0 / alpha) * (lo - y), 0.0) + np.where(
        y > hi, (2.0 / alpha) * (y - hi), 0.0
    )
    winkler = float(np.mean(width + penalty)) if len(y) > 0 else 0.0
    return {
        "nominal_coverage": float(nominal_level),
        "empirical_coverage": emp_cov,
        "coverage_error": emp_cov - float(nominal_level),
        "mean_width_vol": float(np.mean(width)) if len(y) > 0 else 0.0,
        "winkler_score_vol": winkler,
    }


@dataclass(frozen=True)
class RegimeThresholds:
    low_upper: float
    normal_upper: float
    high_upper: float


def fit_regime_thresholds(
    train_rv: np.ndarray,
    q_cfg: RegimeQuantileConfig,
) -> RegimeThresholds:
    """Fit causal volatility regime thresholds strictly on training-window realized variance."""
    arr = np.maximum(0.0, np.asarray(train_rv, dtype=np.float64))
    if len(arr) == 0:
        return RegimeThresholds(0.0, 0.0, 0.0)
    return RegimeThresholds(
        low_upper=float(np.quantile(arr, q_cfg.low_upper)),
        normal_upper=float(np.quantile(arr, q_cfg.normal_upper)),
        high_upper=float(np.quantile(arr, q_cfg.high_upper)),
    )


def assign_regimes(rv_series: np.ndarray, thresholds: RegimeThresholds) -> np.ndarray:
    """Classify each observation causally into low / normal / high / shock regimes."""
    arr = np.maximum(0.0, np.asarray(rv_series, dtype=np.float64))
    labels = np.full(len(arr), "normal", dtype=object)
    # Handle degenerate constant series cleanly
    if thresholds.high_upper <= 1e-16:
        labels[:] = "normal"
        return labels
    labels[arr <= thresholds.low_upper] = "low"
    labels[(arr > thresholds.low_upper) & (arr <= thresholds.normal_upper)] = "normal"
    labels[(arr > thresholds.normal_upper) & (arr <= thresholds.high_upper)] = "high"
    labels[arr > thresholds.high_upper] = "shock"
    return labels


def compute_regime_diagnostics(
    model_name: str,
    regimes: np.ndarray,
    y_true_var: np.ndarray,
    y_pred_var: np.ndarray,
    *,
    lower_80_vol: np.ndarray | None = None,
    upper_80_vol: np.ndarray | None = None,
    annualization_factor: int = 252,
    variance_floor: float = 1e-10,
) -> list[dict[str, Any]]:
    """Compute per-regime QLIKE, MAE, RMSE, bias, and 80% interval coverage."""
    y_true_v = np.asarray(y_true_var, dtype=np.float64)
    y_pred_v = np.asarray(y_pred_var, dtype=np.float64)
    y_true_vol = variance_to_vol(y_true_v, annualization_factor)
    y_pred_vol = variance_to_vol(y_pred_v, annualization_factor)

    rows: list[dict[str, Any]] = []
    for reg in REGIME_ORDER:
        mask = regimes == reg
        n_obs = int(np.sum(mask))
        if n_obs == 0:
            continue
        cov_80: float | None = None
        if lower_80_vol is not None and upper_80_vol is not None:
            cov_80 = float(
                np.mean(
                    (y_true_vol[mask] >= lower_80_vol[mask] - 1e-12)
                    & (y_true_vol[mask] <= upper_80_vol[mask] + 1e-12)
                )
            )
        rows.append(
            {
                "model": model_name,
                "regime": reg,
                "n_obs": n_obs,
                "mean_true_vol": float(np.mean(y_true_vol[mask])),
                "mean_pred_vol": float(np.mean(y_pred_vol[mask])),
                "bias_vol": float(np.mean(y_pred_vol[mask] - y_true_vol[mask])),
                "qlike": qlike_loss(
                    y_true_v[mask], y_pred_v[mask], variance_floor=variance_floor
                ),
                "mae_vol": mae_loss(y_true_vol[mask], y_pred_vol[mask]),
                "rmse_vol": rmse_loss(y_true_vol[mask], y_pred_vol[mask]),
                "coverage_80": cov_80 if cov_80 is not None else math.nan,
            }
        )
    return rows
