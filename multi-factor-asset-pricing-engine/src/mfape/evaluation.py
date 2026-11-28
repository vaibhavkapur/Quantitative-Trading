"""Pure domain module for out-of-sample (OOS) evaluation and CAPM baseline comparison."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from mfape.errors import DateLeakageError, InsufficientObservationsError
from mfape.fama_macbeth import compute_pricing_error_metrics
from mfape.schemas import (
    AlignedPanel,
    ComparisonSummary,
    FactorModelSpec,
    FamaMacBethResult,
    ModelEvaluationResult,
    OOSEvaluationResult,
    TimeSeriesModelResult,
)


def evaluate_out_of_sample(
    train_panel: AlignedPanel,
    test_panel: AlignedPanel,
    ts_train: TimeSeriesModelResult,
    fm_train: FamaMacBethResult,
    spec: FactorModelSpec,
) -> OOSEvaluationResult:
    """Evaluate training-fitted betas and premia on a strictly future test panel.

    Zero date leakage guarantee:
      - Asserts max(train_panel.dates) < min(test_panel.dates).
      - Uses ONLY train-window betas and train-window Fama-MacBeth premia.
      - Never refits preprocessing, betas, or premia on test observations.
    """
    if train_panel.dates[-1] >= test_panel.dates[0]:
        raise DateLeakageError(
            f"OOS evaluation rejected due to date leakage: train_end ({train_panel.dates[-1]}) "
            f">= test_start ({test_panel.dates[0]}).",
            {"train_end": train_panel.dates[-1], "test_start": test_panel.dates[0]},
        )

    asset_to_ts = {ar.asset_id: ar for ar in ts_train.asset_results}
    premia_by_name = {p.name: p.estimate for p in fm_train.premia}
    gamma_0 = premia_by_name.get("gamma_0", 0.0) if spec.include_cs_intercept else 0.0
    lambda_vec = np.array([premia_by_name[f] for f in spec.factors], dtype=np.float64)

    name_to_fidx = {name: i for i, name in enumerate(test_panel.factor_names)}
    f_indices = [name_to_fidx[f] for f in spec.factors]
    test_factors = test_panel.factor_returns[:, f_indices]  # (T_test, K)

    eval_assets: list[str] = []
    realized_test_means: list[float] = []
    predicted_cs_means: list[float] = []
    train_hist_means: list[float] = []

    all_actual_ts: list[np.ndarray] = []
    all_pred_ts_zero_alpha: list[np.ndarray] = []
    all_pred_ts_with_alpha: list[np.ndarray] = []

    for j, asset_id in enumerate(test_panel.assets):
        if asset_id not in asset_to_ts:
            # Asset entered after training window: cannot leak future betas
            continue
        mask_j = test_panel.active_mask[:, j]
        if not np.any(mask_j):
            continue

        ar_train = asset_to_ts[asset_id]
        beta_vec = np.array([ar_train.betas[f] for f in spec.factors], dtype=np.float64)
        alpha_train = ar_train.alpha if ar_train.alpha is not None else 0.0

        y_test_j = test_panel.excess_returns[mask_j, j]
        f_test_j = test_factors[mask_j, :]

        # OOS cross-sectional expected excess return: gamma_0_train + beta_train' * lambda_train
        pred_cs_mean = float(gamma_0 + beta_vec @ lambda_vec)
        realized_mean = float(np.mean(y_test_j))

        # OOS time-series realized factor predictions using frozen train betas
        pred_ts_zero = f_test_j @ beta_vec
        pred_ts_alpha = alpha_train + pred_ts_zero

        eval_assets.append(asset_id)
        realized_test_means.append(realized_mean)
        predicted_cs_means.append(pred_cs_mean)
        train_hist_means.append(ar_train.mean_excess_return)
        all_actual_ts.append(y_test_j)
        all_pred_ts_zero_alpha.append(pred_ts_zero)
        all_pred_ts_with_alpha.append(pred_ts_alpha)

    if len(eval_assets) < 2:
        raise InsufficientObservationsError(
            f"OOS evaluation ({spec.name}) requires at least 2 overlapping train/test assets, "
            f"found {len(eval_assets)}.",
            {"model": spec.name, "evaluated_assets": len(eval_assets)},
        )

    actual_cs_arr = np.array(realized_test_means, dtype=np.float64)
    pred_cs_arr = np.array(predicted_cs_means, dtype=np.float64)
    naive_train_mean_arr = np.array(train_hist_means, dtype=np.float64)

    n_params = len(spec.factors) + (1 if spec.include_cs_intercept else 0)
    oos_cs_metrics = compute_pricing_error_metrics(
        actual_means=actual_cs_arr,
        predicted_means=pred_cs_arr,
        n_params=n_params,
    )

    sse_model_cs = float(np.sum((actual_cs_arr - pred_cs_arr) ** 2))
    sse_naive_cs = float(np.sum((actual_cs_arr - naive_train_mean_arr) ** 2))
    if sse_naive_cs <= 1e-18:
        oos_r2_vs_naive = 1.0 if sse_model_cs <= 1e-18 else 0.0
    else:
        oos_r2_vs_naive = 1.0 - (sse_model_cs / sse_naive_cs)

    # Pooled OOS time-series R^2 and residual error metrics
    y_all = np.concatenate(all_actual_ts)
    pred_zero_all = np.concatenate(all_pred_ts_zero_alpha)
    pred_alpha_all = np.concatenate(all_pred_ts_with_alpha)

    err_zero = y_all - pred_zero_all
    err_alpha = y_all - pred_alpha_all
    ss_tot_ts = float(np.sum((y_all - np.mean(y_all)) ** 2))

    if ss_tot_ts <= 1e-18:
        oos_ts_r2_zero = 1.0 if float(np.sum(err_zero ** 2)) <= 1e-18 else 0.0
        oos_ts_r2_alpha = 1.0 if float(np.sum(err_alpha ** 2)) <= 1e-18 else 0.0
    else:
        oos_ts_r2_zero = 1.0 - float(np.sum(err_zero ** 2)) / ss_tot_ts
        oos_ts_r2_alpha = 1.0 - float(np.sum(err_alpha ** 2)) / ss_tot_ts

    oos_ts_rmse_zero = float(np.sqrt(np.mean(err_zero ** 2)))
    oos_ts_mae_zero = float(np.mean(np.abs(err_zero)))

    oos_errors_by_asset = {
        eval_assets[i]: float(actual_cs_arr[i] - pred_cs_arr[i])
        for i in range(len(eval_assets))
    }

    return OOSEvaluationResult(
        model_name=spec.name,
        train_start=train_panel.dates[0],
        train_end=train_panel.dates[-1],
        test_start=test_panel.dates[0],
        test_end=test_panel.dates[-1],
        n_train_periods=len(train_panel.dates),
        n_test_periods=len(test_panel.dates),
        n_evaluated_assets=len(eval_assets),
        oos_cs_pricing_metrics=oos_cs_metrics,
        oos_ts_r2_zero_alpha=float(oos_ts_r2_zero),
        oos_ts_r2_with_train_alpha=float(oos_ts_r2_alpha),
        oos_ts_rmse_zero_alpha=oos_ts_rmse_zero,
        oos_ts_mae_zero_alpha=oos_ts_mae_zero,
        oos_cs_r2_vs_train_mean=float(oos_r2_vs_naive),
        oos_pricing_errors_by_asset=oos_errors_by_asset,
    )


def build_comparison_summary(
    results: Sequence[ModelEvaluationResult],
    baseline_model_name: str,
) -> ComparisonSummary:
    """Compare all fitted models against the designated baseline model (typically CAPM)."""
    by_name = {r.spec.name: r for r in results}
    baseline = by_name[baseline_model_name]

    base_ts_r2 = baseline.time_series_train.mean_r_squared
    base_ts_abs_alpha = baseline.time_series_train.mean_abs_alpha
    base_fm_rmse = baseline.fama_macbeth_train.pricing_metrics.rmse
    base_fm_r2 = baseline.fama_macbeth_train.pricing_metrics.cross_sectional_r2
    base_oos_cs_rmse = (
        baseline.oos_evaluation.oos_cs_pricing_metrics.rmse
        if baseline.oos_evaluation is not None
        else None
    )
    base_oos_ts_r2 = (
        baseline.oos_evaluation.oos_ts_r2_zero_alpha
        if baseline.oos_evaluation is not None
        else None
    )

    rows: list[dict[str, Any]] = []
    for res in results:
        ts = res.time_series_train
        fm = res.fama_macbeth_train
        oos = res.oos_evaluation

        oos_cs_r2 = oos.oos_cs_pricing_metrics.cross_sectional_r2 if oos else None
        oos_cs_rmse = oos.oos_cs_pricing_metrics.rmse if oos else None
        oos_cs_mae = oos.oos_cs_pricing_metrics.mae if oos else None
        oos_ts_r2_zero = oos.oos_ts_r2_zero_alpha if oos else None
        oos_ts_rmse_zero = oos.oos_ts_rmse_zero_alpha if oos else None
        oos_cs_r2_vs_naive = oos.oos_cs_r2_vs_train_mean if oos else None

        delta_oos_rmse = (
            (oos_cs_rmse - base_oos_cs_rmse)
            if (oos_cs_rmse is not None and base_oos_cs_rmse is not None)
            else None
        )
        delta_oos_ts_r2 = (
            (oos_ts_r2_zero - base_oos_ts_r2)
            if (oos_ts_r2_zero is not None and base_oos_ts_r2 is not None)
            else None
        )

        rows.append(
            {
                "model_name": res.spec.name,
                "is_baseline": res.spec.name == baseline_model_name,
                "factors": list(res.spec.factors),
                "n_factors": len(res.spec.factors),
                "ts_mean_r2": ts.mean_r_squared,
                "ts_mean_adj_r2": ts.mean_adj_r_squared,
                "ts_mean_abs_alpha": ts.mean_abs_alpha,
                "ts_rmse_alpha": ts.rmse_alpha,
                "ts_grs_f_stat": ts.grs_f_stat,
                "ts_grs_pvalue": ts.grs_pvalue,
                "fm_cs_r2": fm.pricing_metrics.cross_sectional_r2,
                "fm_adj_cs_r2": fm.pricing_metrics.adj_cross_sectional_r2,
                "fm_pricing_mae": fm.pricing_metrics.mae,
                "fm_pricing_rmse": fm.pricing_metrics.rmse,
                "fm_avg_period_cs_r2": fm.avg_period_cs_r2,
                "oos_cs_r2": oos_cs_r2,
                "oos_cs_mae": oos_cs_mae,
                "oos_cs_rmse": oos_cs_rmse,
                "oos_cs_r2_vs_train_mean": oos_cs_r2_vs_naive,
                "oos_ts_r2_zero_alpha": oos_ts_r2_zero,
                "oos_ts_rmse_zero_alpha": oos_ts_rmse_zero,
                "delta_ts_mean_r2_vs_baseline": ts.mean_r_squared - base_ts_r2,
                "delta_ts_mean_abs_alpha_vs_baseline": ts.mean_abs_alpha - base_ts_abs_alpha,
                "delta_fm_cs_r2_vs_baseline": fm.pricing_metrics.cross_sectional_r2 - base_fm_r2,
                "delta_fm_pricing_rmse_vs_baseline": fm.pricing_metrics.rmse - base_fm_rmse,
                "delta_oos_cs_rmse_vs_baseline": delta_oos_rmse,
                "delta_oos_ts_r2_vs_baseline": delta_oos_ts_r2,
            }
        )

    return ComparisonSummary(
        baseline_model_name=baseline_model_name,
        rows=tuple(rows),
    )
