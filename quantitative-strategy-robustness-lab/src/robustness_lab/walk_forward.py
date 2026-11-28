"""Walk-forward out-of-sample evaluation with strict training-window selection and purging.

Pure domain module (no file I/O). Enforces temporal separation:
  - Trial/parameter selection and any in-sample ranking occur strictly on training slices.
  - A configurable purge gap separates training end from OOS test start to mitigate label/serial overlap.
  - Exact split timestamps and selected trial IDs are stored on every fold for reproducibility.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .models import (
    ContractValidationError,
    InsufficientSampleError,
    UnsupportedConfigurationError,
    WalkForwardFold,
    WalkForwardResult,
)
from .statistics import compute_sharpe_ratio


def run_walk_forward(
    timestamps: tuple[str, ...],
    trial_ids: list[str],
    net_returns_matrix: np.ndarray,
    gross_returns_matrix: Optional[np.ndarray] = None,
    benchmark_returns: Optional[np.ndarray] = None,
    mode: str = "expanding",
    train_window: int = 120,
    test_window: int = 40,
    step_size: int = 40,
    purge_gap: int = 5,
    annualization_factor: int = 252,
) -> WalkForwardResult:
    """Execute walk-forward selection and OOS stitching across candidate trials."""
    if mode not in ("expanding", "rolling"):
        raise UnsupportedConfigurationError(f"Unsupported walk-forward mode '{mode}'. Use 'expanding' or 'rolling'.")
    if train_window < 10 or test_window < 5 or step_size < 1 or purge_gap < 0:
        raise UnsupportedConfigurationError(
            f"Invalid walk-forward parameters: train={train_window}, test={test_window}, "
            f"step={step_size}, purge={purge_gap}"
        )

    net_mat = np.asarray(net_returns_matrix, dtype=np.float64)
    if net_mat.ndim != 2:
        raise ContractValidationError(f"net_returns_matrix must be 2D (T x N), got shape {net_mat.shape}")
    t_obs, n_trials = net_mat.shape
    if len(timestamps) != t_obs:
        raise ContractValidationError(f"timestamps length ({len(timestamps)}) != matrix rows ({t_obs})")
    if len(trial_ids) != n_trials or n_trials < 1:
        raise ContractValidationError(f"trial_ids length ({len(trial_ids)}) != matrix cols ({n_trials})")
    if not np.all(np.isfinite(net_mat)):
        raise ContractValidationError("net_returns_matrix contains NaN or infinite values")

    gross_mat = (
        np.asarray(gross_returns_matrix, dtype=np.float64)
        if gross_returns_matrix is not None
        else net_mat.copy()
    )
    if gross_mat.shape != net_mat.shape:
        raise ContractValidationError("gross_returns_matrix shape must match net_returns_matrix")

    if benchmark_returns is not None:
        bench_arr = np.asarray(benchmark_returns, dtype=np.float64)
        if bench_arr.shape != (t_obs,):
            raise ContractValidationError(f"benchmark_returns shape {bench_arr.shape} != ({t_obs},)")
        if not np.all(np.isfinite(bench_arr)):
            raise ContractValidationError("benchmark_returns contains NaN or infinite values")
    else:
        bench_arr = None

    min_required = train_window + purge_gap + test_window
    if t_obs < min_required:
        raise InsufficientSampleError(
            f"Walk-forward requires at least {min_required} observations "
            f"(train={train_window} + purge={purge_gap} + test={test_window}), got {t_obs}"
        )

    folds: list[WalkForwardFold] = []
    oos_net_chunks: list[np.ndarray] = []
    oos_gross_chunks: list[np.ndarray] = []
    oos_bench_chunks: list[np.ndarray] = []
    scale = math.sqrt(float(annualization_factor))

    train_end_idx = train_window
    fold_idx = 0
    while train_end_idx + purge_gap + test_window <= t_obs:
        train_start_idx = 0 if mode == "expanding" else train_end_idx - train_window
        test_start_idx = train_end_idx + purge_gap
        test_end_idx = test_start_idx + test_window

        # Fit / select strictly on training window
        train_slice = net_mat[train_start_idx:train_end_idx, :]
        tr_means = np.mean(train_slice, axis=0)
        tr_stds = np.std(train_slice, axis=0, ddof=1)
        tr_sharpes = np.where(tr_stds > 1e-15, (tr_means / tr_stds) * scale, -1e9)
        best_col = int(np.argmax(tr_sharpes))
        best_train_sr = float(tr_sharpes[best_col])

        # Evaluate strictly out-of-sample
        test_net = net_mat[test_start_idx:test_end_idx, best_col]
        test_gross = gross_mat[test_start_idx:test_end_idx, best_col]
        _, test_net_sr = compute_sharpe_ratio(test_net, annualization_factor=annualization_factor)

        test_bench_sr: Optional[float] = None
        if bench_arr is not None:
            test_bench = bench_arr[test_start_idx:test_end_idx]
            _, test_bench_sr = compute_sharpe_ratio(test_bench, annualization_factor=annualization_factor)
            oos_bench_chunks.append(test_bench)

        purge_end_ts = timestamps[test_start_idx - 1] if purge_gap > 0 else timestamps[train_end_idx - 1]

        folds.append(
            WalkForwardFold(
                fold_index=fold_idx,
                train_start=timestamps[train_start_idx],
                train_end=timestamps[train_end_idx - 1],
                purge_end=purge_end_ts,
                test_start=timestamps[test_start_idx],
                test_end=timestamps[test_end_idx - 1],
                n_train=train_end_idx - train_start_idx,
                n_test=test_end_idx - test_start_idx,
                selected_trial_id=trial_ids[best_col],
                train_net_sharpe=best_train_sr,
                test_net_sharpe=test_net_sr,
                test_benchmark_sharpe=test_bench_sr,
                test_mean_return=float(np.mean(test_net)),
            )
        )
        oos_net_chunks.append(test_net)
        oos_gross_chunks.append(test_gross)

        fold_idx += 1
        train_end_idx += step_size

    stitched_oos_net = np.concatenate(oos_net_chunks)
    stitched_oos_gross = np.concatenate(oos_gross_chunks)

    _, agg_oos_net_sr = compute_sharpe_ratio(stitched_oos_net, annualization_factor=annualization_factor)
    _, agg_oos_gross_sr = compute_sharpe_ratio(stitched_oos_gross, annualization_factor=annualization_factor)

    agg_oos_bench_sr: Optional[float] = None
    if oos_bench_chunks:
        stitched_bench = np.concatenate(oos_bench_chunks)
        _, agg_oos_bench_sr = compute_sharpe_ratio(stitched_bench, annualization_factor=annualization_factor)

    mean_is_sr = float(np.mean([f.train_net_sharpe for f in folds]))
    if mean_is_sr > 1e-6:
        degradation = (mean_is_sr - agg_oos_net_sr) / mean_is_sr
    else:
        degradation = 1.0 if agg_oos_net_sr <= 0.0 else 0.0

    hit_rate = float(np.mean([1.0 if f.test_net_sharpe > 0.0 else 0.0 for f in folds]))

    assumptions = (
        f"Walk-forward uses {mode} training windows with a {purge_gap}-period purge gap before each OOS test fold.",
        "Candidate selection is performed strictly using training-window net Sharpe ratios.",
        "Stitched OOS metrics assume non-overlapping test folds when step_size >= test_window.",
    )

    return WalkForwardResult(
        mode=mode,
        train_window=train_window,
        test_window=test_window,
        step_size=step_size,
        purge_gap=purge_gap,
        n_folds=len(folds),
        total_oos_observations=int(stitched_oos_net.size),
        mean_is_net_sharpe=mean_is_sr,
        aggregate_oos_net_sharpe=agg_oos_net_sr,
        aggregate_oos_gross_sharpe=agg_oos_gross_sr,
        aggregate_oos_benchmark_sharpe=agg_oos_bench_sr,
        is_to_oos_degradation_ratio=float(degradation),
        oos_hit_rate=hit_rate,
        folds=tuple(folds),
        assumptions=assumptions,
    )
