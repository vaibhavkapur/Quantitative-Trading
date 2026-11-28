"""Stationary and circular block bootstrap diagnostics for time-series returns.

Pure domain module (no file I/O). Preserves short-range serial dependence and
volatility clustering while estimating confidence intervals for Sharpe ratio,
annualized mean return, and maximum drawdown.
"""

from __future__ import annotations

import math

import numpy as np

from .models import (
    BootstrapResult,
    ContractValidationError,
    InsufficientSampleError,
    UnsupportedConfigurationError,
)
from .statistics import compute_sharpe_ratio


def compute_max_drawdown(returns: np.ndarray) -> float:
    """Compute maximum peak-to-trough drawdown as a positive magnitude in [0, 1]."""
    arr = np.asarray(returns, dtype=np.float64)
    equity = np.cumprod(1.0 + arr)
    running_peak = np.maximum.accumulate(np.maximum(equity, 1e-12))
    drawdowns = 1.0 - (equity / running_peak)
    return float(np.max(drawdowns)) if drawdowns.size > 0 else 0.0


def _sample_circular_block_indices(
    rng: np.random.Generator,
    n_obs: int,
    block_length: int,
    n_resamples: int,
) -> np.ndarray:
    """Generate (n_resamples, n_obs) index matrix via circular block bootstrap."""
    n_blocks = math.ceil(n_obs / float(block_length))
    starts = rng.integers(0, n_obs, size=(n_resamples, n_blocks))
    offsets = np.arange(block_length, dtype=np.int64)
    # Broadcast: (n_resamples, n_blocks, block_length) -> flatten last 2 dims -> slice n_obs
    idx = (starts[:, :, None] + offsets[None, None, :]) % n_obs
    return idx.reshape(n_resamples, n_blocks * block_length)[:, :n_obs]


def _sample_stationary_block_indices(
    rng: np.random.Generator,
    n_obs: int,
    mean_block_length: int,
    n_resamples: int,
) -> np.ndarray:
    """Generate (n_resamples, n_obs) index matrix via Politis-Romano stationary bootstrap.

    At each step t, starts a new random position with probability p = 1 / mean_block_length,
    otherwise continues sequentially (mod n_obs).
    """
    p_new = 1.0 / float(mean_block_length)
    indices = np.empty((n_resamples, n_obs), dtype=np.int64)
    indices[:, 0] = rng.integers(0, n_obs, size=n_resamples)
    if n_obs == 1:
        return indices

    new_block_flags = rng.random((n_resamples, n_obs - 1)) < p_new
    random_starts = rng.integers(0, n_obs, size=(n_resamples, n_obs - 1))

    for t in range(1, n_obs):
        cont = (indices[:, t - 1] + 1) % n_obs
        indices[:, t] = np.where(new_block_flags[:, t - 1], random_starts[:, t - 1], cont)
    return indices


def run_block_bootstrap(
    returns: np.ndarray,
    n_resamples: int = 1000,
    block_length: int = 10,
    method: str = "stationary",
    confidence_level: float = 0.95,
    annualization_factor: int = 252,
    seed: int = 42,
    min_observations: int = 30,
) -> BootstrapResult:
    """Run deterministic block bootstrap on a 1D net return series."""
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim != 1:
        raise ContractValidationError(f"returns must be 1D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ContractValidationError("returns contains NaN or infinite values")
    if arr.size < min_observations:
        raise InsufficientSampleError(
            f"Block bootstrap requires at least {min_observations} observations, got {arr.size}"
        )
    if method not in ("stationary", "circular"):
        raise UnsupportedConfigurationError(
            f"Unsupported bootstrap method '{method}'. Supported: 'stationary', 'circular'"
        )
    if n_resamples < 50:
        raise UnsupportedConfigurationError(f"n_resamples must be >= 50, got {n_resamples}")
    if block_length < 1 or block_length >= arr.size:
        raise UnsupportedConfigurationError(
            f"block_length must satisfy 1 <= block_length < T ({arr.size}), got {block_length}"
        )
    if not (0.5 < confidence_level < 1.0):
        raise UnsupportedConfigurationError(
            f"confidence_level must be in (0.5, 1.0), got {confidence_level}"
        )

    _, observed_sr = compute_sharpe_ratio(arr, annualization_factor=annualization_factor)
    observed_mdd = compute_max_drawdown(arr)

    rng = np.random.default_rng(seed)
    if method == "stationary":
        idx = _sample_stationary_block_indices(rng, arr.size, block_length, n_resamples)
    else:
        idx = _sample_circular_block_indices(rng, arr.size, block_length, n_resamples)

    samples = arr[idx]  # (n_resamples, T)
    means = np.mean(samples, axis=1)
    stds = np.std(samples, axis=1, ddof=1)
    scale = math.sqrt(float(annualization_factor))
    boot_sharpes = np.where(stds > 1e-15, (means / stds) * scale, 0.0)
    boot_ann_means = means * float(annualization_factor)

    # Vectorized max drawdown across resamples
    equity = np.cumprod(1.0 + samples, axis=1)
    peaks = np.maximum.accumulate(np.maximum(equity, 1e-12), axis=1)
    boot_mdds = np.max(1.0 - (equity / peaks), axis=1)

    alpha = 1.0 - confidence_level
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)

    sr_lo, sr_hi = np.percentile(boot_sharpes, [lo_q, hi_q])
    mean_lo, mean_hi = np.percentile(boot_ann_means, [lo_q, hi_q])
    mdd_hi = float(np.percentile(boot_mdds, 100.0 * confidence_level))
    p_val_zero = float(np.mean(boot_sharpes <= 0.0))

    assumptions = (
        f"Uses {method} block bootstrap (mean/fixed block_length={block_length}) to preserve local serial dependence.",
        "Assumes weak stationarity of the return-generating process across the sample.",
        "Bootstrap percentile intervals reflect historical sampling variability, not structural regime shifts.",
    )

    return BootstrapResult(
        method=method,
        n_resamples=n_resamples,
        block_length=block_length,
        seed=seed,
        confidence_level=confidence_level,
        observed_sharpe_annual=observed_sr,
        sharpe_ci_lower=float(sr_lo),
        sharpe_ci_upper=float(sr_hi),
        sharpe_std_error=float(np.std(boot_sharpes, ddof=1)),
        mean_return_ci_lower=float(mean_lo),
        mean_return_ci_upper=float(mean_hi),
        max_drawdown_observed=observed_mdd,
        max_drawdown_ci_upper=mdd_hi,
        p_value_vs_zero=p_val_zero,
        assumptions=assumptions,
    )
