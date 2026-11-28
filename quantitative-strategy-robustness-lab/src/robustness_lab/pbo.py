"""Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric Cross-Validation.

Pure domain module (no file I/O). Implements Bailey, Borwein, Lopez de Prado,
and Zhu (2015), 'The Probability of Backtest Overfitting'.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats

from .models import (
    ContractValidationError,
    InsufficientSampleError,
    PBOResult,
    UnsupportedConfigurationError,
)


def _column_sharpes(mat: np.ndarray, annualization_factor: int) -> np.ndarray:
    """Vectorized annualized Sharpe ratios across columns of a T x N matrix."""
    means = np.mean(mat, axis=0)
    stds = np.std(mat, axis=0, ddof=1)
    scale = math.sqrt(float(annualization_factor))
    sharpes = np.where(stds > 1e-15, (means / stds) * scale, 0.0)
    return sharpes


def compute_pbo_cscv(
    trial_returns_matrix: np.ndarray,
    n_partitions: int = 8,
    annualization_factor: int = 252,
    max_combinations: int = 12870,
    seed: int = 42,
    min_block_observations: int = 5,
) -> PBOResult:
    """Run CSCV to estimate Probability of Backtest Overfitting (PBO).

    Parameters:
      trial_returns_matrix: 2D array (T x N) of net returns across N strategy trials.
      n_partitions: Even integer S >= 4 splitting the sample into S contiguous blocks.
      annualization_factor: Periods per year.
      max_combinations: Maximum C(S, S/2) splits evaluated; if exceeded, deterministic
        stratified random sampling without replacement is used.
      seed: Fixed RNG seed used only when sampling combinations.
      min_block_observations: Minimum observations per partition block.
    """
    mat = np.asarray(trial_returns_matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ContractValidationError(f"trial_returns_matrix must be 2D (T x N), got shape {mat.shape}")
    if not np.all(np.isfinite(mat)):
        raise ContractValidationError("trial_returns_matrix contains NaN or infinite values")

    t_obs, n_trials = mat.shape
    if n_trials < 2:
        raise InsufficientSampleError(f"CSCV PBO requires at least 2 trials, got {n_trials}")
    if not isinstance(n_partitions, int) or n_partitions < 4 or n_partitions % 2 != 0:
        raise UnsupportedConfigurationError(
            f"n_partitions must be an even integer >= 4, got {n_partitions!r}"
        )
    if max_combinations < 1:
        raise UnsupportedConfigurationError(f"max_combinations must be >= 1, got {max_combinations!r}")

    block_size = t_obs // n_partitions
    if block_size < min_block_observations:
        raise InsufficientSampleError(
            f"Sample length {t_obs} yields block size {block_size} for S={n_partitions}; "
            f"minimum required per block is {min_block_observations}."
        )

    # Align to exact multiple of n_partitions by retaining the most recent usable observations
    usable_t = block_size * n_partitions
    trimmed = mat[t_obs - usable_t :, :]
    blocks = [trimmed[i * block_size : (i + 1) * block_size, :] for i in range(n_partitions)]

    half = n_partitions // 2
    total_combos = math.comb(n_partitions, half)
    all_indices = tuple(range(n_partitions))

    if total_combos <= max_combinations:
        chosen_combos = list(itertools.combinations(all_indices, half))
        exact_enum = True
    else:
        rng = np.random.default_rng(seed)
        seen: set[tuple[int, ...]] = set()
        chosen_combos = []
        while len(chosen_combos) < max_combinations:
            combo = tuple(sorted(int(x) for x in rng.choice(n_partitions, size=half, replace=False)))
            if combo not in seen:
                seen.add(combo)
                chosen_combos.append(combo)
        exact_enum = False

    logits: list[float] = []
    oos_ranks: list[float] = []
    is_best_sharpes: list[float] = []
    oos_selected_sharpes: list[float] = []

    all_idx_set = set(all_indices)
    for train_idx in chosen_combos:
        test_idx = tuple(sorted(all_idx_set.difference(train_idx)))
        train_mat = np.vstack([blocks[i] for i in train_idx])
        test_mat = np.vstack([blocks[i] for i in test_idx])

        is_sr = _column_sharpes(train_mat, annualization_factor)
        oos_sr = _column_sharpes(test_mat, annualization_factor)

        best_n = int(np.argmax(is_sr))
        selected_is = float(is_sr[best_n])
        selected_oos = float(oos_sr[best_n])

        # Rank OOS performance from 1 (worst) to N (best) with average tie-breaking
        ranks = stats.rankdata(oos_sr, method="average")
        omega_bar = float(ranks[best_n]) / float(n_trials + 1)
        logit = math.log(omega_bar / (1.0 - omega_bar))

        logits.append(logit)
        oos_ranks.append(omega_bar)
        is_best_sharpes.append(selected_is)
        oos_selected_sharpes.append(selected_oos)

    logits_arr = np.asarray(logits, dtype=np.float64)
    is_arr = np.asarray(is_best_sharpes, dtype=np.float64)
    oos_arr = np.asarray(oos_selected_sharpes, dtype=np.float64)

    pbo = float(np.mean(logits_arr <= 0.0))
    prob_oos_loss = float(np.mean(oos_arr < 0.0))
    median_logit = float(np.median(logits_arr))
    mean_oos_rank = float(np.mean(oos_ranks))

    # Performance degradation linear fit: OOS_SR = intercept + slope * IS_SR
    if float(np.std(is_arr)) > 1e-12:
        lin = stats.linregress(is_arr, oos_arr)
        slope = float(lin.slope)
        intercept = float(lin.intercept)
    else:
        slope = 0.0
        intercept = float(np.mean(oos_arr))

    warnings: list[str] = []
    if not exact_enum:
        warnings.append(
            f"Evaluated {len(chosen_combos)} sampled CSCV splits out of {total_combos} total combinations (seed={seed})."
        )
    if n_trials < 4:
        warnings.append(
            f"Small trial count (N={n_trials}) makes discrete OOS ranks coarse; interpret PBO cautiously."
        )
    if usable_t < t_obs:
        warnings.append(
            f"Dropped {t_obs - usable_t} earliest observation(s) to form {n_partitions} equal-length blocks of {block_size}."
        )

    assumptions = (
        "CSCV assumes return blocks are approximately stationary and exchangeable across time partitions.",
        "Strong serial dependence or regime breaks spanning partition boundaries can distort PBO.",
        "OOS relative rank uses (rank / (N + 1)) with average tie-breaking so logits remain finite.",
        "Low PBO indicates consistency across historical partitions, not a guarantee of future out-of-sample returns.",
    )

    return PBOResult(
        n_partitions=n_partitions,
        n_combinations_evaluated=len(chosen_combos),
        total_possible_combinations=total_combos,
        exact_enumeration=exact_enum,
        n_trials=n_trials,
        n_observations_used=usable_t,
        pbo=pbo,
        prob_oos_loss=prob_oos_loss,
        median_logit=median_logit,
        mean_oos_rank=mean_oos_rank,
        is_to_oos_regression_slope=slope,
        is_to_oos_regression_intercept=intercept,
        mean_is_sharpe_selected=float(np.mean(is_arr)),
        mean_oos_sharpe_selected=float(np.mean(oos_arr)),
        assumptions=assumptions,
        warnings=tuple(warnings),
    )
