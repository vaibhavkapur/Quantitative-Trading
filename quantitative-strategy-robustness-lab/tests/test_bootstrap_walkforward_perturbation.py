"""Unit and integration tests for block bootstrap, walk-forward, and parameter perturbation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from robustness_lab.bootstrap import run_block_bootstrap
from robustness_lab.models import (
    CostModel,
    ParameterGridSpec,
    TrialRecord,
    TrialStatus,
    UnsupportedConfigurationError,
)
from robustness_lab.perturbation import evaluate_parameter_perturbation
from robustness_lab.walk_forward import run_walk_forward


def test_block_bootstrap_reproducible_and_discriminates_signal_from_null() -> None:
    rng = np.random.default_rng(701)
    strong = rng.normal(loc=0.0015, scale=0.008, size=300)
    null = rng.normal(loc=0.0, scale=0.008, size=300)
    null = null - np.mean(null)  # exact zero-drift null sample

    b1 = run_block_bootstrap(strong, n_resamples=400, block_length=10, method="stationary", seed=42)
    b2 = run_block_bootstrap(strong, n_resamples=400, block_length=10, method="stationary", seed=42)
    assert b1 == b2
    assert b1.sharpe_ci_lower > 0.5
    assert b1.max_drawdown_ci_upper >= b1.max_drawdown_observed

    b_circ = run_block_bootstrap(strong, n_resamples=400, block_length=10, method="circular", seed=42)
    assert b_circ.sharpe_ci_lower > 0.5

    b_null = run_block_bootstrap(null, n_resamples=400, block_length=10, method="stationary", seed=42)
    assert b_null.sharpe_ci_lower < 0.0 < b_null.sharpe_ci_upper

    with pytest.raises(UnsupportedConfigurationError):
        run_block_bootstrap(strong, method="wild_unsupported")


def test_walk_forward_strict_temporal_separation_and_purge_gap() -> None:
    rng = np.random.default_rng(702)
    t_obs, n_trials = 320, 4
    ts = tuple(d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-02", periods=t_obs))
    mat = rng.normal(loc=0.0, scale=0.009, size=(t_obs, n_trials))
    # Trial 2 has genuine persistent positive drift across entire horizon
    mat[:, 2] = rng.normal(loc=0.0014, scale=0.007, size=t_obs)
    bench = rng.normal(loc=0.0001, scale=0.007, size=t_obs)

    wf_exp = run_walk_forward(
        timestamps=ts,
        trial_ids=[f"t{i}" for i in range(n_trials)],
        net_returns_matrix=mat,
        benchmark_returns=bench,
        mode="expanding",
        train_window=120,
        test_window=40,
        step_size=40,
        purge_gap=5,
    )
    assert wf_exp.n_folds == 4
    assert wf_exp.total_oos_observations == 160
    assert wf_exp.aggregate_oos_net_sharpe > 1.0
    assert wf_exp.aggregate_oos_benchmark_sharpe is not None

    # Verify strict purge gap and monotonically expanding training windows
    for f in wf_exp.folds:
        assert f.train_end < f.purge_end < f.test_start <= f.test_end
    assert wf_exp.folds[1].n_train > wf_exp.folds[0].n_train

    # Verify rolling window keeps n_train constant
    wf_roll = run_walk_forward(
        timestamps=ts,
        trial_ids=[f"t{i}" for i in range(n_trials)],
        net_returns_matrix=mat,
        mode="rolling",
        train_window=120,
        test_window=40,
        step_size=40,
        purge_gap=5,
    )
    assert all(f.n_train == 120 for f in wf_roll.folds)


def test_perturbation_distinguishes_plateau_from_isolated_spike() -> None:
    grid = ParameterGridSpec(
        grid_id="g",
        version="1",
        parameters={"lb": (15, 20, 25), "vt": (0.08, 0.10, 0.12)},
    )
    trials: list[TrialRecord] = []
    for idx, cfg in enumerate(grid.all_configurations(), start=1):
        trials.append(
            TrialRecord(
                trial_id=f"t{idx}",
                sequence_number=idx,
                experiment_id="e",
                strategy_name="s",
                parameters=cfg,
                status=TrialStatus.COMPLETED,
                created_at_utc="2026-01-01T00:00:00Z",
                dataset_version="v1",
                code_version="0.1",
                cost_model=CostModel(),
                series_id=f"t{idx}",
            )
        )
    center = next(t for t in trials if t.parameters == {"lb": 20, "vt": 0.10})

    # Plateau: all neighbors remain strong
    plateau_sharpes = {t.trial_id: 1.45 for t in trials}
    plateau_sharpes[center.trial_id] = 1.60
    res_plateau = evaluate_parameter_perturbation(center, trials, plateau_sharpes, relative_step=0.25, parameter_grid=grid)
    assert res_plateau.n_neighbors_evaluated == 8
    assert res_plateau.positive_neighbor_fraction == 1.0
    assert res_plateau.sharpe_cliff_ratio < 0.15
    assert res_plateau.stability_score > 0.90

    # Isolated spike: center is high, neighbors collapse negative
    spike_sharpes = {t.trial_id: -0.35 for t in trials}
    spike_sharpes[center.trial_id] = 2.10
    res_spike = evaluate_parameter_perturbation(center, trials, spike_sharpes, relative_step=0.25, parameter_grid=grid)
    assert res_spike.positive_neighbor_fraction == 0.0
    assert res_spike.sharpe_cliff_ratio > 0.95
    assert res_spike.stability_score == 0.0
