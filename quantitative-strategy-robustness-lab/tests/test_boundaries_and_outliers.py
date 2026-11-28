"""Boundary and failure-mode tests: insufficient sample, extreme outliers, degenerate inputs."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from robustness_lab.bootstrap import run_block_bootstrap
from robustness_lab.models import (
    ContractValidationError,
    InsufficientSampleError,
    PromotionDecision,
    UnsupportedConfigurationError,
)
from robustness_lab.pbo import compute_pbo_cscv
from robustness_lab.pipeline import evaluate_from_directory
from robustness_lab.statistics import compute_dsr, compute_sharpe_ratio, diagnose_outliers
from robustness_lab.walk_forward import run_walk_forward


def test_insufficient_sample_errors_explicitly_raised() -> None:
    rng = np.random.default_rng(901)
    tiny = rng.normal(0.0, 0.01, size=20)
    tiny_mat = rng.normal(0.0, 0.01, size=(20, 4))

    with pytest.raises(InsufficientSampleError):
        compute_dsr("t1", tiny, tiny_mat, min_observations=60)

    with pytest.raises(InsufficientSampleError):
        compute_pbo_cscv(tiny_mat, n_partitions=8, min_block_observations=5)

    with pytest.raises(InsufficientSampleError):
        compute_pbo_cscv(rng.normal(0.0, 0.01, size=(100, 1)), n_partitions=8)

    with pytest.raises(UnsupportedConfigurationError):
        compute_pbo_cscv(rng.normal(0.0, 0.01, size=(100, 4)), n_partitions=7)  # odd S unsupported

    with pytest.raises(InsufficientSampleError):
        run_block_bootstrap(tiny, min_observations=60)

    ts = tuple(f"2024-01-{i+1:02d}" for i in range(20))
    with pytest.raises(InsufficientSampleError):
        run_walk_forward(ts, ["a", "b", "c", "d"], tiny_mat, train_window=15, test_window=10, purge_gap=2)


def test_extreme_outliers_diagnosed_and_rejected_in_pipeline(
    fixtures_dir: Path, default_policy_path: Path
) -> None:
    rng = np.random.default_rng(902)
    r = rng.normal(loc=-0.0001, scale=0.005, size=300)
    r[50] = 0.25
    r[200] = 0.28

    diag = diagnose_outliers(r, annualization_factor=252)
    assert diag.n_extreme_observations >= 2
    assert diag.excess_kurtosis > 50.0
    assert diag.raw_sharpe_annual > 0.5
    assert diag.winsorized_sharpe_annual < 0.2
    assert diag.outlier_impact_pct > 0.5
    assert len(diag.warnings) >= 2

    # End-to-end fixture evaluation must REJECT the outlier-driven candidate
    payload = evaluate_from_directory(
        experiment_dir=fixtures_dir / "boundary" / "extreme_outlier_experiment",
        policy_path=default_policy_path,
        candidate_trial_id="outlier_t3",
    )
    assert payload["verdict"]["decision"] == PromotionDecision.REJECT.value
    gate_map = {g["gate_name"]: g for g in payload["verdict"]["gates"]}
    assert gate_map["outlier_dominance_check"]["passed"] is False


def test_degenerate_constant_returns_handled_explicitly() -> None:
    zeros = np.zeros(50)
    assert compute_sharpe_ratio(zeros) == (0.0, 0.0)

    const_pos = np.full(50, 0.005)
    with pytest.raises(ContractValidationError, match="zero volatility"):
        compute_sharpe_ratio(const_pos)
