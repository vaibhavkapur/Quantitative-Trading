"""Contract validation, hand-calculated reference oracle, and transaction cost tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robustness_lab.costs import apply_transaction_costs
from robustness_lab.io_adapters import load_policy_config, load_return_series_csv
from robustness_lab.models import (
    ContractValidationError,
    CostModel,
    ParameterGridSpec,
    ReturnFrequency,
    ReturnSeriesSpec,
    UnsupportedConfigurationError,
)
from robustness_lab.statistics import compute_sample_moments, compute_sharpe_ratio


def test_hand_calculated_reference_oracle(fixtures_dir: Path) -> None:
    ref = json.loads((fixtures_dir / "valid" / "hand_calculated_reference.json").read_text(encoding="utf-8"))
    tol = float(ref["tolerance"])
    exp = ref["expected"]

    timestamps = tuple(f"2024-01-{day:02d}" for day in range(1, len(ref["gross_returns"]) + 1))
    spec = ReturnSeriesSpec(
        series_id="oracle_ref",
        version="1.0",
        frequency=ReturnFrequency.DAILY,
        timestamps=timestamps,
        gross_returns=tuple(ref["gross_returns"]),
        turnover=tuple(ref["turnover"]),
    )
    cm = CostModel.from_dict(ref["cost_model"])
    adj = apply_transaction_costs(spec, cm)

    mean_g, std_g, skew_g, _ = compute_sample_moments(adj.gross_returns)
    sr_g_p, sr_g_a = compute_sharpe_ratio(adj.gross_returns, annualization_factor=ref["annualization_factor"])
    sr_n_p, _ = compute_sharpe_ratio(adj.net_returns, annualization_factor=ref["annualization_factor"])

    assert abs(mean_g - exp["gross_mean"]) < tol
    assert abs(std_g - exp["gross_sample_std"]) < tol
    assert abs(sr_g_p - exp["gross_sharpe_period"]) < tol
    assert abs(sr_g_a - exp["gross_sharpe_annual"]) < tol
    assert abs(float(np.mean(adj.cost_drag)) - exp["period_cost_drag"]) < tol
    assert abs(float(np.mean(adj.net_returns)) - exp["net_mean"]) < tol
    assert abs(sr_n_p - exp["net_sharpe_period"]) < tol
    assert abs(skew_g - exp["skewness"]) < tol


def test_invalid_fixtures_rejected(fixtures_dir: Path) -> None:
    inv = fixtures_dir / "invalid"
    with pytest.raises(ContractValidationError, match="strictly increasing"):
        load_return_series_csv(inv / "non_monotonic_timestamps.csv", "bad_ts", "v1")

    with pytest.raises(ContractValidationError, match="Non-finite"):
        load_return_series_csv(inv / "nan_returns.csv", "bad_nan", "v1")

    with pytest.raises(ContractValidationError, match="-1.0"):
        load_return_series_csv(inv / "below_minus_100pct.csv", "bad_ret", "v1")

    with pytest.raises(UnsupportedConfigurationError, match="Unsupported return frequency"):
        load_return_series_csv(inv / "nan_returns.csv", "bad_freq", "v1", frequency="tick")


def test_cost_model_and_grid_validation() -> None:
    with pytest.raises(ContractValidationError):
        CostModel(commission_bps=-1.0)
    with pytest.raises(ContractValidationError):
        CostModel(annualization_factor=0)
    with pytest.raises(ContractValidationError):
        ParameterGridSpec(grid_id="g", version="1", parameters={})
    with pytest.raises(ContractValidationError):
        ParameterGridSpec(grid_id="g", version="1", parameters={"a": ()})

    grid = ParameterGridSpec(grid_id="g1", version="1.0", parameters={"a": (1, 2), "b": (10, 20, 30)})
    assert grid.expected_trial_count == 6
    assert len(grid.all_configurations()) == 6


def test_missing_policy_config_raises(tmp_path: Path) -> None:
    with pytest.raises(ContractValidationError, match="not found"):
        load_policy_config(tmp_path / "does_not_exist.json")

    bad_cfg = tmp_path / "incomplete_policy.json"
    bad_cfg.write_text('{"policy_version": "1.0"}', encoding="utf-8")
    with pytest.raises(ContractValidationError, match="missing required section"):
        load_policy_config(bad_cfg)
