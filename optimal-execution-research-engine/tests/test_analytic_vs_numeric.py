"""Boundary and regression tests comparing analytic vs numeric Almgren-Chriss schedules."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from optimal_execution.contracts import MarketProfile
from optimal_execution.io import load_scenario
from optimal_execution.schedules import (
    almgren_chriss_analytic_schedule,
    almgren_chriss_numeric_schedule,
    compare_analytic_and_numeric_schedules,
)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "data" / "fixtures"


@pytest.mark.parametrize("lam", [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4])
def test_analytic_and_numeric_agree_unconstrained(lam: float) -> None:
    scenario = load_scenario(CONFIG_DIR / "default_scenario.json")
    cmp = compare_analytic_and_numeric_schedules(
        scenario,
        risk_aversion_lambda=lam,
        enforce_capacity_on_numeric=False,
        rel_l2_tolerance=1e-4,
    )
    assert cmp["within_tolerance"] is True, (
        f"lambda={lam}: relative_l2_diff={cmp['relative_l2_diff']:.3e} exceeded tolerance"
    )
    assert cmp["relative_objective_diff"] < 1e-6


def test_analytic_and_numeric_agree_on_4slice_reference() -> None:
    scenario = load_scenario(FIXTURES_DIR / "reference_deterministic.json")
    cmp = compare_analytic_and_numeric_schedules(
        scenario,
        risk_aversion_lambda=5e-6,
        enforce_capacity_on_numeric=False,
        rel_l2_tolerance=1e-5,
    )
    assert cmp["within_tolerance"] is True
    assert cmp["max_abs_trade_diff_shares"] < 0.1


def test_constrained_numeric_respects_binding_capacity_where_analytic_exceeds() -> None:
    scenario = load_scenario(CONFIG_DIR / "default_scenario.json")
    # High urgency makes unconstrained analytic front-load heavily into slice 0,
    # exceeding slice 0 capacity (0.25 * 180,000 = 45,000).
    lam = 5e-4
    analytic = almgren_chriss_analytic_schedule(scenario, risk_aversion_lambda=lam)
    numeric_cap = almgren_chriss_numeric_schedule(
        scenario,
        risk_aversion_lambda=lam,
        enforce_capacity=True,
    )
    capacities = scenario.market.slice_capacities()

    # Analytic exceeds slice 0 capacity
    assert analytic.planned_quantities[0] > capacities[0]
    # Capacity-constrained numeric strictly respects every slice capacity
    for planned, cap in zip(numeric_cap.planned_quantities, capacities):
        assert planned <= cap + 1e-6
    # Full execution still achieved because total capacity > parent quantity
    assert np.isclose(sum(numeric_cap.planned_quantities), scenario.order.quantity, atol=1e-4)

    # Compare helper flags the expected divergence when capacity binds
    cmp = compare_analytic_and_numeric_schedules(
        scenario,
        risk_aversion_lambda=lam,
        enforce_capacity_on_numeric=True,
        rel_l2_tolerance=1e-4,
    )
    assert cmp["within_tolerance"] is False
    assert cmp["numeric_objective_usd"] >= cmp["analytic_objective_usd"] - 1e-6


def test_numeric_optimizer_handles_infeasible_capacity_gracefully() -> None:
    scenario = load_scenario(FIXTURES_DIR / "partial_fill_capacity.json")
    numeric_cap = almgren_chriss_numeric_schedule(
        scenario,
        risk_aversion_lambda=1e-6,
        enforce_capacity=True,
    )
    capacities = scenario.market.slice_capacities()
    assert numeric_cap.parameters["capacity_infeasible_full_execution"] is True
    assert np.allclose(numeric_cap.planned_quantities, capacities)
    assert np.isclose(
        sum(numeric_cap.planned_quantities) + numeric_cap.planned_residual,
        scenario.order.quantity,
    )


def test_numeric_optimizer_handles_all_zero_capacity() -> None:
    scenario = load_scenario(FIXTURES_DIR / "reference_deterministic.json")
    dry = replace(
        scenario,
        market=MarketProfile(
            volume_curve=(0.0, 0.0, 0.0, 0.0),
            daily_volatility=0.02,
            bid_ask_spread_bps=10.0,
            max_participation_rate=0.25,
        ),
    )
    sched = almgren_chriss_numeric_schedule(dry, risk_aversion_lambda=1e-6, enforce_capacity=True)
    assert all(x == 0.0 for x in sched.planned_quantities)
    assert np.isclose(sched.planned_residual, dry.order.quantity)
