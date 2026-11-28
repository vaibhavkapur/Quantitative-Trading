"""Deterministic unit tests for execution schedules: conservation and time constraints."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import numpy as np
import pytest

from optimal_execution.contracts import (
    CONSERVATION_TOLERANCE,
    ImpactParams,
    MarketProfile,
    ScheduleError,
    ScheduleRequest,
    StrategyType,
    UnsupportedScenarioError,
)
from optimal_execution.io import load_scenario
from optimal_execution.schedules import (
    almgren_chriss_analytic_schedule,
    almgren_chriss_numeric_schedule,
    generate_schedule,
    participation_schedule,
    twap_schedule,
    vwap_schedule,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "data" / "fixtures"
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


@pytest.mark.parametrize(
    "strategy",
    [
        StrategyType.TWAP,
        StrategyType.VWAP,
        StrategyType.PARTICIPATION,
        StrategyType.ALMGREN_CHRISS_ANALYTIC,
        StrategyType.ALMGREN_CHRISS_NUMERIC,
    ],
)
def test_order_conservation_and_time_constraints_all_strategies(strategy: StrategyType) -> None:
    scenario = load_scenario(CONFIG_DIR / "default_scenario.json")
    req = ScheduleRequest(
        strategy=strategy,
        risk_aversion_lambda=1e-6,
        target_participation_rate=0.12,
    )
    sched = generate_schedule(scenario, req)

    # Time constraints: exact slice count and trajectory length N + 1
    n = scenario.order.horizon_slices
    assert len(sched.planned_quantities) == n
    assert len(sched.remaining_trajectory) == n + 1

    # Non-negativity & monotonically non-increasing remaining inventory
    for qty in sched.planned_quantities:
        assert math.isfinite(qty)
        assert qty >= 0.0
    for idx in range(n):
        assert sched.remaining_trajectory[idx] >= sched.remaining_trajectory[idx + 1] - 1e-9
    assert math.isclose(sched.remaining_trajectory[0], scenario.order.quantity, rel_tol=0.0, abs_tol=1e-9)

    # Conservation: planned + planned_residual == parent
    total = sum(sched.planned_quantities) + sched.planned_residual
    assert abs(total - scenario.order.quantity) <= CONSERVATION_TOLERANCE


def test_twap_uniform_slices() -> None:
    scenario = load_scenario(FIXTURES_DIR / "reference_deterministic.json")
    sched = twap_schedule(scenario)
    assert np.allclose(sched.planned_quantities, [2500.0, 2500.0, 2500.0, 2500.0])
    assert sched.planned_residual == 0.0


def test_vwap_matches_volume_weights() -> None:
    scenario = load_scenario(FIXTURES_DIR / "reference_deterministic.json")
    sched = vwap_schedule(scenario)
    expected = [10000.0 / 3.0, 10000.0 / 6.0, 10000.0 / 6.0, 10000.0 / 3.0]
    assert np.allclose(sched.planned_quantities, expected, rtol=1e-12)
    assert sched.planned_residual == 0.0


def test_participation_respects_target_rate_and_leaves_residual_when_under_volume() -> None:
    scenario = load_scenario(FIXTURES_DIR / "reference_deterministic.json")
    # Total volume = 120,000; 5% POV can schedule at most 6,000 shares < 10,000 parent
    sched = participation_schedule(scenario, target_participation_rate=0.05)
    assert np.allclose(sched.planned_quantities, [2000.0, 1000.0, 1000.0, 2000.0])
    assert math.isclose(sched.planned_residual, 4000.0, abs_tol=1e-9)
    assert math.isclose(sum(sched.planned_quantities) + sched.planned_residual, 10000.0, abs_tol=1e-9)


def test_almgren_chriss_risk_neutral_limit_equals_twap() -> None:
    scenario = load_scenario(CONFIG_DIR / "default_scenario.json")
    twap = twap_schedule(scenario)
    ac_zero = almgren_chriss_analytic_schedule(scenario, risk_aversion_lambda=0.0)
    assert np.allclose(ac_zero.planned_quantities, twap.planned_quantities, rtol=1e-12)


def test_almgren_chriss_front_loading_increases_with_risk_aversion() -> None:
    scenario = load_scenario(CONFIG_DIR / "default_scenario.json")
    low = almgren_chriss_analytic_schedule(scenario, risk_aversion_lambda=1e-7)
    mid = almgren_chriss_analytic_schedule(scenario, risk_aversion_lambda=1e-6)
    high = almgren_chriss_analytic_schedule(scenario, risk_aversion_lambda=1e-4)

    assert low.planned_quantities[0] < mid.planned_quantities[0] < high.planned_quantities[0]
    # Strictly decreasing slices for positive risk aversion
    for sched in (low, mid, high):
        diffs = np.diff(sched.planned_quantities)
        assert np.all(diffs < 0.0)


def test_almgren_chriss_extreme_lambda_no_overflow() -> None:
    scenario = load_scenario(CONFIG_DIR / "default_scenario.json")
    extreme = almgren_chriss_analytic_schedule(scenario, risk_aversion_lambda=1e3)
    assert all(math.isfinite(x) for x in extreme.planned_quantities)
    assert abs(sum(extreme.planned_quantities) - scenario.order.quantity) <= CONSERVATION_TOLERANCE
    # Nearly entire order front-loaded into slice 0
    assert extreme.planned_quantities[0] > 0.99 * scenario.order.quantity


def test_almgren_chriss_rejects_non_convex_eta_tilde() -> None:
    scenario = load_scenario(FIXTURES_DIR / "reference_deterministic.json")
    # tau = 0.25, gamma = 1e-4 -> 0.5*gamma*tau = 1.25e-5 > eta = 1e-6 -> eta_tilde < 0
    bad = replace(
        scenario,
        impact=ImpactParams(
            permanent_impact_gamma=1e-4,
            temporary_impact_eta=1e-6,
            temporary_impact_alpha=1.0,
        ),
    )
    with pytest.raises(ScheduleError, match="convexity condition violated"):
        almgren_chriss_analytic_schedule(bad, risk_aversion_lambda=1e-6)
    with pytest.raises(ScheduleError, match="convexity condition violated"):
        almgren_chriss_numeric_schedule(bad, risk_aversion_lambda=1e-6)


def test_almgren_chriss_analytic_rejects_nonlinear_alpha_while_numeric_succeeds() -> None:
    scenario = load_scenario(FIXTURES_DIR / "reference_deterministic.json")
    sqrt_impact = replace(
        scenario,
        impact=ImpactParams(
            permanent_impact_gamma=2.0e-6,
            temporary_impact_eta=5.0e-4,
            temporary_impact_alpha=0.5,
        ),
    )
    with pytest.raises(UnsupportedScenarioError, match="requires linear temporary impact"):
        almgren_chriss_analytic_schedule(sqrt_impact, risk_aversion_lambda=1e-6)

    numeric = almgren_chriss_numeric_schedule(sqrt_impact, risk_aversion_lambda=1e-6)
    assert abs(sum(numeric.planned_quantities) - sqrt_impact.order.quantity) <= CONSERVATION_TOLERANCE


def test_all_zero_liquidity_vwap_fallback_conserves_order() -> None:
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
    sched = vwap_schedule(dry)
    assert all(x == 0.0 for x in sched.planned_quantities)
    assert math.isclose(sched.planned_residual, dry.order.quantity, abs_tol=1e-9)
    assert sched.parameters["zero_liquidity_fallback"] is True
