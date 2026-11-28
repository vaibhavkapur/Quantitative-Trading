"""Simulation tests: hand-calculated oracle, zero liquidity, partial fill, and price shocks."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

from optimal_execution.contracts import (
    CONSERVATION_TOLERANCE,
    CapacityPolicy,
    MarketProfile,
    OrderSide,
    PriceShock,
    ScheduleRequest,
    StrategyType,
)
from optimal_execution.io import load_scenario
from optimal_execution.schedules import (
    generate_schedule,
    twap_schedule,
    vwap_schedule,
)
from optimal_execution.simulator import (
    simulate_deterministic,
    simulate_monte_carlo,
    simulate_single_stochastic_path,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "data" / "fixtures"
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def test_hand_calculated_reference_oracle_twap_and_vwap() -> None:
    scenario = load_scenario(FIXTURES_DIR / "reference_deterministic.json")
    expected = json.loads((FIXTURES_DIR / "reference_expected.json").read_text(encoding="utf-8"))
    tol_usd = expected["tolerance_usd"]
    tol_bps = expected["tolerance_bps"]

    # TWAP exact verification
    twap_res = simulate_deterministic(scenario, twap_schedule(scenario))
    exp_twap = expected["twap"]
    m = twap_res.metrics
    assert abs(m.filled_quantity - exp_twap["filled_quantity"]) <= tol_usd
    assert abs(m.residual_quantity - exp_twap["residual_quantity"]) <= tol_usd
    assert abs(m.spread_cost_usd - exp_twap["spread_cost_usd"]) <= tol_usd
    assert abs(m.temporary_impact_cost_usd - exp_twap["temporary_impact_cost_usd"]) <= tol_usd
    assert abs(m.permanent_impact_cost_usd - exp_twap["permanent_impact_cost_usd"]) <= tol_usd
    assert abs(m.total_is_usd - exp_twap["total_is_usd"]) <= tol_usd
    assert abs(m.total_is_bps - exp_twap["total_is_bps"]) <= tol_bps
    assert abs(m.vwap_execution_price - exp_twap["vwap_execution_price"]) <= 1e-9
    assert abs(m.terminal_mid_price - exp_twap["terminal_mid_price"]) <= 1e-9
    assert abs(m.analytic_timing_risk_usd - exp_twap["analytic_timing_risk_usd"]) <= tol_usd
    assert abs(m.analytic_timing_risk_bps - exp_twap["analytic_timing_risk_bps"]) <= tol_bps

    # VWAP exact verification
    vwap_res = simulate_deterministic(scenario, vwap_schedule(scenario))
    exp_vwap = expected["vwap"]
    mv = vwap_res.metrics
    assert abs(mv.filled_quantity - exp_vwap["filled_quantity"]) <= tol_usd
    assert abs(mv.spread_cost_usd - exp_vwap["spread_cost_usd"]) <= tol_usd
    assert abs(mv.temporary_impact_cost_usd - exp_vwap["temporary_impact_cost_usd"]) <= tol_usd
    assert abs(mv.permanent_impact_cost_usd - exp_vwap["permanent_impact_cost_usd"]) <= tol_usd
    assert abs(mv.total_is_usd - exp_vwap["total_is_usd"]) <= tol_usd
    assert abs(mv.total_is_bps - exp_vwap["total_is_bps"]) <= tol_bps
    assert abs(mv.analytic_timing_risk_usd - exp_vwap["analytic_timing_risk_usd"]) <= tol_usd


def test_is_accounting_decomposition_exact_identity() -> None:
    scenario = load_scenario(FIXTURES_DIR / "price_shock.json")
    sched = twap_schedule(scenario)
    res = simulate_single_stochastic_path(scenario, sched, seed=77)
    m = res.metrics
    reconstructed = (
        m.spread_cost_usd
        + m.temporary_impact_cost_usd
        + m.permanent_impact_cost_usd
        + m.timing_drift_cost_usd
        + m.shock_cost_usd
    )
    assert math.isclose(m.execution_shortfall_usd, reconstructed, rel_tol=1e-12, abs_tol=1e-9)
    assert math.isclose(
        m.total_is_usd,
        m.execution_shortfall_usd + m.opportunity_cost_usd,
        rel_tol=1e-12,
        abs_tol=1e-9,
    )


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
def test_zero_liquidity_integration_scenario(strategy: StrategyType) -> None:
    scenario = load_scenario(FIXTURES_DIR / "zero_liquidity.json")
    req = ScheduleRequest(strategy=strategy, risk_aversion_lambda=1e-6, target_participation_rate=0.15)
    sched = generate_schedule(scenario, req)
    res = simulate_deterministic(scenario, sched)

    # Zero-liquidity slices (indices 1, 2, 4) must have zero fill and None execution price
    assert res.diagnostics.zero_liquidity_slices == 3
    for idx in (1, 2, 4):
        sl = res.slices[idx]
        assert sl.was_zero_liquidity is True
        assert sl.filled_quantity == 0.0
        assert sl.execution_price is None

    # No fill ever exceeds modeled capacity
    for sl in res.slices:
        assert sl.filled_quantity <= sl.capacity + 1e-9

    # Filled + residual == parent
    assert res.diagnostics.conservation_error <= CONSERVATION_TOLERANCE
    assert math.isclose(
        res.metrics.filled_quantity + res.metrics.residual_quantity,
        scenario.order.quantity,
        abs_tol=CONSERVATION_TOLERANCE,
    )


def test_all_zero_liquidity_total_halt() -> None:
    base = load_scenario(FIXTURES_DIR / "reference_deterministic.json")
    halt = replace(
        base,
        market=MarketProfile(
            volume_curve=(0.0, 0.0, 0.0, 0.0),
            daily_volatility=0.02,
            bid_ask_spread_bps=10.0,
            max_participation_rate=0.25,
        ),
    )
    for strategy in (StrategyType.TWAP, StrategyType.VWAP, StrategyType.PARTICIPATION, StrategyType.ALMGREN_CHRISS_ANALYTIC):
        sched = generate_schedule(halt, ScheduleRequest(strategy=strategy))
        res = simulate_deterministic(halt, sched)
        assert res.metrics.filled_quantity == 0.0
        assert math.isclose(res.metrics.residual_quantity, halt.order.quantity, abs_tol=1e-9)
        assert res.metrics.vwap_execution_price is None
        assert res.diagnostics.zero_liquidity_slices == 4
        assert res.diagnostics.partial_fill is True


def test_partial_fill_capacity_and_spillover_policy() -> None:
    scenario = load_scenario(FIXTURES_DIR / "partial_fill_capacity.json")
    # Total capacity = 0.20 * (10k + 5k + 5k + 10k) = 6,000 < 15,000 parent
    sched = twap_schedule(scenario)
    res_spill = simulate_deterministic(scenario, sched)

    assert math.isclose(res_spill.metrics.filled_quantity, 6000.0, abs_tol=1e-6)
    assert math.isclose(res_spill.metrics.residual_quantity, 9000.0, abs_tol=1e-6)
    assert res_spill.diagnostics.partial_fill is True
    for sl in res_spill.slices:
        assert sl.filled_quantity <= sl.capacity + 1e-9

    # Without spillover (CapacityPolicy.NONE), slices 0 and 3 plan 3750 each (cap 2000),
    # slices 1 and 2 plan 3750 each (cap 1000) -> still 6000 filled here because every slice is capped.
    # Test non-uniform case where spillover increases completion:
    non_uniform = replace(
        scenario,
        capacity_policy=CapacityPolicy.NONE,
        order=replace(scenario.order, quantity=5000.0),  # TWAP = 1250/slice; caps = [2000, 1000, 1000, 2000]
    )
    res_no_spill = simulate_deterministic(non_uniform, twap_schedule(non_uniform))
    res_with_spill = simulate_deterministic(
        replace(non_uniform, capacity_policy=CapacityPolicy.SPILLOVER),
        twap_schedule(non_uniform),
    )
    # Without spillover: slices 1 and 2 lose 250 each -> 4500 filled. With spillover: slice 3 absorbs 500 -> 5000 filled!
    assert math.isclose(res_no_spill.metrics.filled_quantity, 4500.0, abs_tol=1e-6)
    assert math.isclose(res_with_spill.metrics.filled_quantity, 5000.0, abs_tol=1e-6)
    assert res_with_spill.diagnostics.spillover_events >= 1


def test_price_shock_integration_scenario() -> None:
    shock_scenario = load_scenario(FIXTURES_DIR / "price_shock.json")
    no_shock = replace(
        shock_scenario,
        market=replace(shock_scenario.market, price_shocks=()),
    )

    sched = twap_schedule(shock_scenario)
    res_shock = simulate_deterministic(shock_scenario, sched)
    res_clean = simulate_deterministic(no_shock, sched)

    assert res_shock.diagnostics.shock_events_applied == 2
    # Slices 0, 1: 0 shock. Slices 2, 3: +$1.50/sh (4000 shares each -> +$12,000).
    # Slice 4: +$1.50 - $0.40 = +$1.10/sh (4000 shares -> +$4,400). Total shock cost = +$16,400!
    assert math.isclose(res_shock.metrics.shock_cost_usd, 16400.0, abs_tol=1e-6)
    assert math.isclose(
        res_shock.metrics.total_is_usd - res_clean.metrics.total_is_usd,
        16400.0,
        abs_tol=1e-6,
    )

    # For a SELL order, an upward price shock (+150 bps) is FAVORABLE (reduces shortfall)
    sell_shock = replace(shock_scenario, order=replace(shock_scenario.order, side=OrderSide.SELL))
    res_sell = simulate_deterministic(sell_shock, twap_schedule(sell_shock))
    assert math.isclose(res_sell.metrics.shock_cost_usd, -16400.0, abs_tol=1e-6)


def test_buy_sell_symmetry_zero_shocks() -> None:
    buy_sc = load_scenario(FIXTURES_DIR / "reference_deterministic.json")
    sell_sc = replace(buy_sc, order=replace(buy_sc.order, side=OrderSide.SELL))
    res_buy = simulate_deterministic(buy_sc, twap_schedule(buy_sc))
    res_sell = simulate_deterministic(sell_sc, twap_schedule(sell_sc))
    assert math.isclose(res_buy.metrics.total_is_usd, res_sell.metrics.total_is_usd, abs_tol=1e-9)
    assert math.isclose(res_buy.metrics.total_is_bps, res_sell.metrics.total_is_bps, abs_tol=1e-9)


def test_monte_carlo_matches_single_path_mechanics() -> None:
    scenario = load_scenario(FIXTURES_DIR / "partial_fill_capacity.json")
    sched = twap_schedule(scenario)
    mc, path_bps = simulate_monte_carlo(scenario, sched, num_paths=2000, seed=12345)
    assert len(path_bps) == 2000
    assert mc.analytic_expected_is_bps is not None
    assert mc.analytic_timing_risk_bps is not None
    # Monte Carlo empirical mean and std should be within ~3.5 standard errors of exact analytic values
    se_mean = mc.analytic_timing_risk_bps / math.sqrt(2000)
    assert abs(mc.mean_total_is_bps - mc.analytic_expected_is_bps) < 3.5 * se_mean
    assert abs(mc.std_total_is_bps - mc.analytic_timing_risk_bps) / mc.analytic_timing_risk_bps < 0.08
