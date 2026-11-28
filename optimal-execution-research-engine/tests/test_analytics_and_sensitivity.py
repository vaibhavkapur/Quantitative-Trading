"""Tests for Monte Carlo cost distributions, efficient frontier, sensitivity, and Polars outputs."""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from optimal_execution.analytics import (
    sensitivity_to_polars,
    slices_to_polars,
    strategy_comparison_to_polars,
)
from optimal_execution.contracts import ScheduleRequest, StrategyType
from optimal_execution.io import load_scenario
from optimal_execution.schedules import generate_schedule
from optimal_execution.sensitivity import (
    DEFAULT_LIQUIDITY_REGIMES,
    compute_efficient_frontier,
    run_sensitivity_analysis,
)
from optimal_execution.simulator import simulate_deterministic, simulate_monte_carlo

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def test_monte_carlo_convergence_to_analytic_moments_and_tail_risk_ordering() -> None:
    scenario = load_scenario(CONFIG_DIR / "default_scenario.json")
    sched = generate_schedule(
        scenario,
        ScheduleRequest(strategy=StrategyType.ALMGREN_CHRISS_ANALYTIC, risk_aversion_lambda=1e-6),
    )
    mc, _ = simulate_monte_carlo(scenario, sched, num_paths=4000, seed=99)

    assert mc.analytic_expected_is_bps is not None
    assert mc.analytic_timing_risk_bps is not None
    se_mean = mc.analytic_timing_risk_bps / math.sqrt(4000)
    assert abs(mc.mean_total_is_bps - mc.analytic_expected_is_bps) < 3.5 * se_mean
    assert abs(mc.std_total_is_bps - mc.analytic_timing_risk_bps) / mc.analytic_timing_risk_bps < 0.06

    # Tail risk ordering: p05 < p50 < p95 == var_95 <= cvar_95
    assert mc.p05_total_is_bps < mc.p50_total_is_bps < mc.p95_total_is_bps
    assert math.isclose(mc.p95_total_is_bps, mc.var_95_bps, rel_tol=1e-9)
    assert mc.cvar_95_bps >= mc.var_95_bps


def test_efficient_frontier_tradeoff_monotonicity() -> None:
    scenario = load_scenario(CONFIG_DIR / "default_scenario.json")
    lambdas = [0.0, 1e-7, 1e-6, 1e-5, 1e-4]
    frontier = compute_efficient_frontier(scenario, lambdas=lambdas, num_mc_paths=500)
    assert len(frontier) == len(lambdas)

    # Higher urgency lambda increases front-loading, reduces analytic timing risk,
    # and increases deterministic expected cost (for unconstrained fills).
    exp_costs = [pt["analytic_expected_is_bps"] for pt in frontier]
    timing_risks = [pt["analytic_timing_risk_bps"] for pt in frontier]
    for i in range(len(frontier) - 1):
        assert timing_risks[i] >= timing_risks[i + 1] - 1e-9
        assert exp_costs[i] <= exp_costs[i + 1] + 1e-9


def test_sensitivity_analysis_liquidity_regimes_and_polars_outputs() -> None:
    scenario = load_scenario(CONFIG_DIR / "default_scenario.json")
    points = run_sensitivity_analysis(
        base_scenario=scenario,
        regimes=DEFAULT_LIQUIDITY_REGIMES,
        lambdas=(0.0, 1e-6, 1e-4),
        strategies=(StrategyType.TWAP, StrategyType.VWAP, StrategyType.ALMGREN_CHRISS_ANALYTIC),
        num_mc_paths=300,
    )
    df = sensitivity_to_polars(points)
    assert isinstance(df, pl.DataFrame)
    assert df.height == len(points)
    assert {"regime_name", "strategy", "deterministic_is_bps", "mc_cvar_95_bps", "completion_rate"}.issubset(
        set(df.columns)
    )

    # Stressed illiquid TWAP must have strictly higher deterministic IS than deep liquidity TWAP
    stressed_twap = df.filter((pl.col("regime_name") == "stressed_illiquid") & (pl.col("strategy") == "twap"))
    deep_twap = df.filter((pl.col("regime_name") == "deep_liquidity") & (pl.col("strategy") == "twap"))
    assert stressed_twap["deterministic_is_bps"][0] > deep_twap["deterministic_is_bps"][0]


def test_polars_slice_and_comparison_tables() -> None:
    scenario = load_scenario(CONFIG_DIR / "default_scenario.json")
    det_results = []
    mc_summaries = []
    for strat in (StrategyType.TWAP, StrategyType.VWAP, StrategyType.ALMGREN_CHRISS_ANALYTIC):
        sched = generate_schedule(scenario, ScheduleRequest(strategy=strat, risk_aversion_lambda=1e-6))
        det_results.append(simulate_deterministic(scenario, sched))
        mc, _ = simulate_monte_carlo(scenario, sched, num_paths=200)
        mc_summaries.append(mc)

    slices_df = slices_to_polars(det_results[0])
    assert slices_df.height == scenario.order.horizon_slices
    assert "filled_quantity" in slices_df.columns

    cmp_df = strategy_comparison_to_polars(det_results, mc_summaries)
    assert cmp_df.height == 3
    assert {"strategy", "completion_rate", "det_total_is_bps", "mc_mean_is_bps", "mc_cvar_95_bps"}.issubset(
        set(cmp_df.columns)
    )
