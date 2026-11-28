"""Liquidity and urgency regime sensitivity analysis and efficient frontier sweeps."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from optimal_execution.contracts import (
    ExecutionScenario,
    ImpactParams,
    MarketProfile,
    ScheduleRequest,
    SensitivityPoint,
    StrategyType,
    ValidationError,
)
from optimal_execution.schedules import generate_schedule
from optimal_execution.simulator import simulate_deterministic, simulate_monte_carlo


@dataclass(frozen=True)
class LiquidityRegime:
    """Multiplicative market liquidity, spread, volatility, and impact regime."""

    name: str
    liquidity_multiplier: float = 1.0  # Scales market volume curve
    spread_multiplier: float = 1.0  # Scales quoted bid-ask spread
    volatility_multiplier: float = 1.0  # Scales daily volatility
    impact_multiplier: float = 1.0  # Scales permanent & temporary impact

    def apply(self, scenario: ExecutionScenario) -> ExecutionScenario:
        if self.liquidity_multiplier < 0.0:
            raise ValidationError("liquidity_multiplier must be >= 0")
        if self.spread_multiplier < 0.0 or self.volatility_multiplier < 0.0 or self.impact_multiplier < 0.0:
            raise ValidationError("regime multipliers must be >= 0")

        scaled_volumes = tuple(v * self.liquidity_multiplier for v in scenario.market.volume_curve)
        new_market = MarketProfile(
            volume_curve=scaled_volumes,
            daily_volatility=scenario.market.daily_volatility * self.volatility_multiplier,
            bid_ask_spread_bps=scenario.market.bid_ask_spread_bps * self.spread_multiplier,
            max_participation_rate=scenario.market.max_participation_rate,
            max_slice_quantity=scenario.market.max_slice_quantity,
            price_shocks=scenario.market.price_shocks,
        )
        new_impact = ImpactParams(
            permanent_impact_gamma=scenario.impact.permanent_impact_gamma * self.impact_multiplier,
            temporary_impact_eta=scenario.impact.temporary_impact_eta * self.impact_multiplier,
            temporary_impact_alpha=scenario.impact.temporary_impact_alpha,
        )
        new_scenario = replace(
            scenario,
            scenario_id=f"{scenario.scenario_id}__{self.name}",
            market=new_market,
            impact=new_impact,
        )
        new_scenario.validate()
        return new_scenario


DEFAULT_LIQUIDITY_REGIMES: Tuple[LiquidityRegime, ...] = (
    LiquidityRegime(
        name="stressed_illiquid",
        liquidity_multiplier=0.35,
        spread_multiplier=2.5,
        volatility_multiplier=1.75,
        impact_multiplier=2.0,
    ),
    LiquidityRegime(
        name="low_liquidity",
        liquidity_multiplier=0.60,
        spread_multiplier=1.5,
        volatility_multiplier=1.25,
        impact_multiplier=1.4,
    ),
    LiquidityRegime(
        name="baseline",
        liquidity_multiplier=1.0,
        spread_multiplier=1.0,
        volatility_multiplier=1.0,
        impact_multiplier=1.0,
    ),
    LiquidityRegime(
        name="deep_liquidity",
        liquidity_multiplier=1.8,
        spread_multiplier=0.7,
        volatility_multiplier=0.85,
        impact_multiplier=0.65,
    ),
)


def run_sensitivity_analysis(
    base_scenario: ExecutionScenario,
    regimes: Sequence[LiquidityRegime] = DEFAULT_LIQUIDITY_REGIMES,
    lambdas: Sequence[float] = (0.0, 1e-7, 1e-6, 1e-5, 1e-4),
    strategies: Sequence[StrategyType] = (
        StrategyType.TWAP,
        StrategyType.VWAP,
        StrategyType.PARTICIPATION,
        StrategyType.ALMGREN_CHRISS_ANALYTIC,
    ),
    target_participation_rate: float = 0.12,
    num_mc_paths: Optional[int] = None,
) -> List[SensitivityPoint]:
    """Evaluate execution schedules across liquidity and urgency (risk aversion) regimes."""
    base_scenario.validate()
    points: List[SensitivityPoint] = []

    for regime in regimes:
        reg_scenario = regime.apply(base_scenario)
        for strategy in strategies:
            strat_lambdas = (
                lambdas
                if strategy in (StrategyType.ALMGREN_CHRISS_ANALYTIC, StrategyType.ALMGREN_CHRISS_NUMERIC)
                else (0.0,)
            )
            for lam in strat_lambdas:
                req = ScheduleRequest(
                    strategy=strategy,
                    risk_aversion_lambda=float(lam),
                    target_participation_rate=target_participation_rate,
                )
                schedule = generate_schedule(reg_scenario, req)
                det = simulate_deterministic(reg_scenario, schedule)
                mc, _ = simulate_monte_carlo(
                    reg_scenario,
                    schedule,
                    num_paths=num_mc_paths,
                    seed=reg_scenario.seed,
                )
                points.append(
                    SensitivityPoint(
                        regime_name=regime.name,
                        liquidity_multiplier=regime.liquidity_multiplier,
                        spread_multiplier=regime.spread_multiplier,
                        volatility_multiplier=regime.volatility_multiplier,
                        risk_aversion_lambda=float(lam),
                        strategy=strategy.value,
                        completion_rate=det.metrics.completion_rate,
                        deterministic_is_bps=det.metrics.total_is_bps,
                        mc_mean_is_bps=mc.mean_total_is_bps,
                        mc_std_is_bps=mc.std_total_is_bps,
                        mc_p95_is_bps=mc.p95_total_is_bps,
                        mc_cvar_95_bps=mc.cvar_95_bps,
                        analytic_expected_is_bps=det.metrics.analytic_expected_is_bps,
                        analytic_timing_risk_bps=det.metrics.analytic_timing_risk_bps,
                        capacity_capped_slices=det.diagnostics.capacity_capped_slices,
                    )
                )

    return points


def compute_efficient_frontier(
    scenario: ExecutionScenario,
    lambdas: Sequence[float] = (0.0, 1e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4),
    num_mc_paths: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Trace the Almgren-Chriss expected cost vs timing risk efficient frontier."""
    scenario.validate()
    frontier: List[Dict[str, Any]] = []
    for lam in lambdas:
        req = ScheduleRequest(
            strategy=StrategyType.ALMGREN_CHRISS_ANALYTIC,
            risk_aversion_lambda=float(lam),
        )
        schedule = generate_schedule(scenario, req)
        det = simulate_deterministic(scenario, schedule)
        mc, _ = simulate_monte_carlo(
            scenario,
            schedule,
            num_paths=num_mc_paths,
            seed=scenario.seed,
        )
        frontier.append(
            {
                "risk_aversion_lambda": float(lam),
                "completion_rate": det.metrics.completion_rate,
                "analytic_expected_is_bps": det.metrics.analytic_expected_is_bps,
                "analytic_timing_risk_bps": det.metrics.analytic_timing_risk_bps,
                "mc_mean_is_bps": mc.mean_total_is_bps,
                "mc_std_is_bps": mc.std_total_is_bps,
                "mc_var_95_bps": mc.var_95_bps,
                "mc_cvar_95_bps": mc.cvar_95_bps,
                "front_load_first_slice_fraction": (
                    schedule.planned_quantities[0] / scenario.order.quantity
                ),
                "capacity_capped_slices": det.diagnostics.capacity_capped_slices,
            }
        )
    return frontier
