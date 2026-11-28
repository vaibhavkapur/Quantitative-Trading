"""Implementation shortfall analytics, timing risk, and Polars summary builders."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from optimal_execution.contracts import (
    CostMetrics,
    ExecutionScenario,
    FillSlice,
    MonteCarloSummary,
    Schedule,
    SensitivityPoint,
    SimulationResult,
)


def analytic_expected_cost_and_timing_risk(
    scenario: ExecutionScenario,
    executed_quantities: Sequence[float],
) -> Tuple[Optional[float], Optional[float]]:
    """Compute exact closed-form E[IS] and timing risk sqrt(Var[IS]) for a deterministic fill trajectory.

    Returns `(expected_is_usd, timing_risk_usd)`.
    Includes exact opportunity-cost variance contribution if there is an unfilled residual R = Q - sum(f_k):
      - Execution shortfall diffusion term: s * sum_{k=0}^{N-1} f_k * sum_{j=0}^{k-1} dW_j
      - Opportunity cost diffusion term:    s * R * sum_{j=0}^{N-1} dW_j
      - Combined exposure to dW_j (for j = 0 .. N-1):
          w_j = (sum_{k=j+1}^{N-1} f_k) + R = Q - sum_{k=0}^{j} f_k = x_{j+1} !
      Notice that for j = 0 .. N-1, x_{j+1} is the remaining inventory after slice j
      (including residual R for j = N-1 where x_N = R).
      So Var[Total IS] = sigma_$^2 * tau * sum_{j=0}^{N-1} x_{j+1}^2 holds EXACTLY even
      when there is an unfilled residual (when there are no price shocks or when shocks are
      deterministic constants that shift the mean only)!
    """
    fills = np.asarray(executed_quantities, dtype=np.float64)
    n = scenario.order.horizon_slices
    if len(fills) != n:
        return None, None

    q = float(scenario.order.quantity)
    s0 = float(scenario.order.arrival_price)
    side_sign = scenario.order.side.sign
    tau = scenario.order.slice_duration_days
    sigma_usd = s0 * float(scenario.market.daily_volatility)
    eps = scenario.market.half_spread_usd(s0)
    gamma = float(scenario.impact.permanent_impact_gamma)
    eta = float(scenario.impact.temporary_impact_eta)
    alpha = float(scenario.impact.temporary_impact_alpha)

    total_filled = float(np.sum(fills))
    residual = max(0.0, q - total_filled)

    # Deterministic spread + temporary impact + permanent impact on executed fills
    spread_cost = eps * total_filled
    inv_tau_alpha = 1.0 / (tau ** alpha)
    temp_cost = eta * inv_tau_alpha * float(np.sum(np.power(np.maximum(fills, 0.0), 1.0 + alpha)))
    perm_cost_exec = 0.5 * gamma * (total_filled * total_filled - float(np.sum(fills * fills)))

    # Permanent impact embedded in terminal mid-price affects opportunity cost of residual:
    # S_N - S_0 includes s * gamma * total_filled + total_shocks, so opportunity cost mean has
    # R * (gamma * total_filled) + s * R * total_shocks.
    perm_cost_opp = residual * gamma * total_filled

    # Deterministic exogenous price shocks contribution to expected IS:
    shock_per_slice = np.zeros(n, dtype=np.float64)
    for shock in scenario.market.price_shocks:
        shock_per_slice[shock.slice_index] += (shock.shock_bps * 1e-4) * s0
    cum_shocks_before = np.cumsum(shock_per_slice)  # shock at slice k applies before fill k
    total_shock = float(np.sum(shock_per_slice))
    shock_cost_exec = side_sign * float(np.sum(fills * cum_shocks_before))
    shock_cost_opp = side_sign * residual * total_shock

    expected_is_usd = (
        spread_cost
        + temp_cost
        + perm_cost_exec
        + perm_cost_opp
        + shock_cost_exec
        + shock_cost_opp
    )

    # Exact variance of Total IS across Brownian increments dW_0 .. dW_{N-1}:
    # Exposure to dW_j is x_{j+1} = Q - sum_{k=0}^{j} fills[k]
    x_post = q - np.cumsum(fills)
    variance_usd2 = (sigma_usd ** 2) * tau * float(np.sum(x_post * x_post))
    timing_risk_usd = math.sqrt(max(0.0, variance_usd2))
    return float(expected_is_usd), float(timing_risk_usd)


def compute_cost_metrics(
    scenario: ExecutionScenario,
    slices: Sequence[FillSlice],
    terminal_mid_price: float,
) -> CostMetrics:
    """Compute exact implementation shortfall decomposition and analytic timing risk."""
    q = float(scenario.order.quantity)
    s0 = float(scenario.order.arrival_price)
    side_sign = scenario.order.side.sign
    arrival_notional = scenario.order.arrival_notional

    filled_qty = sum(sl.filled_quantity for sl in slices)
    residual_qty = max(0.0, q - filled_qty)
    completion_rate = filled_qty / q if q > 0.0 else 0.0

    if filled_qty > 0.0:
        executed_notional = sum(
            sl.filled_quantity * sl.execution_price
            for sl in slices
            if sl.execution_price is not None and sl.filled_quantity > 0.0
        )
        vwap_px: Optional[float] = executed_notional / filled_qty
    else:
        vwap_px = None

    spread_cost = sum(sl.spread_cost_usd for sl in slices)
    temp_cost = sum(sl.temporary_impact_cost_usd for sl in slices)
    perm_cost = sum(sl.permanent_impact_cost_usd for sl in slices)
    timing_cost = sum(sl.timing_drift_cost_usd for sl in slices)
    shock_cost = sum(sl.shock_cost_usd for sl in slices)
    exec_shortfall = sum(sl.total_slice_shortfall_usd for sl in slices)

    opp_cost = side_sign * residual_qty * (terminal_mid_price - s0)
    total_is = exec_shortfall + opp_cost

    bps_scale = 1e4 / arrival_notional
    exec_shortfall_bps = exec_shortfall * bps_scale
    opp_cost_bps = opp_cost * bps_scale
    total_is_bps = total_is * bps_scale

    analytic_exp, analytic_risk = analytic_expected_cost_and_timing_risk(
        scenario, [sl.filled_quantity for sl in slices]
    )
    analytic_exp_bps = analytic_exp * bps_scale if analytic_exp is not None else None
    analytic_risk_bps = analytic_risk * bps_scale if analytic_risk is not None else None

    return CostMetrics(
        arrival_notional_usd=arrival_notional,
        filled_quantity=filled_qty,
        residual_quantity=residual_qty,
        completion_rate=completion_rate,
        vwap_execution_price=vwap_px,
        terminal_mid_price=terminal_mid_price,
        spread_cost_usd=spread_cost,
        temporary_impact_cost_usd=temp_cost,
        permanent_impact_cost_usd=perm_cost,
        timing_drift_cost_usd=timing_cost,
        shock_cost_usd=shock_cost,
        execution_shortfall_usd=exec_shortfall,
        opportunity_cost_usd=opp_cost,
        total_is_usd=total_is,
        execution_shortfall_bps=exec_shortfall_bps,
        opportunity_cost_bps=opp_cost_bps,
        total_is_bps=total_is_bps,
        analytic_expected_is_usd=analytic_exp,
        analytic_timing_risk_usd=analytic_risk,
        analytic_expected_is_bps=analytic_exp_bps,
        analytic_timing_risk_bps=analytic_risk_bps,
    )


def summarize_monte_carlo(
    scenario: ExecutionScenario,
    strategy_name,
    seed: int,
    total_is_usd: np.ndarray,
    exec_shortfall_usd: np.ndarray,
    opportunity_cost_usd: np.ndarray,
    completion_rates: np.ndarray,
    reference_fills: Sequence[float],
) -> MonteCarloSummary:
    """Compute distributional statistics (mean, std, quantiles, VaR 95%, CVaR 95%) from MC paths."""
    arrival_notional = scenario.order.arrival_notional
    bps_scale = 1e4 / arrival_notional

    is_usd = np.asarray(total_is_usd, dtype=np.float64)
    is_bps = is_usd * bps_scale
    exec_bps = np.asarray(exec_shortfall_usd, dtype=np.float64) * bps_scale
    opp_bps = np.asarray(opportunity_cost_usd, dtype=np.float64) * bps_scale
    comp = np.asarray(completion_rates, dtype=np.float64)

    mean_usd = float(np.mean(is_usd))
    std_usd = float(np.std(is_usd, ddof=1)) if len(is_usd) > 1 else 0.0
    mean_bps = float(np.mean(is_bps))
    std_bps = float(np.std(is_bps, ddof=1)) if len(is_bps) > 1 else 0.0

    p05_bps = float(np.percentile(is_bps, 5))
    p50_bps = float(np.percentile(is_bps, 50))
    p95_bps = float(np.percentile(is_bps, 95))

    var_95_usd = float(np.percentile(is_usd, 95))
    tail_usd = is_usd[is_usd >= var_95_usd]
    cvar_95_usd = float(np.mean(tail_usd)) if len(tail_usd) > 0 else var_95_usd

    var_95_bps = var_95_usd * bps_scale
    cvar_95_bps = cvar_95_usd * bps_scale

    analytic_exp, analytic_risk = analytic_expected_cost_and_timing_risk(scenario, reference_fills)
    analytic_exp_bps = analytic_exp * bps_scale if analytic_exp is not None else None
    analytic_risk_bps = analytic_risk * bps_scale if analytic_risk is not None else None

    return MonteCarloSummary(
        scenario_id=scenario.scenario_id,
        strategy=strategy_name,
        seed=seed,
        num_paths=int(len(is_usd)),
        mean_total_is_usd=mean_usd,
        std_total_is_usd=std_usd,
        mean_total_is_bps=mean_bps,
        std_total_is_bps=std_bps,
        p05_total_is_bps=p05_bps,
        p50_total_is_bps=p50_bps,
        p95_total_is_bps=p95_bps,
        var_95_usd=var_95_usd,
        cvar_95_usd=cvar_95_usd,
        var_95_bps=var_95_bps,
        cvar_95_bps=cvar_95_bps,
        analytic_expected_is_usd=analytic_exp,
        analytic_timing_risk_usd=analytic_risk,
        analytic_expected_is_bps=analytic_exp_bps,
        analytic_timing_risk_bps=analytic_risk_bps,
        mean_completion_rate=float(np.mean(comp)),
        mean_execution_shortfall_bps=float(np.mean(exec_bps)),
        mean_opportunity_cost_bps=float(np.mean(opp_bps)),
    )


def slices_to_polars(result: SimulationResult) -> pl.DataFrame:
    """Convert per-slice simulation fills into a typed Polars DataFrame."""
    rows: List[Dict[str, object]] = []
    for sl in result.slices:
        rows.append(
            {
                "slice_index": sl.slice_index,
                "market_volume": sl.market_volume,
                "capacity": sl.capacity,
                "planned_quantity": sl.planned_quantity,
                "target_quantity": sl.target_quantity,
                "filled_quantity": sl.filled_quantity,
                "unfilled_quantity": sl.unfilled_quantity,
                "remaining_before": sl.remaining_before,
                "remaining_after": sl.remaining_after,
                "mid_price_before": sl.mid_price_before,
                "mid_price_after": sl.mid_price_after,
                "execution_price": sl.execution_price,
                "participation_rate": sl.participation_rate,
                "spread_cost_usd": sl.spread_cost_usd,
                "temporary_impact_cost_usd": sl.temporary_impact_cost_usd,
                "permanent_impact_cost_usd": sl.permanent_impact_cost_usd,
                "timing_drift_cost_usd": sl.timing_drift_cost_usd,
                "shock_cost_usd": sl.shock_cost_usd,
                "total_slice_shortfall_usd": sl.total_slice_shortfall_usd,
                "was_capacity_capped": sl.was_capacity_capped,
                "was_zero_liquidity": sl.was_zero_liquidity,
            }
        )
    return pl.DataFrame(rows)


def strategy_comparison_to_polars(
    det_results: Sequence[SimulationResult],
    mc_summaries: Optional[Sequence[MonteCarloSummary]] = None,
) -> pl.DataFrame:
    """Build a Polars comparison table across strategies."""
    mc_by_strat = {mc.strategy.value: mc for mc in (mc_summaries or ())}
    rows: List[Dict[str, object]] = []
    for res in det_results:
        m = res.metrics
        d = res.diagnostics
        mc = mc_by_strat.get(res.strategy.value)
        row: Dict[str, object] = {
            "scenario_id": res.scenario_id,
            "strategy": res.strategy.value,
            "filled_quantity": m.filled_quantity,
            "residual_quantity": m.residual_quantity,
            "completion_rate": m.completion_rate,
            "det_exec_shortfall_bps": m.execution_shortfall_bps,
            "det_opp_cost_bps": m.opportunity_cost_bps,
            "det_total_is_bps": m.total_is_bps,
            "spread_cost_usd": m.spread_cost_usd,
            "temp_impact_cost_usd": m.temporary_impact_cost_usd,
            "perm_impact_cost_usd": m.permanent_impact_cost_usd,
            "analytic_expected_is_bps": m.analytic_expected_is_bps,
            "analytic_timing_risk_bps": m.analytic_timing_risk_bps,
            "capacity_capped_slices": d.capacity_capped_slices,
            "zero_liquidity_slices": d.zero_liquidity_slices,
        }
        if mc is not None:
            row.update(
                {
                    "mc_paths": mc.num_paths,
                    "mc_mean_is_bps": mc.mean_total_is_bps,
                    "mc_std_is_bps": mc.std_total_is_bps,
                    "mc_p50_is_bps": mc.p50_total_is_bps,
                    "mc_p95_is_bps": mc.p95_total_is_bps,
                    "mc_cvar_95_bps": mc.cvar_95_bps,
                }
            )
        rows.append(row)
    return pl.DataFrame(rows)


def sensitivity_to_polars(points: Sequence[SensitivityPoint]) -> pl.DataFrame:
    """Convert sensitivity grid outputs to a Polars DataFrame."""
    rows = [
        {
            "regime_name": pt.regime_name,
            "liquidity_multiplier": pt.liquidity_multiplier,
            "spread_multiplier": pt.spread_multiplier,
            "volatility_multiplier": pt.volatility_multiplier,
            "risk_aversion_lambda": pt.risk_aversion_lambda,
            "strategy": pt.strategy,
            "completion_rate": pt.completion_rate,
            "deterministic_is_bps": pt.deterministic_is_bps,
            "mc_mean_is_bps": pt.mc_mean_is_bps,
            "mc_std_is_bps": pt.mc_std_is_bps,
            "mc_p95_is_bps": pt.mc_p95_is_bps,
            "mc_cvar_95_bps": pt.mc_cvar_95_bps,
            "analytic_expected_is_bps": pt.analytic_expected_is_bps,
            "analytic_timing_risk_bps": pt.analytic_timing_risk_bps,
            "capacity_capped_slices": pt.capacity_capped_slices,
        }
        for pt in points
    ]
    return pl.DataFrame(rows)
