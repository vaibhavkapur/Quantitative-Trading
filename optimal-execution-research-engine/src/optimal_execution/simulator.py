"""Pure domain fill simulation engine with capacity, impact, liquidity, and shock modeling."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from optimal_execution.analytics import compute_cost_metrics, summarize_monte_carlo
from optimal_execution.contracts import (
    CONSERVATION_TOLERANCE,
    CapacityPolicy,
    ExecutionScenario,
    FillSlice,
    MonteCarloSummary,
    Schedule,
    SimulationDiagnostics,
    SimulationResult,
    ValidationError,
)


def _shocks_by_slice(scenario: ExecutionScenario) -> Tuple[np.ndarray, Dict[int, int]]:
    """Aggregate exogenous price shocks in USD per share for each slice."""
    n = scenario.order.horizon_slices
    s0 = float(scenario.order.arrival_price)
    shocks_usd = np.zeros(n, dtype=np.float64)
    counts: Dict[int, int] = {}
    for shock in scenario.market.price_shocks:
        shocks_usd[shock.slice_index] += (shock.shock_bps * 1e-4) * s0
        counts[shock.slice_index] = counts.get(shock.slice_index, 0) + 1
    return shocks_usd, counts


def _simulate_path(
    scenario: ExecutionScenario,
    schedule: Schedule,
    price_innovations_usd: np.ndarray,
    stochastic: bool,
    seed: Optional[int],
) -> SimulationResult:
    """Core state transition loop isolated from I/O and RNG management.

    For each slice k = 0 .. N-1:
      1. Apply exogenous shock at slice k to mid-price before execution.
      2. Compute target quantity = planned_k (+ carryover if CapacityPolicy.SPILLOVER),
         clamped by remaining parent quantity.
      3. Enforce per-slice capacity cap (`filled_k <= capacity_k`).
         Zero market volume implies `capacity_k == 0.0` and `filled_k == 0.0`.
      4. Execution price:
           exec_px_k = mid_before_k + s * (half_spread + eta * (filled_k / tau)^alpha)
      5. Post-slice mid-price transition:
           mid_after_k = mid_before_k + s * gamma * filled_k + price_innovations_usd[k]
    """
    n = scenario.order.horizon_slices
    q = float(scenario.order.quantity)
    s0 = float(scenario.order.arrival_price)
    side_sign = scenario.order.side.sign
    tau = scenario.order.slice_duration_days
    eps = scenario.market.half_spread_usd(s0)
    gamma = float(scenario.impact.permanent_impact_gamma)
    eta = float(scenario.impact.temporary_impact_eta)
    alpha = float(scenario.impact.temporary_impact_alpha)
    capacities = scenario.market.slice_capacities()
    shocks_usd, shock_counts = _shocks_by_slice(scenario)

    remaining = q
    carryover = 0.0
    cum_diffusion_usd = 0.0
    cum_perm_impact_usd = 0.0  # signed magnitude >= 0; multiplied by side_sign in price
    cum_shock_usd = 0.0

    fill_slices: List[FillSlice] = []
    capped_slices = 0
    zero_liq_slices = 0
    shock_events = 0
    spillover_events = 0
    max_participation = 0.0

    for k in range(n):
        vol_k = float(scenario.market.volume_curve[k])
        cap_k = float(capacities[k])
        planned_k = float(schedule.planned_quantities[k])
        shock_k = float(shocks_usd[k])

        if k in shock_counts:
            shock_events += shock_counts[k]
        cum_shock_usd += shock_k

        mid_before = s0 + cum_diffusion_usd + cum_shock_usd + side_sign * cum_perm_impact_usd

        if scenario.capacity_policy is CapacityPolicy.SPILLOVER and carryover > 0.0:
            spillover_events += 1
            raw_target = planned_k + carryover
        else:
            raw_target = planned_k

        target_k = min(remaining, max(0.0, raw_target))
        was_zero_liq = vol_k <= 0.0 or cap_k <= 0.0
        if was_zero_liq:
            zero_liq_slices += 1

        filled_k = min(target_k, cap_k)
        unfilled_k = max(0.0, target_k - filled_k)
        was_capped = target_k > cap_k + 1e-12

        if was_capped:
            capped_slices += 1

        if scenario.capacity_policy is CapacityPolicy.SPILLOVER:
            carryover = unfilled_k
        else:
            carryover = 0.0

        remaining_before = remaining
        remaining_after = max(0.0, remaining - filled_k)
        remaining = remaining_after

        if filled_k > 0.0:
            rate = filled_k / tau
            temp_unit_usd = eta * (rate ** alpha)
            exec_px: Optional[float] = mid_before + side_sign * (eps + temp_unit_usd)
            spread_cost = filled_k * eps
            temp_cost = filled_k * temp_unit_usd
            perm_cost = filled_k * cum_perm_impact_usd
            timing_cost = side_sign * filled_k * cum_diffusion_usd
            shock_cost = side_sign * filled_k * cum_shock_usd
            slice_shortfall = side_sign * filled_k * (exec_px - s0)
        else:
            exec_px = None
            spread_cost = 0.0
            temp_cost = 0.0
            perm_cost = 0.0
            timing_cost = 0.0
            shock_cost = 0.0
            slice_shortfall = 0.0

        participation = (filled_k / vol_k) if vol_k > 0.0 else 0.0
        if participation > max_participation:
            max_participation = participation

        # Advance state to post-slice mid-price
        perm_increment = gamma * filled_k
        cum_perm_impact_usd += perm_increment
        cum_diffusion_usd += float(price_innovations_usd[k])
        mid_after = s0 + cum_diffusion_usd + cum_shock_usd + side_sign * cum_perm_impact_usd

        fill_slices.append(
            FillSlice(
                slice_index=k,
                market_volume=vol_k,
                capacity=cap_k,
                planned_quantity=planned_k,
                target_quantity=target_k,
                filled_quantity=filled_k,
                unfilled_quantity=unfilled_k,
                remaining_before=remaining_before,
                remaining_after=remaining_after,
                mid_price_before=mid_before,
                mid_price_after=mid_after,
                execution_price=exec_px,
                participation_rate=participation,
                spread_cost_usd=spread_cost,
                temporary_impact_cost_usd=temp_cost,
                permanent_impact_cost_usd=perm_cost,
                timing_drift_cost_usd=timing_cost,
                shock_cost_usd=shock_cost,
                total_slice_shortfall_usd=slice_shortfall,
                was_capacity_capped=was_capped,
                was_zero_liquidity=was_zero_liq,
                shock_applied_usd=shock_k,
            )
        )

    terminal_mid = fill_slices[-1].mid_price_after
    metrics = compute_cost_metrics(scenario, fill_slices, terminal_mid)
    conservation_err = abs((metrics.filled_quantity + metrics.residual_quantity) - q)

    diagnostics = SimulationDiagnostics(
        capacity_capped_slices=capped_slices,
        zero_liquidity_slices=zero_liq_slices,
        shock_events_applied=shock_events,
        spillover_events=spillover_events,
        partial_fill=metrics.residual_quantity > CONSERVATION_TOLERANCE,
        max_observed_participation=max_participation,
        conservation_error=conservation_err,
    )

    return SimulationResult(
        scenario_id=scenario.scenario_id,
        strategy=schedule.strategy,
        schedule=schedule,
        slices=tuple(fill_slices),
        metrics=metrics,
        diagnostics=diagnostics,
        stochastic=stochastic,
        seed=seed,
    )


def simulate_deterministic(
    scenario: ExecutionScenario,
    schedule: Schedule,
) -> SimulationResult:
    """Run a zero-diffusion deterministic execution simulation (includes impact, spread, capacity, shocks)."""
    scenario.validate()
    schedule.validate(scenario.order)
    zero_innovations = np.zeros(scenario.order.horizon_slices, dtype=np.float64)
    return _simulate_path(
        scenario=scenario,
        schedule=schedule,
        price_innovations_usd=zero_innovations,
        stochastic=False,
        seed=None,
    )


def simulate_single_stochastic_path(
    scenario: ExecutionScenario,
    schedule: Schedule,
    seed: Optional[int] = None,
) -> SimulationResult:
    """Run one reproducible stochastic price path using a fixed NumPy Generator seed."""
    scenario.validate()
    schedule.validate(scenario.order)
    effective_seed = scenario.seed if seed is None else seed
    if isinstance(effective_seed, bool) or not isinstance(effective_seed, int) or effective_seed < 0:
        raise ValidationError(f"seed must be a non-negative int, got {effective_seed!r}")

    rng = np.random.default_rng(effective_seed)
    n = scenario.order.horizon_slices
    s0 = float(scenario.order.arrival_price)
    sigma_slice = s0 * float(scenario.market.daily_volatility) * math.sqrt(scenario.order.slice_duration_days)
    innovations = rng.normal(loc=0.0, scale=sigma_slice, size=n)

    return _simulate_path(
        scenario=scenario,
        schedule=schedule,
        price_innovations_usd=innovations,
        stochastic=True,
        seed=effective_seed,
    )


def simulate_monte_carlo(
    scenario: ExecutionScenario,
    schedule: Schedule,
    num_paths: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[MonteCarloSummary, np.ndarray]:
    """Run vectorized Monte Carlo simulation for implementation shortfall distribution.

    Because fill quantities in the modeled capacity/schedule rules depend on market volume
    and planned schedule (independent of price diffusion), vectorized MC computes exact
    pathwise total IS, execution shortfall, and opportunity cost matching `_simulate_path`
    to machine precision while running thousands of paths in milliseconds.

    Returns `(MonteCarloSummary, pathwise_total_is_bps)`.
    """
    scenario.validate()
    schedule.validate(scenario.order)

    paths = scenario.num_monte_carlo_paths if num_paths is None else num_paths
    if isinstance(paths, bool) or not isinstance(paths, int) or paths < 1:
        raise ValidationError(f"num_paths must be an integer >= 1, got {paths!r}")

    effective_seed = scenario.seed if seed is None else seed
    if isinstance(effective_seed, bool) or not isinstance(effective_seed, int) or effective_seed < 0:
        raise ValidationError(f"seed must be a non-negative int, got {effective_seed!r}")

    # Obtain exact deterministic baseline (contains exact capacity-constrained fills,
    # spread, temporary impact, permanent impact, and shock costs).
    det = simulate_deterministic(scenario, schedule)
    fills = np.asarray([sl.filled_quantity for sl in det.slices], dtype=np.float64)
    residual = float(det.metrics.residual_quantity)
    det_exec_shortfall = float(det.metrics.execution_shortfall_usd)
    det_opp_cost = float(det.metrics.opportunity_cost_usd)

    n = scenario.order.horizon_slices
    s0 = float(scenario.order.arrival_price)
    side_sign = scenario.order.side.sign
    sigma_slice = s0 * float(scenario.market.daily_volatility) * math.sqrt(scenario.order.slice_duration_days)

    rng = np.random.default_rng(effective_seed)
    # Shape: (paths, N)
    innovations = rng.normal(loc=0.0, scale=sigma_slice, size=(paths, n))
    # Cumulative diffusion before each slice k: diff_before[:, 0] = 0, diff_before[:, k] = sum_{j<k} dW_j
    cum_diff = np.cumsum(innovations, axis=1)
    diff_before = np.zeros_like(cum_diff)
    if n > 1:
        diff_before[:, 1:] = cum_diff[:, :-1]
    terminal_diff = cum_diff[:, -1]

    # Pathwise execution timing drift + opportunity cost diffusion
    timing_exec_usd = side_sign * np.sum(diff_before * fills, axis=1)
    timing_opp_usd = side_sign * residual * terminal_diff

    exec_shortfall_paths = det_exec_shortfall + timing_exec_usd
    opp_cost_paths = det_opp_cost + timing_opp_usd
    total_is_paths = exec_shortfall_paths + opp_cost_paths
    completion_rates = np.full(paths, det.metrics.completion_rate, dtype=np.float64)

    summary = summarize_monte_carlo(
        scenario=scenario,
        strategy_name=schedule.strategy,
        seed=effective_seed,
        total_is_usd=total_is_paths,
        exec_shortfall_usd=exec_shortfall_paths,
        opportunity_cost_usd=opp_cost_paths,
        completion_rates=completion_rates,
        reference_fills=fills,
    )
    pathwise_bps = total_is_paths * (1e4 / scenario.order.arrival_notional)
    return summary, pathwise_bps
