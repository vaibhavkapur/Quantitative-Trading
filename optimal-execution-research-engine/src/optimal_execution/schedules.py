"""Pure domain execution schedule generators (TWAP, VWAP, POV, Almgren-Chriss)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from optimal_execution.contracts import (
    CONSERVATION_TOLERANCE,
    ExecutionScenario,
    Schedule,
    ScheduleError,
    ScheduleRequest,
    StrategyType,
    UnsupportedScenarioError,
    ValidationError,
)


def _trajectory_from_trades(parent_quantity: float, trades: np.ndarray) -> Tuple[Tuple[float, ...], float]:
    """Compute remaining inventory trajectory x_0..x_N and residual from trade list."""
    remaining = float(parent_quantity)
    traj: List[float] = [remaining]
    for qty in trades:
        remaining = max(0.0, remaining - float(qty))
        traj.append(remaining)
    planned_total = float(np.sum(trades))
    residual = max(0.0, float(parent_quantity) - planned_total)
    traj[-1] = residual
    return tuple(traj), residual


def twap_schedule(scenario: ExecutionScenario) -> Schedule:
    """Time-Weighted Average Price schedule: equal share slices over the horizon."""
    scenario.validate()
    n = scenario.order.horizon_slices
    q = float(scenario.order.quantity)
    trades = np.full(n, q / n, dtype=np.float64)
    traj, residual = _trajectory_from_trades(q, trades)
    schedule = Schedule(
        strategy=StrategyType.TWAP,
        planned_quantities=tuple(float(x) for x in trades),
        remaining_trajectory=traj,
        planned_residual=residual,
        parameters={"horizon_slices": n, "slice_quantity": q / n},
    )
    schedule.validate(scenario.order)
    return schedule


def vwap_schedule(scenario: ExecutionScenario) -> Schedule:
    """Volume-Weighted Average Price schedule proportional to expected market volume."""
    scenario.validate()
    volumes = np.asarray(scenario.market.volume_curve, dtype=np.float64)
    total_vol = float(np.sum(volumes))
    q = float(scenario.order.quantity)
    n = scenario.order.horizon_slices

    if total_vol <= 0.0:
        trades = np.zeros(n, dtype=np.float64)
        traj, residual = _trajectory_from_trades(q, trades)
        schedule = Schedule(
            strategy=StrategyType.VWAP,
            planned_quantities=tuple(float(x) for x in trades),
            remaining_trajectory=traj,
            planned_residual=residual,
            parameters={"total_market_volume": 0.0, "zero_liquidity_fallback": True},
        )
        schedule.validate(scenario.order)
        return schedule

    weights = volumes / total_vol
    trades = q * weights
    traj, residual = _trajectory_from_trades(q, trades)
    schedule = Schedule(
        strategy=StrategyType.VWAP,
        planned_quantities=tuple(float(x) for x in trades),
        remaining_trajectory=traj,
        planned_residual=residual,
        parameters={
            "total_market_volume": total_vol,
            "zero_liquidity_fallback": False,
            "weights": [float(w) for w in weights],
        },
    )
    schedule.validate(scenario.order)
    return schedule


def participation_schedule(
    scenario: ExecutionScenario,
    target_participation_rate: float = 0.10,
) -> Schedule:
    """Percentage-of-Volume (POV) schedule targeting a constant market participation rate.

    Executes `min(remaining, target_participation_rate * market_volume[k])` per slice.
    If cumulative target volume across the horizon is less than the parent quantity,
    the unscheduled balance is left as `planned_residual`.
    """
    scenario.validate()
    if (
        isinstance(target_participation_rate, bool)
        or not isinstance(target_participation_rate, (int, float))
        or not math.isfinite(target_participation_rate)
        or target_participation_rate <= 0.0
        or target_participation_rate > 1.0
    ):
        raise ValidationError(
            f"target_participation_rate must be in (0, 1], got {target_participation_rate}"
        )

    q = float(scenario.order.quantity)
    remaining = q
    trades_list: List[float] = []
    for vol in scenario.market.volume_curve:
        target = float(target_participation_rate) * float(vol)
        qty = min(remaining, max(0.0, target))
        trades_list.append(qty)
        remaining = max(0.0, remaining - qty)

    trades = np.asarray(trades_list, dtype=np.float64)
    traj, residual = _trajectory_from_trades(q, trades)
    schedule = Schedule(
        strategy=StrategyType.PARTICIPATION,
        planned_quantities=tuple(float(x) for x in trades),
        remaining_trajectory=traj,
        planned_residual=residual,
        parameters={
            "target_participation_rate": float(target_participation_rate),
            "total_market_volume": scenario.market.total_market_volume,
        },
    )
    schedule.validate(scenario.order)
    return schedule


def almgren_chriss_objective(
    trades: np.ndarray,
    scenario: ExecutionScenario,
    risk_aversion_lambda: float,
) -> Tuple[float, np.ndarray]:
    """Exact discrete-time Almgren-Chriss objective E[IS] + lambda * Var[IS] and gradient.

    Conventions (Almgren & Chriss 2001):
      - E[IS] = sum_k [ epsilon * n_k + eta * n_k * (n_k / tau)^alpha + gamma * n_k * sum_{j<k} n_j ]
      - Var[IS] = sigma_$^2 * tau * sum_{k=0}^{N-2} x_{k+1}^2, where x_{k+1} = Q - sum_{i<=k} n_i
    """
    n_arr = np.asarray(trades, dtype=np.float64)
    q = float(scenario.order.quantity)
    tau = scenario.order.slice_duration_days
    s0 = float(scenario.order.arrival_price)
    sigma_usd = s0 * float(scenario.market.daily_volatility)
    eps = scenario.market.half_spread_usd(s0)
    gamma = float(scenario.impact.permanent_impact_gamma)
    eta = float(scenario.impact.temporary_impact_eta)
    alpha = float(scenario.impact.temporary_impact_alpha)
    lam = float(risk_aversion_lambda)

    # Guard against tiny negative numerical noise from optimizer line search.
    n_pos = np.maximum(n_arr, 0.0)
    total_n = float(np.sum(n_pos))

    # Spread cost: eps * sum(n_k)
    spread_cost = eps * total_n
    grad_spread = np.full_like(n_arr, eps)

    # Permanent impact cost: gamma * sum_{k} n_k * sum_{j<k} n_j = 0.5 * gamma * (sum(n)^2 - sum(n^2))
    perm_cost = 0.5 * gamma * (total_n * total_n - float(np.sum(n_pos * n_pos)))
    grad_perm = gamma * (total_n - n_pos)

    # Temporary impact cost: eta * sum_k n_k * (n_k / tau)^alpha = (eta / tau^alpha) * sum_k n_k^(1 + alpha)
    inv_tau_alpha = 1.0 / (tau ** alpha)
    if alpha == 1.0:
        temp_cost = eta * inv_tau_alpha * float(np.sum(n_pos * n_pos))
        grad_temp = 2.0 * eta * inv_tau_alpha * n_pos
    else:
        temp_cost = eta * inv_tau_alpha * float(np.sum(np.power(n_pos, 1.0 + alpha)))
        # Avoid 0^(alpha) issues when alpha < 1 by clamping floor for derivative.
        grad_temp = eta * (1.0 + alpha) * inv_tau_alpha * np.power(np.maximum(n_pos, 1e-18), alpha)

    # Remaining inventory x_1 .. x_{N-1} (post-slice 0 .. post-slice N-2)
    if len(n_arr) > 1:
        cum_trades = np.cumsum(n_pos)
        x_post = q - cum_trades[:-1]  # length N - 1: x_1 .. x_{N-1}
        var_factor = (sigma_usd ** 2) * tau
        variance = var_factor * float(np.sum(x_post * x_post))
        # d(Var)/d(n_m) = -2 * var_factor * sum_{k=m}^{N-2} x_{k+1} for m <= N-2, 0 for m = N-1
        suffix_sums = np.cumsum(x_post[::-1])[::-1]
        grad_var = np.zeros_like(n_arr)
        grad_var[:-1] = -2.0 * var_factor * suffix_sums
    else:
        variance = 0.0
        grad_var = np.zeros_like(n_arr)

    objective = (spread_cost + perm_cost + temp_cost) + lam * variance
    gradient = (grad_spread + grad_perm + grad_temp) + lam * grad_var
    return float(objective), gradient


def almgren_chriss_analytic_schedule(
    scenario: ExecutionScenario,
    risk_aversion_lambda: float = 1e-6,
) -> Schedule:
    """Closed-form discrete-time Almgren-Chriss (2001) optimal liquidation/acquisition schedule.

    Minimizes E[IS] + lambda * Var[IS] subject to sum(n_k) = Q under linear temporary
    impact (alpha = 1.0) and strict convexity condition eta_tilde = eta - 0.5 * gamma * tau > 0.
    """
    scenario.validate()
    if (
        isinstance(risk_aversion_lambda, bool)
        or not isinstance(risk_aversion_lambda, (int, float))
        or not math.isfinite(risk_aversion_lambda)
        or risk_aversion_lambda < 0.0
    ):
        raise ValidationError(
            f"risk_aversion_lambda must be finite and >= 0, got {risk_aversion_lambda}"
        )

    if abs(scenario.impact.temporary_impact_alpha - 1.0) > 1e-12:
        raise UnsupportedScenarioError(
            "Analytic Almgren-Chriss schedule requires linear temporary impact "
            "(temporary_impact_alpha == 1.0). Use StrategyType.ALMGREN_CHRISS_NUMERIC "
            "for nonlinear temporary impact."
        )

    tau = scenario.order.slice_duration_days
    eta_tilde = scenario.impact.effective_eta(tau)
    if eta_tilde <= 0.0:
        raise ScheduleError(
            f"Almgren-Chriss convexity condition violated: eta_tilde = eta - 0.5*gamma*tau = "
            f"{eta_tilde:.6e} <= 0. Increase temporary_impact_eta or decrease permanent_impact_gamma."
        )

    n = scenario.order.horizon_slices
    q = float(scenario.order.quantity)
    s0 = float(scenario.order.arrival_price)
    sigma_usd = s0 * float(scenario.market.daily_volatility)
    lam = float(risk_aversion_lambda)

    if lam == 0.0 or sigma_usd == 0.0 or n == 1:
        trades = np.full(n, q / n, dtype=np.float64)
        traj, residual = _trajectory_from_trades(q, trades)
        schedule = Schedule(
            strategy=StrategyType.ALMGREN_CHRISS_ANALYTIC,
            planned_quantities=tuple(float(x) for x in trades),
            remaining_trajectory=traj,
            planned_residual=residual,
            parameters={
                "risk_aversion_lambda": lam,
                "eta_tilde": eta_tilde,
                "kappa_tilde": 0.0,
                "kappa": 0.0,
                "kappa_tau": 0.0,
                "half_life_slices": None,
                "regime": "risk_neutral_twap_limit",
            },
        )
        schedule.validate(scenario.order)
        return schedule

    kappa_tilde = math.sqrt((lam * (sigma_usd ** 2)) / eta_tilde)
    # Exact discrete-time dispersion relation: sinh(kappa * tau / 2) = 0.5 * kappa_tilde * tau
    kappa_tau = 2.0 * math.asinh(0.5 * kappa_tilde * tau)
    kappa = kappa_tau / tau

    # Overflow- and cancellation-safe evaluation of sinh(kappa_tau * (N - j)) / sinh(kappa_tau * N)
    # = exp(-kappa_tau * j) * (-expm1(-2 * kappa_tau * (N - j))) / (-expm1(-2 * kappa_tau * N))
    denom = -math.expm1(-2.0 * kappa_tau * n)
    traj_arr = np.zeros(n + 1, dtype=np.float64)
    traj_arr[0] = q
    for j in range(1, n):
        numer = math.exp(-kappa_tau * j) * (-math.expm1(-2.0 * kappa_tau * (n - j)))
        traj_arr[j] = q * (numer / denom)
    traj_arr[n] = 0.0

    trades = np.maximum(0.0, traj_arr[:-1] - traj_arr[1:])
    # Re-normalize tiny floating-point rounding drift so sum(trades) == q to machine precision.
    trade_sum = float(np.sum(trades))
    if trade_sum > 0.0:
        trades *= q / trade_sum

    traj, residual = _trajectory_from_trades(q, trades)
    half_life_slices = math.log(2.0) / kappa_tau if kappa_tau > 0.0 else None

    schedule = Schedule(
        strategy=StrategyType.ALMGREN_CHRISS_ANALYTIC,
        planned_quantities=tuple(float(x) for x in trades),
        remaining_trajectory=traj,
        planned_residual=residual,
        parameters={
            "risk_aversion_lambda": lam,
            "eta_tilde": eta_tilde,
            "kappa_tilde": kappa_tilde,
            "kappa": kappa,
            "kappa_tau": kappa_tau,
            "half_life_slices": half_life_slices,
            "regime": "analytic_discrete_sinh",
        },
    )
    schedule.validate(scenario.order)
    return schedule


def almgren_chriss_numeric_schedule(
    scenario: ExecutionScenario,
    risk_aversion_lambda: float = 1e-6,
    enforce_capacity: bool = False,
) -> Schedule:
    """Numeric optimization of discrete-time Almgren-Chriss objective using SciPy SLSQP.

    Supports both linear and nonlinear temporary impact (`temporary_impact_alpha > 0`),
    as well as optional per-slice capacity bounds (`enforce_capacity=True`).
    """
    scenario.validate()
    if (
        isinstance(risk_aversion_lambda, bool)
        or not isinstance(risk_aversion_lambda, (int, float))
        or not math.isfinite(risk_aversion_lambda)
        or risk_aversion_lambda < 0.0
    ):
        raise ValidationError(
            f"risk_aversion_lambda must be finite and >= 0, got {risk_aversion_lambda}"
        )

    tau = scenario.order.slice_duration_days
    if abs(scenario.impact.temporary_impact_alpha - 1.0) <= 1e-12:
        eta_tilde = scenario.impact.effective_eta(tau)
        if eta_tilde <= 0.0:
            raise ScheduleError(
                f"Almgren-Chriss convexity condition violated: eta_tilde = {eta_tilde:.6e} <= 0."
            )

    n = scenario.order.horizon_slices
    q = float(scenario.order.quantity)
    lam = float(risk_aversion_lambda)

    if enforce_capacity:
        capacities = np.asarray(scenario.market.slice_capacities(), dtype=np.float64)
        total_cap = float(np.sum(capacities))
        if total_cap <= 0.0:
            trades = np.zeros(n, dtype=np.float64)
            traj, residual = _trajectory_from_trades(q, trades)
            schedule = Schedule(
                strategy=StrategyType.ALMGREN_CHRISS_NUMERIC,
                planned_quantities=tuple(float(x) for x in trades),
                remaining_trajectory=traj,
                planned_residual=residual,
                parameters={
                    "risk_aversion_lambda": lam,
                    "enforce_capacity": True,
                    "capacity_infeasible_full_execution": True,
                    "optimizer_status": "zero_capacity_trivial",
                },
            )
            schedule.validate(scenario.order)
            return schedule
        if total_cap <= q + CONSERVATION_TOLERANCE:
            # Every slice must execute at full capacity to minimize shortfall/residual.
            trades = np.minimum(capacities, q)
            if float(np.sum(trades)) > q:
                trades *= q / float(np.sum(trades))
            traj, residual = _trajectory_from_trades(q, trades)
            schedule = Schedule(
                strategy=StrategyType.ALMGREN_CHRISS_NUMERIC,
                planned_quantities=tuple(float(x) for x in trades),
                remaining_trajectory=traj,
                planned_residual=residual,
                parameters={
                    "risk_aversion_lambda": lam,
                    "enforce_capacity": True,
                    "capacity_infeasible_full_execution": total_cap < q - CONSERVATION_TOLERANCE,
                    "optimizer_status": "capacity_saturated",
                },
            )
            schedule.validate(scenario.order)
            return schedule
        upper_u = np.minimum(1.0, capacities / q)
        # Feasible initial guess proportional to capacity
        u0 = (capacities / total_cap).astype(np.float64)
    else:
        upper_u = np.ones(n, dtype=np.float64)
        u0 = np.full(n, 1.0 / n, dtype=np.float64)

    # Scale objective by initial magnitude to guarantee well-conditioned SLSQP convergence.
    init_obj, _ = almgren_chriss_objective(q * u0, scenario, lam)
    obj_scale = max(abs(init_obj), 1.0)

    def fun(u: np.ndarray) -> float:
        val, _ = almgren_chriss_objective(q * u, scenario, lam)
        return val / obj_scale

    def jac(u: np.ndarray) -> np.ndarray:
        _, grad = almgren_chriss_objective(q * u, scenario, lam)
        return (grad * q) / obj_scale

    bounds = [(0.0, float(hi)) for hi in upper_u]
    constraints = [{"type": "eq", "fun": lambda u: float(np.sum(u) - 1.0), "jac": lambda u: np.ones(n)}]

    result = minimize(
        fun=fun,
        x0=u0,
        method="SLSQP",
        jac=jac,
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-14, "maxiter": 1000},
    )

    if not result.success:
        raise ScheduleError(
            f"Numeric Almgren-Chriss optimizer failed to converge: {result.message}"
        )

    u_opt = np.clip(result.x, 0.0, upper_u)
    u_sum = float(np.sum(u_opt))
    if abs(u_sum - 1.0) > 1e-5:
        raise ScheduleError(
            f"Numeric optimizer violated conservation constraint: sum(u) = {u_sum:.8f}"
        )
    if not enforce_capacity and u_sum > 0.0:
        u_opt = u_opt / u_sum

    trades = q * u_opt
    if enforce_capacity:
        # Ensure hard cap never exceeded by floating-point noise.
        trades = np.minimum(trades, capacities)

    opt_obj, _ = almgren_chriss_objective(trades, scenario, lam)
    traj, residual = _trajectory_from_trades(q, trades)
    schedule = Schedule(
        strategy=StrategyType.ALMGREN_CHRISS_NUMERIC,
        planned_quantities=tuple(float(x) for x in trades),
        remaining_trajectory=traj,
        planned_residual=residual,
        parameters={
            "risk_aversion_lambda": lam,
            "enforce_capacity": enforce_capacity,
            "capacity_infeasible_full_execution": False,
            "optimizer_status": str(result.message),
            "optimizer_iterations": int(result.nit),
            "objective_value_usd": float(opt_obj),
        },
    )
    schedule.validate(scenario.order)
    return schedule


def compare_analytic_and_numeric_schedules(
    scenario: ExecutionScenario,
    risk_aversion_lambda: float = 1e-6,
    enforce_capacity_on_numeric: bool = False,
    rel_l2_tolerance: float = 1e-4,
) -> Dict[str, Any]:
    """Compare closed-form analytic Almgren-Chriss schedule against SciPy numeric solution."""
    analytic = almgren_chriss_analytic_schedule(scenario, risk_aversion_lambda)
    numeric = almgren_chriss_numeric_schedule(
        scenario,
        risk_aversion_lambda,
        enforce_capacity=enforce_capacity_on_numeric,
    )

    a_vec = np.asarray(analytic.planned_quantities, dtype=np.float64)
    n_vec = np.asarray(numeric.planned_quantities, dtype=np.float64)
    diff = a_vec - n_vec

    max_abs_diff = float(np.max(np.abs(diff)))
    l2_diff = float(np.linalg.norm(diff))
    a_norm = float(np.linalg.norm(a_vec))
    rel_l2_diff = l2_diff / a_norm if a_norm > 0.0 else 0.0

    obj_analytic, _ = almgren_chriss_objective(a_vec, scenario, risk_aversion_lambda)
    obj_numeric, _ = almgren_chriss_objective(n_vec, scenario, risk_aversion_lambda)
    obj_rel_diff = abs(obj_analytic - obj_numeric) / max(abs(obj_analytic), 1.0)

    return {
        "scenario_id": scenario.scenario_id,
        "risk_aversion_lambda": float(risk_aversion_lambda),
        "enforce_capacity_on_numeric": enforce_capacity_on_numeric,
        "analytic_trades": [float(x) for x in a_vec],
        "numeric_trades": [float(x) for x in n_vec],
        "max_abs_trade_diff_shares": max_abs_diff,
        "l2_trade_diff_shares": l2_diff,
        "relative_l2_diff": rel_l2_diff,
        "analytic_objective_usd": float(obj_analytic),
        "numeric_objective_usd": float(obj_numeric),
        "relative_objective_diff": float(obj_rel_diff),
        "tolerance": float(rel_l2_tolerance),
        "within_tolerance": bool(rel_l2_diff <= rel_l2_tolerance),
    }


def generate_schedule(
    scenario: ExecutionScenario,
    request: ScheduleRequest,
) -> Schedule:
    """Unified dispatcher for all supported execution schedule strategies."""
    scenario.validate()
    request.validate()

    if request.strategy is StrategyType.TWAP:
        return twap_schedule(scenario)
    if request.strategy is StrategyType.VWAP:
        return vwap_schedule(scenario)
    if request.strategy is StrategyType.PARTICIPATION:
        return participation_schedule(
            scenario,
            target_participation_rate=request.target_participation_rate,
        )
    if request.strategy is StrategyType.ALMGREN_CHRISS_ANALYTIC:
        return almgren_chriss_analytic_schedule(
            scenario,
            risk_aversion_lambda=request.risk_aversion_lambda,
        )
    if request.strategy is StrategyType.ALMGREN_CHRISS_NUMERIC:
        return almgren_chriss_numeric_schedule(
            scenario,
            risk_aversion_lambda=request.risk_aversion_lambda,
            enforce_capacity=request.enforce_capacity_in_optimizer,
        )
    raise UnsupportedScenarioError(f"Unsupported strategy: {request.strategy!r}")
