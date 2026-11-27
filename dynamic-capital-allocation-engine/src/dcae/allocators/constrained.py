"""Constrained multi-strategy capital allocator using ADMM + Dykstra projection.

Maximizes point-in-time risk-adjusted return net of L1 transaction costs and
L2 turnover smoothing subject to cash reserve, per-strategy caps/floors,
group exposure caps, liquidity/ADV participation limits, one-way turnover limit,
and portfolio/stressed volatility caps.

If no feasible solution exists, deterministically returns a safe fallback
(no-change if safe, else cash-only) with explicit infeasibility reasons.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dcae.allocators.baselines import _decision_id, compute_objective_breakdown
from dcae.domain.models import (
    AllocationDecision,
    AllocationStatus,
    EngineConfig,
    FundSnapshot,
)
from dcae.estimators.point_in_time import EstimationResult, PointInTimeEstimator
from dcae.math.linalg import Matrix, Vector, vec_norm2
from dcae.math.projections import (
    ConvexConstraintSet,
    LinearCap,
    dykstra_project,
    eigen_cache,
)


def _soft_threshold(val: float, threshold: float) -> float:
    if val > threshold:
        return val - threshold
    if val < -threshold:
        return val + threshold
    return 0.0


def _solve_qp_l1_block(
    Q: Matrix,
    q: Vector,
    w0: Sequence[float],
    l1_costs: Sequence[float],
    w_init: Sequence[float],
    *,
    sweeps: int = 30,
    tol: float = 1e-11,
) -> Vector:
    """Minimize 0.5 w^T Q w + q^T w + sum_i c_i |w_i - w0_i| via exact coordinate descent."""
    n = len(q)
    w = list(w_init)
    for _ in range(sweeps):
        max_diff = 0.0
        for i in range(n):
            qii = Q[i][i]
            if qii <= 1e-15:
                continue
            # r_i = q_i + sum_{j != i} Q_{ij} w_j
            ri = q[i] + sum(Q[i][j] * w[j] for j in range(n) if j != i)
            grad_at_w0 = ri + qii * w0[i]
            delta = _soft_threshold(-grad_at_w0 / qii, l1_costs[i] / qii)
            w_new = w0[i] + delta
            diff = abs(w_new - w[i])
            if diff > max_diff:
                max_diff = diff
            w[i] = w_new
        if max_diff <= tol:
            break
    return w


class ConstrainedAllocator:
    """Convex constrained multi-strategy capital allocator with safe fallback."""

    name = "constrained_mv"

    def __init__(self, config: EngineConfig, *, name: Optional[str] = None) -> None:
        self.config = config
        self.estimator = PointInTimeEstimator(config)
        if name is not None:
            self.name = name

    def allocate(
        self,
        snapshot: FundSnapshot,
        estimation: Optional[EstimationResult] = None,
    ) -> AllocationDecision:
        est = estimation or self.estimator.estimate(snapshot)
        all_ids = list(est.strategy_ids)
        n = len(all_ids)
        cur_w_map, cur_cash = snapshot.positions.current_weights(all_ids)
        w0 = [cur_w_map[sid] for sid in all_ids]

        # Build effective bounds & convex constraint set
        cset, pre_reasons, bound_details = self._build_constraint_set(
            snapshot=snapshot,
            estimation=est,
            w0=w0,
        )

        if pre_reasons:
            return self._fallback_decision(
                snapshot=snapshot,
                estimation=est,
                current_weights=cur_w_map,
                current_cash=cur_cash,
                reasons=tuple(pre_reasons),
                diagnostics={"stage": "pre_solve_feasibility", "bounds": bound_details},
            )

        # Verify constraint intersection is non-empty via Dykstra projection from w0
        w_feas, init_viol, proj_iters = dykstra_project(
            w0,
            cset,
            max_iters=350,
            tol=1e-10,
        )
        tol = self.config.constraints.constraint_tolerance
        if init_viol > tol:
            return self._fallback_decision(
                snapshot=snapshot,
                estimation=est,
                current_weights=cur_w_map,
                current_cash=cur_cash,
                reasons=(
                    f"empty_feasible_set:max_constraint_violation={init_viol:.6e}>tol={tol:.1e}",
                ),
                diagnostics={
                    "stage": "feasibility_projection",
                    "max_violation": init_viol,
                    "projection_iters": proj_iters,
                    "bounds": bound_details,
                },
            )

        # Solve convex objective via ADMM:
        # min  0.5 * lambda * w^T Sigma_active w - mu^T w
        #      + 0.5 * gamma * ||w - w0||_2^2 + sum c_i |w_i - w0_i|
        # s.t. w in C (via z = w, z in C)
        strat_map = {s.strategy_id: s for s in snapshot.strategies}
        mu = [est.expected_returns_annual[sid] for sid in all_ids]
        l1_costs = [
            strat_map[sid].exposure.transaction_cost_bps / 10000.0
            for sid in all_ids
        ]
        lam = self.config.risk_aversion
        gamma = self.config.turnover_l2_penalty
        Sigma = est.active_covariance_annual

        # Choose well-conditioned ADMM penalty rho
        diag_scale = max(
            [lam * Sigma[i][i] + gamma for i in range(n)] or [1.0]
        )
        rho = max(0.5, 0.25 * diag_scale)

        Q: Matrix = [
            [
                lam * Sigma[i][j] + ((gamma + rho) if i == j else 0.0)
                for j in range(n)
            ]
            for i in range(n)
        ]

        w = list(w_feas)
        z = list(w_feas)
        u = [0.0] * n
        primal_res = 0.0
        dual_res = 0.0
        admm_iters = 0
        converged = False

        for it in range(1, self.config.max_iterations + 1):
            admm_iters = it
            q_vec = [-mu[i] - gamma * w0[i] - rho * (z[i] - u[i]) for i in range(n)]
            w = _solve_qp_l1_block(Q, q_vec, w0, l1_costs, w, sweeps=25, tol=1e-11)

            z_prev = list(z)
            z_target = [w[i] + u[i] for i in range(n)]
            z, z_viol, _ = dykstra_project(z_target, cset, max_iters=180, tol=1e-10)
            u = [u[i] + (w[i] - z[i]) for i in range(n)]

            primal_res = vec_norm2([w[i] - z[i] for i in range(n)])
            dual_res = rho * vec_norm2([z[i] - z_prev[i] for i in range(n)])
            if primal_res <= self.config.convergence_tol and dual_res <= self.config.convergence_tol:
                converged = True
                break

        # Final high-precision projection on z so constraints hold tightly
        z, final_viol, _ = dykstra_project(z, cset, max_iters=350, tol=1e-11)
        if final_viol > tol:
            return self._fallback_decision(
                snapshot=snapshot,
                estimation=est,
                current_weights=cur_w_map,
                current_cash=cur_cash,
                reasons=(
                    f"post_solve_constraint_violation:{final_viol:.6e}>tol={tol:.1e}",
                ),
                diagnostics={
                    "stage": "post_solve_check",
                    "admm_iters": admm_iters,
                    "primal_residual": primal_res,
                    "dual_residual": dual_res,
                    "final_violation": final_viol,
                },
            )

        # Clean near-zero dust while preserving sum/bounds
        cleaned = [0.0 if abs(v) < 1e-11 and cset.lower[i] <= 0.0 else v for i, v in enumerate(z)]
        if cset.max_violation(cleaned) <= tol:
            z = cleaned

        target_map = {sid: float(z[i]) for i, sid in enumerate(all_ids)}
        cash_w = 1.0 - sum(target_map.values())
        obj = compute_objective_breakdown(
            strategy_ids=all_ids,
            target_weights=target_map,
            current_weights=cur_w_map,
            estimation=est,
            snapshot=snapshot,
            config=self.config,
        )

        status = AllocationStatus.OPTIMAL if converged else AllocationStatus.CONVERGED_APPROX
        return AllocationDecision(
            decision_id=_decision_id(self.name, snapshot, self.config),
            allocator_name=self.name,
            snapshot_id=snapshot.snapshot_id,
            snapshot_hash=snapshot.content_hash(),
            config_version=self.config.config_version,
            config_hash=self.config.config_hash(),
            as_of_utc=snapshot.as_of_utc,
            status=status,
            target_weights=target_map,
            cash_weight=cash_w,
            current_weights=cur_w_map,
            current_cash_weight=cur_cash,
            objective=obj,
            inputs_summary=est.to_summary_dict(),
            diagnostics={
                "solver_backend": self.config.solver_backend,
                "admm_iterations": admm_iters,
                "converged": converged,
                "primal_residual": primal_res,
                "dual_residual": dual_res,
                "final_constraint_violation": final_viol,
                "bounds": bound_details,
                "correlation_stress_active": est.correlation_stress_active,
            },
        )

    def _build_constraint_set(
        self,
        *,
        snapshot: FundSnapshot,
        estimation: EstimationResult,
        w0: Sequence[float],
    ) -> Tuple[ConvexConstraintSet, List[str], Dict[str, Dict[str, float]]]:
        con = self.config.constraints
        nav = snapshot.positions.total_nav_usd
        all_ids = estimation.strategy_ids
        strat_map = {s.strategy_id: s for s in snapshot.strategies}

        lower: List[float] = []
        upper: List[float] = []
        reasons: List[str] = []
        bound_details: Dict[str, Dict[str, float]] = {}

        for i, sid in enumerate(all_ids):
            strat = strat_map[sid]
            is_excluded = sid in estimation.excluded_reasons

            # Base bounds from config + strategy override
            lo = con.strategy_min_weights.get(
                sid,
                strat.min_weight if strat.min_weight is not None else con.default_min_weight,
            )
            hi = con.strategy_max_weights.get(
                sid,
                strat.max_weight if strat.max_weight is not None else con.default_max_weight,
            )

            # Capacity cap
            cap_weight = strat.exposure.max_capacity_usd / nav
            hi = min(hi, cap_weight)

            # ADV liquidity participation cap on trade size |w_i - w0_i| <= max_adv_part * ADV / NAV
            adv_trade_limit = (con.max_adv_participation * strat.exposure.adv_usd) / nav
            lo_adv = max(0.0, w0[i] - adv_trade_limit)
            hi_adv = min(1.0, w0[i] + adv_trade_limit)

            if is_excluded:
                # Do not allocate fresh capital to ineligible/excluded strategies;
                # allow liquidating down to 0 subject to ADV liquidity limit.
                lo = 0.0
                hi = min(hi, max(0.0, w0[i] - min(w0[i], adv_trade_limit)) if w0[i] > 0.0 else 0.0)
                # Ensure lo <= hi if existing position must unwind gradually
                hi = max(hi, 0.0)
                if w0[i] > 0.0:
                    # Allow any weight between [max(0, w0 - adv), w0] so unwind is feasible
                    lo = max(0.0, w0[i] - adv_trade_limit)
                    hi = min(w0[i], hi_adv)
            else:
                lo = max(lo, lo_adv)
                hi = min(hi, hi_adv)

            if lo > hi + con.constraint_tolerance:
                reasons.append(
                    f"infeasible_strategy_bounds:{sid}:lower={lo:.6f}>upper={hi:.6f}"
                )
            lower.append(lo)
            upper.append(hi)
            bound_details[sid] = {
                "effective_lower": lo,
                "effective_upper": hi,
                "capacity_cap": cap_weight,
                "adv_trade_limit": adv_trade_limit,
            }

        sum_min = max(0.0, 1.0 - con.cash_reserve_max)
        sum_max = max(0.0, 1.0 - con.cash_reserve_min)

        if sum(upper) + con.constraint_tolerance < sum_min:
            reasons.append(
                f"infeasible_caps:sum_upper={sum(upper):.6f}<min_invested={sum_min:.6f}"
            )
        if sum(lower) - con.constraint_tolerance > sum_max:
            reasons.append(
                f"infeasible_floors:sum_lower={sum(lower):.6f}>max_invested={sum_max:.6f}"
            )

        # Group exposure caps
        linear_caps: List[LinearCap] = []
        groups_present: Dict[str, List[int]] = {}
        for i, sid in enumerate(all_ids):
            grp = strat_map[sid].exposure.strategy_group
            groups_present.setdefault(grp, []).append(i)

        group_upper_sum = 0.0
        for grp, idxs in sorted(groups_present.items()):
            cap = con.group_caps.get(grp)
            grp_lo = sum(lower[i] for i in idxs)
            grp_hi = sum(upper[i] for i in idxs)
            if cap is not None:
                if grp_lo > cap + con.constraint_tolerance:
                    reasons.append(
                        f"infeasible_group_cap:{grp}:sum_lower={grp_lo:.6f}>cap={cap:.6f}"
                    )
                coeffs = tuple(1.0 if i in idxs else 0.0 for i in range(len(all_ids)))
                linear_caps.append(LinearCap(name=f"group_cap:{grp}", coeffs=coeffs, rhs=cap))
                group_upper_sum += min(grp_hi, cap)
            else:
                group_upper_sum += grp_hi

        if group_upper_sum + con.constraint_tolerance < sum_min:
            reasons.append(
                f"infeasible_group_caps_total:max_deployable={group_upper_sum:.6f}<min_invested={sum_min:.6f}"
            )

        # Gross leverage cap: sum(gross_i * w_i) <= max_gross_leverage
        gross_coeffs = tuple(strat_map[sid].exposure.gross_exposure for sid in all_ids)
        if any(g > 0.0 for g in gross_coeffs):
            linear_caps.append(
                LinearCap(
                    name="gross_leverage_cap",
                    coeffs=gross_coeffs,
                    rhs=con.max_gross_leverage,
                )
            )

        # Volatility ellipsoid caps
        cov_eigvals = None
        cov_eigvecs = None
        max_var = None
        if con.max_annualized_volatility is not None:
            cov_eigvals, cov_eigvecs = eigen_cache(estimation.covariance_annual)
            max_var = con.max_annualized_volatility ** 2

        stressed_eigvals = None
        stressed_eigvecs = None
        max_svar = None
        if con.max_stressed_annualized_volatility is not None:
            stressed_eigvals, stressed_eigvecs = eigen_cache(estimation.stressed_covariance_annual)
            max_svar = con.max_stressed_annualized_volatility ** 2

        cset = ConvexConstraintSet(
            lower=tuple(lower),
            upper=tuple(upper),
            sum_min=sum_min,
            sum_max=sum_max,
            w0=tuple(w0),
            max_one_way_turnover=con.max_one_way_turnover,
            linear_caps=tuple(linear_caps),
            cov_eigvals=cov_eigvals,
            cov_eigvecs=cov_eigvecs,
            max_variance=max_var,
            stressed_eigvals=stressed_eigvals,
            stressed_eigvecs=stressed_eigvecs,
            max_stressed_variance=max_svar,
        )
        return cset, reasons, bound_details

    def _fallback_decision(
        self,
        *,
        snapshot: FundSnapshot,
        estimation: EstimationResult,
        current_weights: Dict[str, float],
        current_cash: float,
        reasons: Tuple[str, ...],
        diagnostics: Dict[str, Any],
    ) -> AllocationDecision:
        all_ids = list(estimation.strategy_ids)
        cur_sum = sum(current_weights.values()) + current_cash
        can_hold_no_change = (
            self.config.fallback_policy == "no_change_then_cash"
            and abs(cur_sum - 1.0) <= 1e-4
            and current_cash >= -1e-6
            and all(v >= -1e-6 for v in current_weights.values())
        )

        if can_hold_no_change:
            target = dict(current_weights)
            cash = float(current_cash)
            status = AllocationStatus.FALLBACK_NO_CHANGE
            fallback_name = "no_change"
        else:
            target = {sid: 0.0 for sid in all_ids}
            cash = 1.0
            status = AllocationStatus.FALLBACK_CASH
            fallback_name = "cash_fallback"

        obj = compute_objective_breakdown(
            strategy_ids=all_ids,
            target_weights=target,
            current_weights=current_weights,
            estimation=estimation,
            snapshot=snapshot,
            config=self.config,
        )
        diag = dict(diagnostics)
        diag["fallback_policy"] = self.config.fallback_policy
        diag["fallback_selected"] = fallback_name

        return AllocationDecision(
            decision_id=_decision_id(self.name, snapshot, self.config),
            allocator_name=self.name,
            snapshot_id=snapshot.snapshot_id,
            snapshot_hash=snapshot.content_hash(),
            config_version=self.config.config_version,
            config_hash=self.config.config_hash(),
            as_of_utc=snapshot.as_of_utc,
            status=status,
            target_weights=target,
            cash_weight=cash,
            current_weights=current_weights,
            current_cash_weight=current_cash,
            objective=obj,
            inputs_summary=estimation.to_summary_dict(),
            diagnostics=diag,
            infeasibility_reasons=reasons,
            fallback_applied=fallback_name,
        )
