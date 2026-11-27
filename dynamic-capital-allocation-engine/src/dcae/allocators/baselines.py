"""Deterministic equal-weight, inverse-volatility, and risk-budget baseline allocators."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from dcae.domain.models import (
    AllocationDecision,
    AllocationStatus,
    EngineConfig,
    FundSnapshot,
    ObjectiveBreakdown,
    sha256_hex,
)
from dcae.estimators.point_in_time import EstimationResult, PointInTimeEstimator
from dcae.math.linalg import mat_vec, quad_form


def compute_objective_breakdown(
    *,
    strategy_ids: Sequence[str],
    target_weights: Dict[str, float],
    current_weights: Dict[str, float],
    estimation: EstimationResult,
    snapshot: FundSnapshot,
    config: EngineConfig,
) -> ObjectiveBreakdown:
    """Compute exact objective terms and turnover/cost breakdown for any allocation."""
    w = [target_weights.get(sid, 0.0) for sid in strategy_ids]
    w0 = [current_weights.get(sid, 0.0) for sid in strategy_ids]
    mu = [estimation.expected_returns_annual.get(sid, 0.0) for sid in strategy_ids]
    strat_map = {s.strategy_id: s for s in snapshot.strategies}

    exp_ret = sum(wi * mi for wi, mi in zip(w, mu))
    var = max(0.0, quad_form(w, estimation.covariance_annual))
    vol = math.sqrt(var)
    svar = max(0.0, quad_form(w, estimation.stressed_covariance_annual))
    svol = math.sqrt(svar)
    active_var = max(0.0, quad_form(w, estimation.active_covariance_annual))
    var_pen = 0.5 * config.risk_aversion * active_var

    tcost_drag = 0.0
    l1_diff = 0.0
    l2_sq = 0.0
    for sid, wi, w0i in zip(strategy_ids, w, w0):
        delta = abs(wi - w0i)
        l1_diff += delta
        l2_sq += delta * delta
        bps = (
            strat_map[sid].exposure.transaction_cost_bps
            if sid in strat_map
            else config.default_transaction_cost_bps
        )
        tcost_drag += (bps / 10000.0) * delta

    one_way_turnover = 0.5 * l1_diff
    l2_pen = 0.5 * config.turnover_l2_penalty * l2_sq
    net_obj = exp_ret - var_pen - tcost_drag - l2_pen
    est_cost_usd = tcost_drag * snapshot.positions.total_nav_usd

    return ObjectiveBreakdown(
        expected_return_annual=exp_ret,
        portfolio_volatility_annual=vol,
        stressed_volatility_annual=svol,
        variance_penalty=var_pen,
        transaction_cost_drag=tcost_drag,
        turnover_l2_penalty=l2_pen,
        net_objective=net_obj,
        one_way_turnover=one_way_turnover,
        estimated_cost_usd=est_cost_usd,
    )


def _decision_id(allocator_name: str, snapshot: FundSnapshot, config: EngineConfig) -> str:
    digest = sha256_hex(
        {
            "allocator": allocator_name,
            "snapshot_id": snapshot.snapshot_id,
            "snapshot_hash": snapshot.content_hash(),
            "config_hash": config.config_hash(),
        }
    )
    return f"dec_{allocator_name}_{digest[:12]}"


def _build_baseline_decision(
    *,
    allocator_name: str,
    snapshot: FundSnapshot,
    config: EngineConfig,
    estimation: EstimationResult,
    target_weights: Dict[str, float],
    cash_weight: float,
    status: AllocationStatus,
    diagnostics: Dict[str, object],
    infeasibility_reasons: Tuple[str, ...] = (),
    fallback_applied: Optional[str] = None,
) -> AllocationDecision:
    all_ids = list(estimation.strategy_ids)
    cur_w, cur_cash = snapshot.positions.current_weights(all_ids)
    obj = compute_objective_breakdown(
        strategy_ids=all_ids,
        target_weights=target_weights,
        current_weights=cur_w,
        estimation=estimation,
        snapshot=snapshot,
        config=config,
    )
    return AllocationDecision(
        decision_id=_decision_id(allocator_name, snapshot, config),
        allocator_name=allocator_name,
        snapshot_id=snapshot.snapshot_id,
        snapshot_hash=snapshot.content_hash(),
        config_version=config.config_version,
        config_hash=config.config_hash(),
        as_of_utc=snapshot.as_of_utc,
        status=status,
        target_weights=target_weights,
        cash_weight=cash_weight,
        current_weights=cur_w,
        current_cash_weight=cur_cash,
        objective=obj,
        inputs_summary=estimation.to_summary_dict(),
        diagnostics=dict(diagnostics),
        infeasibility_reasons=infeasibility_reasons,
        fallback_applied=fallback_applied,
    )


class EqualWeightAllocator:
    """Equal-weight capital allocation across eligible strategies preserving min cash reserve."""

    name = "equal_weight"

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.estimator = PointInTimeEstimator(config)

    def allocate(
        self,
        snapshot: FundSnapshot,
        estimation: Optional[EstimationResult] = None,
    ) -> AllocationDecision:
        est = estimation or self.estimator.estimate(snapshot)
        all_ids = est.strategy_ids
        elig = est.eligible_ids
        investable = max(0.0, 1.0 - self.config.constraints.cash_reserve_min)

        if not elig:
            weights = {sid: 0.0 for sid in all_ids}
            return _build_baseline_decision(
                allocator_name=self.name,
                snapshot=snapshot,
                config=self.config,
                estimation=est,
                target_weights=weights,
                cash_weight=1.0,
                status=AllocationStatus.FALLBACK_CASH,
                diagnostics={"method": "equal_weight", "eligible_count": 0},
                infeasibility_reasons=("no_eligible_strategies",),
                fallback_applied="cash_fallback",
            )

        w_each = investable / len(elig)
        weights = {sid: (w_each if sid in elig else 0.0) for sid in all_ids}
        cash = 1.0 - sum(weights.values())
        return _build_baseline_decision(
            allocator_name=self.name,
            snapshot=snapshot,
            config=self.config,
            estimation=est,
            target_weights=weights,
            cash_weight=cash,
            status=AllocationStatus.OPTIMAL,
            diagnostics={"method": "equal_weight", "eligible_count": len(elig)},
        )


class InverseVolatilityAllocator:
    """Inverse-volatility risk-budget baseline across eligible strategies."""

    name = "inverse_volatility"

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.estimator = PointInTimeEstimator(config)

    def allocate(
        self,
        snapshot: FundSnapshot,
        estimation: Optional[EstimationResult] = None,
    ) -> AllocationDecision:
        est = estimation or self.estimator.estimate(snapshot)
        all_ids = est.strategy_ids
        elig = est.eligible_ids
        investable = max(0.0, 1.0 - self.config.constraints.cash_reserve_min)

        inv_vols: Dict[str, float] = {}
        for sid in elig:
            vol = est.volatilities_annual.get(sid, 0.0)
            if vol > self.config.zero_volatility_eps:
                inv_vols[sid] = 1.0 / vol

        if not inv_vols:
            weights = {sid: 0.0 for sid in all_ids}
            return _build_baseline_decision(
                allocator_name=self.name,
                snapshot=snapshot,
                config=self.config,
                estimation=est,
                target_weights=weights,
                cash_weight=1.0,
                status=AllocationStatus.FALLBACK_CASH,
                diagnostics={"method": "inverse_volatility", "eligible_nonzero_vol_count": 0},
                infeasibility_reasons=("no_positive_volatility_eligible_strategies",),
                fallback_applied="cash_fallback",
            )

        denom = sum(inv_vols.values())
        weights = {
            sid: (investable * inv_vols[sid] / denom if sid in inv_vols else 0.0)
            for sid in all_ids
        }
        cash = 1.0 - sum(weights.values())
        return _build_baseline_decision(
            allocator_name=self.name,
            snapshot=snapshot,
            config=self.config,
            estimation=est,
            target_weights=weights,
            cash_weight=cash,
            status=AllocationStatus.OPTIMAL,
            diagnostics={
                "method": "inverse_volatility",
                "eligible_nonzero_vol_count": len(inv_vols),
            },
        )


class RiskBudgetAllocator:
    """Risk-parity / general risk-budget baseline solved via cyclical coordinate descent.

    Solves for w >= 0 such that risk contribution RC_i = w_i * (Sigma w)_i / (w^T Sigma w)
    matches target risk budget b_i (equal risk contribution when risk_budget is unset).
    """

    name = "risk_budget"

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.estimator = PointInTimeEstimator(config)

    def allocate(
        self,
        snapshot: FundSnapshot,
        estimation: Optional[EstimationResult] = None,
    ) -> AllocationDecision:
        est = estimation or self.estimator.estimate(snapshot)
        all_ids = est.strategy_ids
        elig = est.eligible_ids
        investable = max(0.0, 1.0 - self.config.constraints.cash_reserve_min)

        if not elig:
            weights = {sid: 0.0 for sid in all_ids}
            return _build_baseline_decision(
                allocator_name=self.name,
                snapshot=snapshot,
                config=self.config,
                estimation=est,
                target_weights=weights,
                cash_weight=1.0,
                status=AllocationStatus.FALLBACK_CASH,
                diagnostics={"method": "risk_budget", "eligible_count": 0},
                infeasibility_reasons=("no_eligible_strategies",),
                fallback_applied="cash_fallback",
            )

        idx_of = {sid: i for i, sid in enumerate(all_ids)}
        elig_idx = [idx_of[sid] for sid in elig]
        m = len(elig_idx)
        sub_cov = [
            [est.active_covariance_annual[i][j] for j in elig_idx] for i in elig_idx
        ]

        strat_map = {s.strategy_id: s for s in snapshot.strategies}
        raw_budgets = [
            strat_map[sid].risk_budget if strat_map[sid].risk_budget is not None else 1.0
            for sid in elig
        ]
        b_sum = sum(raw_budgets)
        if b_sum <= 0.0:
            budgets = [1.0 / m] * m
        else:
            budgets = [b / b_sum for b in raw_budgets]

        # Spinu / Griveau-Billion cyclical coordinate descent for unscaled y > 0:
        # Minimize 0.5 y^T Sigma y - sum b_i ln(y_i)
        y = [1.0 / math.sqrt(max(sub_cov[i][i], 1e-8)) for i in range(m)]
        iters = 0
        for it in range(1, 300):
            iters = it
            max_change = 0.0
            for i in range(m):
                aii = max(sub_cov[i][i], 1e-12)
                bi = sum(sub_cov[i][j] * y[j] for j in range(m) if j != i)
                # Solve aii * y_i^2 + bi * y_i - budgets[i] = 0 for y_i > 0
                disc = bi * bi + 4.0 * aii * budgets[i]
                y_new = (-bi + math.sqrt(max(disc, 0.0))) / (2.0 * aii)
                y_new = max(y_new, 1e-12)
                max_change = max(max_change, abs(y_new - y[i]))
                y[i] = y_new
            if max_change <= 1e-11:
                break

        y_sum = sum(y)
        w_elig = [investable * yi / y_sum for yi in y]

        # Compute achieved risk contribution shares for diagnostics
        sy = mat_vec(sub_cov, w_elig)
        tot_var = max(quad_form(w_elig, sub_cov), 1e-18)
        rc_shares = {
            elig[i]: (w_elig[i] * sy[i]) / tot_var for i in range(m)
        }
        max_rc_err = max(abs(rc_shares[elig[i]] - budgets[i]) for i in range(m))

        weights = {sid: 0.0 for sid in all_ids}
        for i, sid in enumerate(elig):
            weights[sid] = w_elig[i]
        cash = 1.0 - sum(weights.values())

        return _build_baseline_decision(
            allocator_name=self.name,
            snapshot=snapshot,
            config=self.config,
            estimation=est,
            target_weights=weights,
            cash_weight=cash,
            status=AllocationStatus.OPTIMAL,
            diagnostics={
                "method": "risk_budget",
                "iterations": iters,
                "target_risk_budgets": {elig[i]: budgets[i] for i in range(m)},
                "achieved_risk_contributions": rc_shares,
                "max_risk_contribution_error": max_rc_err,
            },
        )
