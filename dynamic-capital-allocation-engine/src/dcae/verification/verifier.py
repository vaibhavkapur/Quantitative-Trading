"""Independent constraint verification engine.

Decoupled from the optimizer: independently checks every capital, cash reserve,
exposure, liquidity, turnover, risk, and freshness constraint on any
AllocationDecision and emits a structured FeasibilityReport.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from dcae.domain.models import (
    AllocationDecision,
    ConstraintCheckResult,
    EngineConfig,
    FeasibilityReport,
    FundSnapshot,
    parse_utc,
)
from dcae.estimators.point_in_time import EstimationResult
from dcae.math.linalg import quad_form


class ConstraintVerifier:
    """Independently audits an AllocationDecision against all EngineConfig constraints."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    def verify(
        self,
        *,
        snapshot: FundSnapshot,
        decision: AllocationDecision,
        estimation: Optional[EstimationResult] = None,
    ) -> FeasibilityReport:
        con = self.config.constraints
        tol = con.constraint_tolerance
        nav = snapshot.positions.total_nav_usd
        checks: List[ConstraintCheckResult] = []

        strat_map = {s.strategy_id: s for s in snapshot.strategies}
        all_ids = [s.strategy_id for s in snapshot.strategies]
        cur_w, _ = snapshot.positions.current_weights(all_ids)
        target_w = {sid: float(decision.target_weights.get(sid, 0.0)) for sid in all_ids}
        cash_w = float(decision.cash_weight)

        # 1. Allocation sum + cash reserve reconciliation
        invested_sum = sum(target_w.values())
        total_sum = invested_sum + cash_w
        rec_err = abs(total_sum - 1.0)
        checks.append(
            ConstraintCheckResult(
                name="allocation_sum_reconciliation",
                passed=rec_err <= tol,
                actual=total_sum,
                limit=1.0,
                slack=tol - rec_err,
                unit="weight",
                severity="hard",
                message=(
                    f"sum(weights) + cash = {total_sum:.8f} (error={rec_err:.2e}, tol={tol:.1e})"
                ),
            )
        )

        # 2. Cash reserve bounds
        checks.append(
            ConstraintCheckResult(
                name="cash_reserve_min",
                passed=cash_w >= con.cash_reserve_min - tol,
                actual=cash_w,
                limit=con.cash_reserve_min,
                slack=cash_w - con.cash_reserve_min,
                unit="weight",
                severity="hard",
                message=f"cash={cash_w:.6f} >= min={con.cash_reserve_min:.6f}",
            )
        )
        checks.append(
            ConstraintCheckResult(
                name="cash_reserve_max",
                passed=cash_w <= con.cash_reserve_max + tol,
                actual=cash_w,
                limit=con.cash_reserve_max,
                slack=con.cash_reserve_max - cash_w,
                unit="weight",
                severity="hard",
                message=f"cash={cash_w:.6f} <= max={con.cash_reserve_max:.6f}",
            )
        )

        # 3. Position freshness check
        as_of = parse_utc(snapshot.as_of_utc)
        pos_updated = parse_utc(snapshot.positions.updated_at_utc)
        age_hours = (as_of - pos_updated).total_seconds() / 3600.0
        checks.append(
            ConstraintCheckResult(
                name="position_freshness",
                passed=0.0 <= age_hours <= self.config.max_position_staleness_hours,
                actual=age_hours,
                limit=self.config.max_position_staleness_hours,
                slack=self.config.max_position_staleness_hours - age_hours,
                unit="hours",
                severity="hard",
                message=(
                    f"position age {age_hours:.2f}h <= max {self.config.max_position_staleness_hours:.2f}h"
                ),
            )
        )

        # 4. Per-strategy weight bounds, eligibility, capacity, and ADV participation
        excluded_map: Dict[str, str] = {}
        if estimation is not None:
            excluded_map = dict(estimation.excluded_reasons)
        elif isinstance(decision.inputs_summary, dict):
            excluded_map = dict(decision.inputs_summary.get("excluded_reasons", {}))

        for sid in all_ids:
            strat = strat_map[sid]
            w = target_w[sid]
            w0 = cur_w[sid]
            is_excluded = (not strat.eligible) or (sid in excluded_map)

            # Non-negativity
            checks.append(
                ConstraintCheckResult(
                    name="strategy_non_negative",
                    passed=w >= -tol,
                    actual=w,
                    limit=0.0,
                    slack=w,
                    unit="weight",
                    severity="hard",
                    scope=sid,
                    message=f"{sid}: weight={w:.6f} >= 0",
                )
            )

            if is_excluded:
                # Must not increase allocation to ineligible/excluded strategy
                checks.append(
                    ConstraintCheckResult(
                        name="ineligible_no_new_capital",
                        passed=w <= w0 + tol,
                        actual=w,
                        limit=w0,
                        slack=w0 - w,
                        unit="weight",
                        severity="hard",
                        scope=sid,
                        message=f"{sid} (excluded): target={w:.6f} <= current={w0:.6f}",
                    )
                )
            else:
                lo = con.strategy_min_weights.get(
                    sid,
                    strat.min_weight if strat.min_weight is not None else con.default_min_weight,
                )
                hi = con.strategy_max_weights.get(
                    sid,
                    strat.max_weight if strat.max_weight is not None else con.default_max_weight,
                )
                checks.append(
                    ConstraintCheckResult(
                        name="strategy_min_weight",
                        passed=w >= lo - tol,
                        actual=w,
                        limit=lo,
                        slack=w - lo,
                        unit="weight",
                        severity="hard",
                        scope=sid,
                        message=f"{sid}: weight={w:.6f} >= min={lo:.6f}",
                    )
                )
                checks.append(
                    ConstraintCheckResult(
                        name="strategy_max_weight",
                        passed=w <= hi + tol,
                        actual=w,
                        limit=hi,
                        slack=hi - w,
                        unit="weight",
                        severity="hard",
                        scope=sid,
                        message=f"{sid}: weight={w:.6f} <= max={hi:.6f}",
                    )
                )

            # Capacity limit
            notional = w * nav
            cap_usd = strat.exposure.max_capacity_usd
            checks.append(
                ConstraintCheckResult(
                    name="strategy_capacity_limit",
                    passed=notional <= cap_usd + tol * nav,
                    actual=notional,
                    limit=cap_usd,
                    slack=cap_usd - notional,
                    unit="USD",
                    severity="hard",
                    scope=sid,
                    message=f"{sid}: notional=${notional:,.0f} <= cap=${cap_usd:,.0f}",
                )
            )

            # ADV liquidity participation limit
            trade_usd = abs(w - w0) * nav
            max_trade_usd = con.max_adv_participation * strat.exposure.adv_usd
            checks.append(
                ConstraintCheckResult(
                    name="strategy_adv_liquidity_limit",
                    passed=trade_usd <= max_trade_usd + tol * nav,
                    actual=trade_usd,
                    limit=max_trade_usd,
                    slack=max_trade_usd - trade_usd,
                    unit="USD",
                    severity="hard",
                    scope=sid,
                    message=f"{sid}: trade=${trade_usd:,.0f} <= max_adv_trade=${max_trade_usd:,.0f}",
                )
            )

        # 5. Group exposure caps
        group_totals: Dict[str, float] = {}
        for sid in all_ids:
            grp = strat_map[sid].exposure.strategy_group
            group_totals[grp] = group_totals.get(grp, 0.0) + target_w[sid]

        for grp, cap in sorted(con.group_caps.items()):
            actual_grp = group_totals.get(grp, 0.0)
            checks.append(
                ConstraintCheckResult(
                    name="group_exposure_cap",
                    passed=actual_grp <= cap + tol,
                    actual=actual_grp,
                    limit=cap,
                    slack=cap - actual_grp,
                    unit="weight",
                    severity="hard",
                    scope=grp,
                    message=f"group '{grp}': weight={actual_grp:.6f} <= cap={cap:.6f}",
                )
            )

        # 6. Gross leverage cap
        gross_lev = sum(
            strat_map[sid].exposure.gross_exposure * target_w[sid] for sid in all_ids
        )
        checks.append(
            ConstraintCheckResult(
                name="gross_leverage_cap",
                passed=gross_lev <= con.max_gross_leverage + tol,
                actual=gross_lev,
                limit=con.max_gross_leverage,
                slack=con.max_gross_leverage - gross_lev,
                unit="leverage",
                severity="hard",
                message=f"gross_leverage={gross_lev:.6f} <= max={con.max_gross_leverage:.6f}",
            )
        )

        # 7. One-way turnover limit
        one_way_to = 0.5 * sum(abs(target_w[sid] - cur_w[sid]) for sid in all_ids)
        checks.append(
            ConstraintCheckResult(
                name="one_way_turnover_limit",
                passed=one_way_to <= con.max_one_way_turnover + tol,
                actual=one_way_to,
                limit=con.max_one_way_turnover,
                slack=con.max_one_way_turnover - one_way_to,
                unit="turnover",
                severity="hard",
                message=(
                    f"one_way_turnover={one_way_to:.6f} <= max={con.max_one_way_turnover:.6f}"
                ),
            )
        )

        # 8. Portfolio and stressed volatility caps (independently computed if estimation provided)
        if estimation is not None:
            w_vec = [target_w[sid] for sid in estimation.strategy_ids]
            port_vol = math.sqrt(max(0.0, quad_form(w_vec, estimation.covariance_annual)))
            stress_vol = math.sqrt(
                max(0.0, quad_form(w_vec, estimation.stressed_covariance_annual))
            )
        else:
            port_vol = decision.objective.portfolio_volatility_annual
            stress_vol = decision.objective.stressed_volatility_annual

        if con.max_annualized_volatility is not None:
            checks.append(
                ConstraintCheckResult(
                    name="portfolio_volatility_cap",
                    passed=port_vol <= con.max_annualized_volatility + tol,
                    actual=port_vol,
                    limit=con.max_annualized_volatility,
                    slack=con.max_annualized_volatility - port_vol,
                    unit="annualized_vol",
                    severity="hard",
                    message=(
                        f"portfolio_vol={port_vol:.6f} <= max={con.max_annualized_volatility:.6f}"
                    ),
                )
            )

        if con.max_stressed_annualized_volatility is not None:
            checks.append(
                ConstraintCheckResult(
                    name="stressed_volatility_cap",
                    passed=stress_vol <= con.max_stressed_annualized_volatility + tol,
                    actual=stress_vol,
                    limit=con.max_stressed_annualized_volatility,
                    slack=con.max_stressed_annualized_volatility - stress_vol,
                    unit="annualized_vol",
                    severity="hard",
                    message=(
                        f"stressed_vol={stress_vol:.6f} <= "
                        f"max={con.max_stressed_annualized_volatility:.6f}"
                    ),
                )
            )

        hard_violations = tuple(c for c in checks if not c.passed and c.severity == "hard")
        warnings = tuple(c for c in checks if not c.passed and c.severity != "hard")
        return FeasibilityReport(
            feasible=(len(hard_violations) == 0),
            hard_violations=hard_violations,
            warnings=warnings,
            checks=tuple(checks),
            allocation_sum=invested_sum,
            cash_weight=cash_w,
            reconciliation_error=rec_err,
        )
