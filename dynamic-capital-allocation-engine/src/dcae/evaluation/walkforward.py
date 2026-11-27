"""Strictly chronological point-in-time walk-forward evaluation across allocators.

Evaluates equal-weight, inverse-volatility, risk-budget, and constrained
allocators over identical periods, identical transaction cost schedules,
identical initial capital, and identical cash reserve rules.

Explicitly reports fallback events, constraint violations, and failed runs
alongside net performance diagnostics (never claiming live performance).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from dcae.allocators.baselines import (
    EqualWeightAllocator,
    InverseVolatilityAllocator,
    RiskBudgetAllocator,
)
from dcae.allocators.constrained import ConstrainedAllocator
from dcae.domain.models import (
    AllocationStatus,
    EngineConfig,
    FundSnapshot,
    PositionSnapshot,
    Provenance,
    ReturnObservation,
    StrategyExposure,
    StrategySnapshot,
)
from dcae.estimators.point_in_time import PointInTimeEstimator
from dcae.paper.ledger import PaperPortfolioLedger
from dcae.verification.verifier import ConstraintVerifier


@dataclass(frozen=True)
class WalkForwardStrategySpec:
    """Static metadata + full chronological return history for walk-forward simulation."""

    strategy_id: str
    name: str
    exposure: StrategyExposure
    returns: Tuple[ReturnObservation, ...]
    eligible: bool = True
    min_weight: Optional[float] = None
    max_weight: Optional[float] = None
    risk_budget: Optional[float] = None


@dataclass(frozen=True)
class AllocatorEvalSummary:
    allocator_name: str
    periods_evaluated: int
    rebalance_steps: int
    initial_nav_usd: float
    final_nav_usd: float
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_diagnostic: float
    max_drawdown: float
    calmar_diagnostic: float
    avg_one_way_turnover: float
    total_transaction_cost_usd: float
    avg_cash_weight: float
    fallback_count: int
    verifier_violation_count: int
    failed_step_count: int
    stress_active_steps: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allocator_name": self.allocator_name,
            "periods_evaluated": self.periods_evaluated,
            "rebalance_steps": self.rebalance_steps,
            "initial_nav_usd": round(self.initial_nav_usd, 2),
            "final_nav_usd": round(self.final_nav_usd, 2),
            "cumulative_return": round(self.cumulative_return, 6),
            "annualized_return": round(self.annualized_return, 6),
            "annualized_volatility": round(self.annualized_volatility, 6),
            "sharpe_diagnostic": round(self.sharpe_diagnostic, 4),
            "max_drawdown": round(self.max_drawdown, 6),
            "calmar_diagnostic": round(self.calmar_diagnostic, 4),
            "avg_one_way_turnover": round(self.avg_one_way_turnover, 6),
            "total_transaction_cost_usd": round(self.total_transaction_cost_usd, 2),
            "avg_cash_weight": round(self.avg_cash_weight, 6),
            "fallback_count": self.fallback_count,
            "verifier_violation_count": self.verifier_violation_count,
            "failed_step_count": self.failed_step_count,
            "stress_active_steps": self.stress_active_steps,
        }


class WalkForwardEvaluator:
    """Chronological walk-forward comparator across multiple allocators."""

    def __init__(self, base_config: EngineConfig) -> None:
        base_config.validate()
        self.base_config = base_config

    def evaluate(
        self,
        *,
        strategies: Sequence[WalkForwardStrategySpec],
        provenance: Provenance,
        initial_nav_usd: float = 10_000_000.0,
        warmup_periods: int = 30,
        rebalance_every: int = 5,
        lookback_window: Optional[int] = 60,
        allocator_names: Sequence[str] = (
            "equal_weight",
            "inverse_volatility",
            "risk_budget",
            "constrained_mv",
            "constrained_mv_stressed",
        ),
    ) -> Dict[str, Any]:
        if not strategies:
            raise ValueError("strategies must be non-empty")
        total_len = min(len(s.returns) for s in strategies)
        if total_len <= warmup_periods + 2:
            raise ValueError(
                f"Need more than warmup_periods+2 ({warmup_periods + 2}) observations; got {total_len}"
            )

        rebalance_indices = list(range(warmup_periods, total_len - 1, rebalance_every))
        results: Dict[str, AllocatorEvalSummary] = {}
        equity_curves: Dict[str, List[float]] = {}
        step_logs: Dict[str, List[Dict[str, Any]]] = {}

        for alloc_name in allocator_names:
            cfg = self._config_for_allocator(alloc_name)
            summary, curve, logs = self._run_single_allocator(
                allocator_name=alloc_name,
                config=cfg,
                strategies=strategies,
                provenance=provenance,
                initial_nav_usd=initial_nav_usd,
                warmup_periods=warmup_periods,
                rebalance_indices=set(rebalance_indices),
                total_len=total_len,
                lookback_window=lookback_window,
            )
            results[alloc_name] = summary
            equity_curves[alloc_name] = curve
            step_logs[alloc_name] = logs

        return {
            "disclaimer": (
                "Synthetic/historical walk-forward diagnostic only; never infer live "
                "performance or target Sharpe ratio from simulated evaluation."
            ),
            "provenance": provenance.to_dict(),
            "evaluation_window": {
                "total_observations": total_len,
                "warmup_periods": warmup_periods,
                "evaluated_periods": total_len - 1 - warmup_periods,
                "rebalance_every": rebalance_every,
                "rebalance_count": len(rebalance_indices),
                "lookback_window": lookback_window,
                "initial_nav_usd": initial_nav_usd,
                "start_utc": strategies[0].returns[warmup_periods].event_at_utc,
                "end_utc": strategies[0].returns[total_len - 1].event_at_utc,
            },
            "summaries": {k: v.to_dict() for k, v in results.items()},
            "step_diagnostics_sample": {
                k: v[:5] for k, v in step_logs.items()
            },
            "final_nav_by_allocator": {
                k: round(curve[-1], 2) for k, curve in equity_curves.items()
            },
        }

    def _config_for_allocator(self, allocator_name: str) -> EngineConfig:
        if allocator_name == "constrained_mv_stressed":
            return replace(self.base_config, correlation_stress_enabled=True)
        return self.base_config

    def _make_allocator(self, allocator_name: str, cfg: EngineConfig):
        if allocator_name == "equal_weight":
            return EqualWeightAllocator(cfg)
        if allocator_name == "inverse_volatility":
            return InverseVolatilityAllocator(cfg)
        if allocator_name == "risk_budget":
            return RiskBudgetAllocator(cfg)
        if allocator_name in ("constrained_mv", "constrained_mv_stressed"):
            return ConstrainedAllocator(cfg, name=allocator_name)
        raise ValueError(f"Unknown allocator_name: {allocator_name}")

    def _run_single_allocator(
        self,
        *,
        allocator_name: str,
        config: EngineConfig,
        strategies: Sequence[WalkForwardStrategySpec],
        provenance: Provenance,
        initial_nav_usd: float,
        warmup_periods: int,
        rebalance_indices: set[int],
        total_len: int,
        lookback_window: Optional[int],
    ) -> Tuple[AllocatorEvalSummary, List[float], List[Dict[str, Any]]]:
        allocator = self._make_allocator(allocator_name, config)
        estimator = PointInTimeEstimator(config)
        verifier = ConstraintVerifier(config)

        start_ts = strategies[0].returns[warmup_periods].available_at_utc
        ledger = PaperPortfolioLedger(
            initial_cash_usd=initial_nav_usd,
            strategy_notional_usd={s.strategy_id: 0.0 for s in strategies},
            as_of_utc=start_ts,
        )

        nav_curve: List[float] = [initial_nav_usd]
        daily_returns: List[float] = []
        turnovers: List[float] = []
        cash_weights: List[float] = []
        fallback_count = 0
        verifier_violation_count = 0
        failed_step_count = 0
        stress_active_steps = 0
        logs: List[Dict[str, Any]] = []

        # Track strategy peaks up to time t for strictly point-in-time drawdown calculation
        strat_peaks: Dict[str, float] = {
            s.strategy_id: max(obs.nav_usd for obs in s.returns[: warmup_periods + 1])
            for s in strategies
        }

        for t in range(warmup_periods, total_len - 1):
            as_of_ts = strategies[0].returns[t].available_at_utc

            # Update point-in-time peaks at time t
            for s in strategies:
                nav_t = s.returns[t].nav_usd
                if nav_t > strat_peaks[s.strategy_id]:
                    strat_peaks[s.strategy_id] = nav_t

            if t in rebalance_indices:
                start_idx = max(0, t + 1 - lookback_window) if lookback_window else 0
                strat_snaps: List[StrategySnapshot] = []
                for s in strategies:
                    hist = s.returns[start_idx : t + 1]
                    peak = strat_peaks[s.strategy_id]
                    cur_dd = (s.returns[t].nav_usd / peak - 1.0) if peak > 0.0 else 0.0
                    cur_dd = min(0.0, max(-1.0, cur_dd))
                    strat_snaps.append(
                        StrategySnapshot(
                            strategy_id=s.strategy_id,
                            name=s.name,
                            currency="USD",
                            eligible=s.eligible,
                            eligibility_reason="ok" if s.eligible else "disabled",
                            exposure=s.exposure,
                            returns=hist,
                            min_weight=s.min_weight,
                            max_weight=s.max_weight,
                            risk_budget=s.risk_budget,
                            current_drawdown=cur_dd,
                        )
                    )

                pos_snap = PositionSnapshot(
                    as_of_utc=as_of_ts,
                    updated_at_utc=as_of_ts,
                    currency="USD",
                    total_nav_usd=ledger.total_nav_usd,
                    cash_usd=ledger.cash_usd,
                    strategy_notional_usd=dict(ledger.strategy_notional_usd),
                )
                fund_snap = FundSnapshot(
                    snapshot_id=f"wf_{allocator_name}_t{t}",
                    as_of_utc=as_of_ts,
                    published_at_utc=as_of_ts,
                    provenance=provenance,
                    positions=pos_snap,
                    strategies=tuple(strat_snaps),
                )

                try:
                    est = estimator.estimate(fund_snap)
                    if est.correlation_stress_active:
                        stress_active_steps += 1
                    dec = allocator.allocate(fund_snap, estimation=est)
                    rep = verifier.verify(snapshot=fund_snap, decision=dec, estimation=est)
                    if not rep.feasible:
                        verifier_violation_count += 1
                    if dec.status in (
                        AllocationStatus.FALLBACK_NO_CHANGE,
                        AllocationStatus.FALLBACK_CASH,
                    ):
                        fallback_count += 1

                    # Apply transaction cost & rebalance paper positions identically for all allocators
                    nav_before = ledger.total_nav_usd
                    tcost_usd = 0.0
                    new_notional: Dict[str, float] = {}
                    for s in strategies:
                        sid = s.strategy_id
                        cur_n = ledger.strategy_notional_usd.get(sid, 0.0)
                        tgt_n = dec.target_weights.get(sid, 0.0) * nav_before
                        tcost_usd += abs(tgt_n - cur_n) * (s.exposure.transaction_cost_bps / 10000.0)
                        new_notional[sid] = tgt_n

                    nav_after = max(0.0, nav_before - tcost_usd)
                    scale = (nav_after / nav_before) if nav_before > 0.0 else 0.0
                    ledger.strategy_notional_usd = {
                        sid: n * scale for sid, n in new_notional.items()
                    }
                    ledger.cash_usd = max(
                        0.0, nav_after - sum(ledger.strategy_notional_usd.values())
                    )
                    ledger.cumulative_transaction_cost_usd += tcost_usd
                    ledger.rebalance_count += 1
                    turnovers.append(dec.objective.one_way_turnover)
                    cash_weights.append(dec.cash_weight)
                    logs.append(
                        {
                            "t": t,
                            "as_of_utc": as_of_ts,
                            "status": dec.status.value,
                            "feasible": rep.feasible,
                            "turnover": round(dec.objective.one_way_turnover, 5),
                            "cost_usd": round(tcost_usd, 2),
                            "cash_weight": round(dec.cash_weight, 5),
                            "stress_active": est.correlation_stress_active,
                        }
                    )
                except Exception as exc:
                    failed_step_count += 1
                    logs.append(
                        {
                            "t": t,
                            "as_of_utc": as_of_ts,
                            "status": "FAILED",
                            "error": str(exc),
                        }
                    )

            # Realize next period return t+1 on held paper positions (strictly out-of-sample!)
            prev_nav = ledger.total_nav_usd
            next_returns = {s.strategy_id: s.returns[t + 1].period_return for s in strategies}
            next_ts = strategies[0].returns[t + 1].available_at_utc
            new_nav = ledger.mark_to_market(period_returns=next_returns, as_of_utc=next_ts)
            nav_curve.append(new_nav)
            daily_returns.append((new_nav / prev_nav - 1.0) if prev_nav > 0.0 else 0.0)

        ann = float(config.annualization_factor)
        n_days = len(daily_returns)
        cum_ret = (nav_curve[-1] / initial_nav_usd) - 1.0
        ann_ret = (nav_curve[-1] / initial_nav_usd) ** (ann / max(1, n_days)) - 1.0

        mean_d = sum(daily_returns) / max(1, n_days)
        var_d = sum((r - mean_d) ** 2 for r in daily_returns) / max(1, n_days - 1)
        ann_vol = math.sqrt(max(var_d, 0.0)) * math.sqrt(ann)
        sharpe = (ann_ret / ann_vol) if ann_vol > 1e-12 else 0.0

        # Max drawdown
        peak_nav = nav_curve[0]
        max_dd = 0.0
        for v in nav_curve:
            if v > peak_nav:
                peak_nav = v
            dd = (v / peak_nav) - 1.0 if peak_nav > 0.0 else 0.0
            if dd < max_dd:
                max_dd = dd
        calmar = (ann_ret / abs(max_dd)) if abs(max_dd) > 1e-12 else 0.0

        summary = AllocatorEvalSummary(
            allocator_name=allocator_name,
            periods_evaluated=n_days,
            rebalance_steps=len(rebalance_indices),
            initial_nav_usd=initial_nav_usd,
            final_nav_usd=nav_curve[-1],
            cumulative_return=cum_ret,
            annualized_return=ann_ret,
            annualized_volatility=ann_vol,
            sharpe_diagnostic=sharpe,
            max_drawdown=max_dd,
            calmar_diagnostic=calmar,
            avg_one_way_turnover=sum(turnovers) / max(1, len(turnovers)),
            total_transaction_cost_usd=ledger.cumulative_transaction_cost_usd,
            avg_cash_weight=sum(cash_weights) / max(1, len(cash_weights)),
            fallback_count=fallback_count,
            verifier_violation_count=verifier_violation_count,
            failed_step_count=failed_step_count,
            stress_active_steps=stress_active_steps,
        )
        return summary, nav_curve, logs
