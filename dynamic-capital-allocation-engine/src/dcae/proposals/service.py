"""Rebalance proposal generation, feasibility reporting, and approval workflow."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from dcae.allocators.baselines import (
    EqualWeightAllocator,
    InverseVolatilityAllocator,
    RiskBudgetAllocator,
)
from dcae.allocators.constrained import ConstrainedAllocator
from dcae.audit.store import SQLiteAuditStore
from dcae.domain.models import (
    AllocationDecision,
    AllocationStatus,
    EngineConfig,
    FeasibilityReport,
    FundSnapshot,
    ProposalStatus,
    format_utc,
    parse_utc,
    sha256_hex,
)
from dcae.errors import (
    ApprovalRequiredError,
    DualControlViolationError,
    LiveExecutionProhibitedError,
    ProposalStateError,
    ValidationError,
)
from dcae.estimators.point_in_time import PointInTimeEstimator
from dcae.snapshots.publisher import SnapshotPublisher
from dcae.verification.verifier import ConstraintVerifier


@dataclass(frozen=True)
class RebalanceTrade:
    strategy_id: str
    current_weight: float
    target_weight: float
    delta_weight: float
    current_notional_usd: float
    target_notional_usd: float
    delta_notional_usd: float
    estimated_cost_usd: float
    adv_utilization: float
    action: str  # "BUY" | "SELL" | "HOLD"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "current_weight": float(self.current_weight),
            "target_weight": float(self.target_weight),
            "delta_weight": float(self.delta_weight),
            "current_notional_usd": float(self.current_notional_usd),
            "target_notional_usd": float(self.target_notional_usd),
            "delta_notional_usd": float(self.delta_notional_usd),
            "estimated_cost_usd": float(self.estimated_cost_usd),
            "adv_utilization": float(self.adv_utilization),
            "action": self.action,
        }


@dataclass(frozen=True)
class RebalanceProposal:
    proposal_id: str
    idempotency_key: str
    status: ProposalStatus
    proposer_id: str
    reviewer_id: Optional[str]
    review_notes: str
    created_at_utc: str
    updated_at_utc: str
    expires_at_utc: str
    decision: AllocationDecision
    feasibility_report: FeasibilityReport
    trades: Tuple[RebalanceTrade, ...]
    cash_delta_usd: float
    target_cash_usd: float
    requires_manual_approval: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "idempotency_key": self.idempotency_key,
            "status": self.status.value,
            "proposer_id": self.proposer_id,
            "reviewer_id": self.reviewer_id,
            "review_notes": self.review_notes,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "expires_at_utc": self.expires_at_utc,
            "decision": self.decision.to_dict(),
            "feasibility_report": self.feasibility_report.to_dict(),
            "trades": [t.to_dict() for t in self.trades],
            "cash_delta_usd": float(self.cash_delta_usd),
            "target_cash_usd": float(self.target_cash_usd),
            "requires_manual_approval": self.requires_manual_approval,
        }


class OperationalMetrics:
    """In-memory operational health, failure, and decision metrics."""

    def __init__(self) -> None:
        self.proposals_created = 0
        self.proposals_approved = 0
        self.proposals_rejected = 0
        self.paper_executions = 0
        self.fallback_count = 0
        self.infeasible_count = 0
        self.validation_failures = 0
        self.decision_latencies_ms: List[float] = []

    def record_latency_ms(self, ms: float) -> None:
        self.decision_latencies_ms.append(ms)

    def snapshot(self) -> Dict[str, Any]:
        lats = sorted(self.decision_latencies_ms)
        count = len(lats)
        p50 = lats[count // 2] if count else 0.0
        p95 = lats[min(count - 1, int(0.95 * count))] if count else 0.0
        return {
            "proposals_created": self.proposals_created,
            "proposals_approved": self.proposals_approved,
            "proposals_rejected": self.proposals_rejected,
            "paper_executions": self.paper_executions,
            "fallback_count": self.fallback_count,
            "infeasible_count": self.infeasible_count,
            "validation_failures": self.validation_failures,
            "decision_count": count,
            "decision_latency_p50_ms": round(p50, 3),
            "decision_latency_p95_ms": round(p95, 3),
        }


class ProposalService:
    """Orchestrates snapshot -> estimate -> allocate -> verify -> propose -> review workflow."""

    def __init__(
        self,
        config: EngineConfig,
        audit_store: SQLiteAuditStore,
    ) -> None:
        config.validate()
        self.config = config
        self.audit_store = audit_store
        self.publisher = SnapshotPublisher(config)
        self.estimator = PointInTimeEstimator(config)
        self.verifier = ConstraintVerifier(config)
        self.metrics = OperationalMetrics()

    def _get_allocator(self, allocator_name: str):
        if allocator_name == "equal_weight":
            return EqualWeightAllocator(self.config)
        if allocator_name == "inverse_volatility":
            return InverseVolatilityAllocator(self.config)
        if allocator_name == "risk_budget":
            return RiskBudgetAllocator(self.config)
        if allocator_name == "constrained_mv":
            return ConstrainedAllocator(self.config, name="constrained_mv")
        raise ValidationError(
            f"Unsupported allocator_name: {allocator_name}",
            context={"allocator_name": allocator_name},
        )

    def register_snapshot(
        self,
        snapshot: FundSnapshot,
        *,
        actor_id: str = "system",
        enforce_freshness: bool = True,
    ) -> Dict[str, Any]:
        manifest = self.publisher.validate_and_publish(
            snapshot,
            enforce_freshness=enforce_freshness,
        )
        self.audit_store.save_snapshot(snapshot)
        self.audit_store.append_event(
            event_type="SNAPSHOT_PUBLISHED",
            entity_id=snapshot.snapshot_id,
            actor_id=actor_id,
            payload=manifest.to_dict(),
        )
        return manifest.to_dict()

    def create_proposal(
        self,
        snapshot: FundSnapshot,
        *,
        proposer_id: str,
        allocator_name: str = "constrained_mv",
        idempotency_key: Optional[str] = None,
        now_utc: Optional[str] = None,
        enforce_freshness: bool = True,
    ) -> RebalanceProposal:
        if not proposer_id or not proposer_id.strip():
            raise ValidationError("proposer_id must be non-empty")

        t0 = time.perf_counter()
        try:
            # 1. Validate snapshot & freshness
            manifest = self.publisher.validate_and_publish(
                snapshot,
                enforce_freshness=enforce_freshness,
            )
            self.audit_store.save_snapshot(snapshot)

            idem = idempotency_key or sha256_hex(
                {
                    "snapshot_id": snapshot.snapshot_id,
                    "snapshot_hash": manifest.content_hash,
                    "allocator_name": allocator_name,
                    "config_hash": self.config.config_hash(),
                }
            )[:24]

            # Idempotent return if identical request already created
            existing = self.audit_store.load_proposal_by_idempotency_key(idem)
            if existing is not None:
                return self._proposal_from_dict(existing)

            # 2. Point-in-time estimation
            estimation = self.estimator.estimate(snapshot)

            # 3. Run allocator
            allocator = self._get_allocator(allocator_name)
            decision = allocator.allocate(snapshot, estimation=estimation)

            # 4. Independent post-solve constraint verification
            report = self.verifier.verify(
                snapshot=snapshot,
                decision=decision,
                estimation=estimation,
            )

            # Fail-safe: if constrained_mv returned OPTIMAL/APPROX yet independent verifier
            # finds a hard violation, force safe fallback!
            if (
                allocator_name == "constrained_mv"
                and not report.feasible
                and decision.fallback_applied is None
            ):
                fallback_allocator = ConstrainedAllocator(self.config, name=allocator_name)
                cur_w, cur_cash = snapshot.positions.current_weights(list(estimation.strategy_ids))
                reasons = tuple(
                    f"verifier_violation:{v.name}:{v.scope}:{v.message}"
                    for v in report.hard_violations
                )
                decision = fallback_allocator._fallback_decision(
                    snapshot=snapshot,
                    estimation=estimation,
                    current_weights=cur_w,
                    current_cash=cur_cash,
                    reasons=reasons,
                    diagnostics={"stage": "independent_verifier_override"},
                )
                report = self.verifier.verify(
                    snapshot=snapshot,
                    decision=decision,
                    estimation=estimation,
                )

            decision = replace(decision, feasibility_report=report)

            # 5. Build per-strategy rebalance trades
            nav = snapshot.positions.total_nav_usd
            strat_map = {s.strategy_id: s for s in snapshot.strategies}
            trades: List[RebalanceTrade] = []
            for strat in snapshot.strategies:
                sid = strat.strategy_id
                cw = decision.current_weights.get(sid, 0.0)
                tw = decision.target_weights.get(sid, 0.0)
                dw = tw - cw
                c_usd = cw * nav
                t_usd = tw * nav
                d_usd = dw * nav
                bps = strat_map[sid].exposure.transaction_cost_bps
                cost_usd = abs(d_usd) * (bps / 10000.0)
                adv_util = abs(d_usd) / strat.exposure.adv_usd
                if dw > 1e-6:
                    action = "BUY"
                elif dw < -1e-6:
                    action = "SELL"
                else:
                    action = "HOLD"
                trades.append(
                    RebalanceTrade(
                        strategy_id=sid,
                        current_weight=cw,
                        target_weight=tw,
                        delta_weight=dw,
                        current_notional_usd=c_usd,
                        target_notional_usd=t_usd,
                        delta_notional_usd=d_usd,
                        estimated_cost_usd=cost_usd,
                        adv_utilization=adv_util,
                        action=action,
                    )
                )

            target_cash_usd = decision.cash_weight * nav
            cash_delta_usd = target_cash_usd - snapshot.positions.cash_usd

            now_dt = parse_utc(now_utc) if now_utc else parse_utc(snapshot.as_of_utc)
            exp_dt = now_dt + timedelta(minutes=self.config.proposal_ttl_minutes)
            proposal_id = f"prop_{sha256_hex({'idem': idem, 'dec': decision.decision_id})[:14]}"

            proposal = RebalanceProposal(
                proposal_id=proposal_id,
                idempotency_key=idem,
                status=ProposalStatus.PENDING_APPROVAL,
                proposer_id=proposer_id,
                reviewer_id=None,
                review_notes="",
                created_at_utc=format_utc(now_dt),
                updated_at_utc=format_utc(now_dt),
                expires_at_utc=format_utc(exp_dt),
                decision=decision,
                feasibility_report=report,
                trades=tuple(trades),
                cash_delta_usd=cash_delta_usd,
                target_cash_usd=target_cash_usd,
                requires_manual_approval=True,
            )

            self.audit_store.save_proposal(proposal.to_dict())
            self.audit_store.append_event(
                event_type="PROPOSAL_CREATED",
                entity_id=proposal.proposal_id,
                actor_id=proposer_id,
                occurred_at_utc=proposal.created_at_utc,
                payload={
                    "proposal_id": proposal.proposal_id,
                    "decision_id": decision.decision_id,
                    "snapshot_id": snapshot.snapshot_id,
                    "snapshot_hash": manifest.content_hash,
                    "config_hash": self.config.config_hash(),
                    "allocator_name": allocator_name,
                    "allocation_status": decision.status.value,
                    "fallback_applied": decision.fallback_applied,
                    "feasible": report.feasible,
                    "net_objective": decision.objective.net_objective,
                    "one_way_turnover": decision.objective.one_way_turnover,
                },
            )

            self.metrics.proposals_created += 1
            if decision.fallback_applied is not None:
                self.metrics.fallback_count += 1
            if not report.feasible:
                self.metrics.infeasible_count += 1
            return proposal
        except Exception:
            self.metrics.validation_failures += 1
            raise
        finally:
            self.metrics.record_latency_ms((time.perf_counter() - t0) * 1000.0)

    def review_proposal(
        self,
        proposal_id: str,
        *,
        reviewer_id: str,
        approve: bool,
        notes: str = "",
        reviewed_at_utc: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not reviewer_id or not reviewer_id.strip():
            raise ValidationError("reviewer_id must be non-empty")
        raw = self.audit_store.load_proposal(proposal_id)
        if raw is None:
            raise ProposalStateError(
                f"Proposal not found: {proposal_id}",
                context={"proposal_id": proposal_id},
            )
        if raw["status"] != ProposalStatus.PENDING_APPROVAL.value:
            raise ProposalStateError(
                f"Proposal {proposal_id} is in status {raw['status']}, cannot review",
                context={"proposal_id": proposal_id, "status": raw["status"]},
            )
        if self.config.enforce_dual_control and reviewer_id == raw["proposer_id"]:
            raise DualControlViolationError(
                "Dual-control policy forbids proposer from self-approving/reviewing",
                context={
                    "proposal_id": proposal_id,
                    "proposer_id": raw["proposer_id"],
                    "reviewer_id": reviewer_id,
                },
            )

        now_dt = parse_utc(reviewed_at_utc) if reviewed_at_utc else datetime.now(timezone.utc)
        exp_dt = parse_utc(raw["expires_at_utc"])
        if approve and now_dt > exp_dt:
            raw["status"] = ProposalStatus.EXPIRED.value
            raw["updated_at_utc"] = format_utc(now_dt)
            raw["review_notes"] = "Expired before approval"
            self.audit_store.save_proposal(raw)
            self.audit_store.append_event(
                event_type="PROPOSAL_EXPIRED",
                entity_id=proposal_id,
                actor_id=reviewer_id,
                occurred_at_utc=raw["updated_at_utc"],
                payload={"expires_at_utc": raw["expires_at_utc"]},
            )
            raise ProposalStateError(
                f"Proposal {proposal_id} expired at {raw['expires_at_utc']}",
                context={"proposal_id": proposal_id, "expires_at_utc": raw["expires_at_utc"]},
            )

        new_status = ProposalStatus.APPROVED if approve else ProposalStatus.REJECTED
        raw["status"] = new_status.value
        raw["reviewer_id"] = reviewer_id
        raw["review_notes"] = notes
        raw["updated_at_utc"] = format_utc(now_dt)
        self.audit_store.save_proposal(raw)
        self.audit_store.append_event(
            event_type="PROPOSAL_APPROVED" if approve else "PROPOSAL_REJECTED",
            entity_id=proposal_id,
            actor_id=reviewer_id,
            occurred_at_utc=raw["updated_at_utc"],
            payload={"reviewer_id": reviewer_id, "notes": notes},
        )
        if approve:
            self.metrics.proposals_approved += 1
        else:
            self.metrics.proposals_rejected += 1
        return raw

    def require_execution_authorized(
        self,
        proposal_id: str,
        *,
        mode: str = "paper",
    ) -> Dict[str, Any]:
        if mode != "paper":
            raise LiveExecutionProhibitedError(
                "Automated live capital reallocation is strictly prohibited in MVP",
                context={"requested_mode": mode, "proposal_id": proposal_id},
            )
        if not self.config.paper_execution_enabled:
            raise ApprovalRequiredError(
                "Paper execution is disabled by configuration policy",
                context={"proposal_id": proposal_id},
            )
        raw = self.audit_store.load_proposal(proposal_id)
        if raw is None:
            raise ProposalStateError(
                f"Proposal not found: {proposal_id}",
                context={"proposal_id": proposal_id},
            )
        if raw["status"] not in (
            ProposalStatus.APPROVED.value,
            ProposalStatus.PAPER_EXECUTED.value,
        ):
            raise ApprovalRequiredError(
                f"Proposal {proposal_id} requires explicit APPROVED status before paper execution "
                f"(current status: {raw['status']})",
                context={"proposal_id": proposal_id, "status": raw["status"]},
            )
        return raw

    @staticmethod
    def _proposal_from_dict(raw: Dict[str, Any]) -> RebalanceProposal:
        from dcae.domain.models import (
            ConstraintCheckResult,
            ObjectiveBreakdown,
        )

        dec_raw = raw["decision"]
        obj_raw = dec_raw["objective"]
        obj = ObjectiveBreakdown(**obj_raw)
        fr_raw = raw["feasibility_report"]
        checks = tuple(ConstraintCheckResult(**c) for c in fr_raw["checks"])
        hard_v = tuple(ConstraintCheckResult(**c) for c in fr_raw["hard_violations"])
        warns = tuple(ConstraintCheckResult(**c) for c in fr_raw["warnings"])
        report = FeasibilityReport(
            feasible=bool(fr_raw["feasible"]),
            hard_violations=hard_v,
            warnings=warns,
            checks=checks,
            allocation_sum=float(fr_raw["allocation_sum"]),
            cash_weight=float(fr_raw["cash_weight"]),
            reconciliation_error=float(fr_raw["reconciliation_error"]),
        )
        decision = AllocationDecision(
            decision_id=str(dec_raw["decision_id"]),
            allocator_name=str(dec_raw["allocator_name"]),
            snapshot_id=str(dec_raw["snapshot_id"]),
            snapshot_hash=str(dec_raw["snapshot_hash"]),
            config_version=str(dec_raw["config_version"]),
            config_hash=str(dec_raw["config_hash"]),
            as_of_utc=str(dec_raw["as_of_utc"]),
            status=AllocationStatus(dec_raw["status"]),
            target_weights=dict(dec_raw["target_weights"]),
            cash_weight=float(dec_raw["cash_weight"]),
            current_weights=dict(dec_raw["current_weights"]),
            current_cash_weight=float(dec_raw["current_cash_weight"]),
            objective=obj,
            inputs_summary=dict(dec_raw["inputs_summary"]),
            diagnostics=dict(dec_raw["diagnostics"]),
            infeasibility_reasons=tuple(dec_raw.get("infeasibility_reasons", [])),
            fallback_applied=dec_raw.get("fallback_applied"),
            feasibility_report=report,
        )
        trades = tuple(RebalanceTrade(**t) for t in raw["trades"])
        return RebalanceProposal(
            proposal_id=str(raw["proposal_id"]),
            idempotency_key=str(raw["idempotency_key"]),
            status=ProposalStatus(raw["status"]),
            proposer_id=str(raw["proposer_id"]),
            reviewer_id=raw.get("reviewer_id"),
            review_notes=str(raw.get("review_notes", "")),
            created_at_utc=str(raw["created_at_utc"]),
            updated_at_utc=str(raw["updated_at_utc"]),
            expires_at_utc=str(raw["expires_at_utc"]),
            decision=decision,
            feasibility_report=report,
            trades=trades,
            cash_delta_usd=float(raw["cash_delta_usd"]),
            target_cash_usd=float(raw["target_cash_usd"]),
            requires_manual_approval=bool(raw.get("requires_manual_approval", True)),
        )
