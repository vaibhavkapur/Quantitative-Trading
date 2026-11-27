#!/usr/bin/env python3
"""End-to-end reproducible demonstration of DCAE (Milestones M1-M4)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dcae.allocators import (  # noqa: E402
    ConstrainedAllocator,
    EqualWeightAllocator,
    InverseVolatilityAllocator,
    RiskBudgetAllocator,
)
from dcae.audit import SQLiteAuditStore  # noqa: E402
from dcae.domain import EngineConfig, FundSnapshot  # noqa: E402
from dcae.errors import DualControlViolationError, LiveExecutionProhibitedError  # noqa: E402
from dcae.estimators import PointInTimeEstimator  # noqa: E402
from dcae.fixtures import build_baseline_snapshot  # noqa: E402
from dcae.paper import PaperPortfolioLedger  # noqa: E402
from dcae.proposals import ProposalService  # noqa: E402
from dcae.verification import ConstraintVerifier  # noqa: E402


def main() -> None:
    cfg = EngineConfig.from_json_file(str(ROOT / "config" / "default_config.json"))
    fixture_path = ROOT / "data" / "fixtures" / "baseline_snapshot.json"
    if fixture_path.exists():
        snapshot = FundSnapshot.from_dict(json.loads(fixture_path.read_text(encoding="utf-8")))
    else:
        snapshot = build_baseline_snapshot()

    db_path = ROOT / "reports" / "demo_audit.sqlite3"
    if db_path.exists():
        db_path.unlink()
    store = SQLiteAuditStore(str(db_path))
    service = ProposalService(cfg, store)

    print("=== DCAE End-to-End Demo (M1-M4) ===")
    manifest = service.register_snapshot(snapshot, actor_id="demo_ingest")
    print(
        f"[M1] Published snapshot {manifest['snapshot_id']} "
        f"(hash={manifest['content_hash'][:12]}..., age={manifest['position_age_hours']:.2f}h)"
    )

    estimator = PointInTimeEstimator(cfg)
    est = estimator.estimate(snapshot)
    verifier = ConstraintVerifier(cfg)

    print("\n[M2] Comparing baselines vs constrained allocator on identical PIT inputs:")
    for alloc in (
        EqualWeightAllocator(cfg),
        InverseVolatilityAllocator(cfg),
        RiskBudgetAllocator(cfg),
        ConstrainedAllocator(cfg),
    ):
        dec = alloc.allocate(snapshot, estimation=est)
        rep = verifier.verify(snapshot=snapshot, decision=dec, estimation=est)
        print(
            f"  - {alloc.name:18s} | status={dec.status.value:16s} | "
            f"feasible={str(rep.feasible):5s} | cash={dec.cash_weight:.3f} | "
            f"exp_ret={dec.objective.expected_return_annual:.3%} | "
            f"vol={dec.objective.portfolio_volatility_annual:.3%} | "
            f"turnover={dec.objective.one_way_turnover:.3%} | "
            f"cost=${dec.objective.estimated_cost_usd:,.0f} | "
            f"net_obj={dec.objective.net_objective:.5f}"
        )

    print("\n[M3] Creating rebalance proposal & enforcing dual-control approval:")
    proposal = service.create_proposal(
        snapshot,
        proposer_id="pm_alice",
        allocator_name="constrained_mv",
        idempotency_key="demo_proposal_v1",
        now_utc=snapshot.as_of_utc,
    )
    print(
        f"  Proposal {proposal.proposal_id}: status={proposal.status.value}, "
        f"checks={len(proposal.feasibility_report.checks)}, "
        f"hard_violations={len(proposal.feasibility_report.hard_violations)}"
    )

    try:
        service.review_proposal(
            proposal.proposal_id,
            reviewer_id="pm_alice",
            approve=True,
            reviewed_at_utc=snapshot.as_of_utc,
        )
    except DualControlViolationError as err:
        print(f"  Dual-control guard blocked self-approval: {err.code}")

    approved = service.review_proposal(
        proposal.proposal_id,
        reviewer_id="risk_officer_bob",
        approve=True,
        notes="Verified feasibility report, liquidity limits, and stress vol.",
        reviewed_at_utc=snapshot.as_of_utc,
    )
    print(f"  Approved by {approved['reviewer_id']} -> status={approved['status']}")

    print("\n[M4] Executing approved proposal on paper portfolio ledger:")
    ledger = PaperPortfolioLedger(
        initial_cash_usd=snapshot.positions.cash_usd,
        strategy_notional_usd=dict(snapshot.positions.strategy_notional_usd),
        as_of_utc=snapshot.as_of_utc,
    )
    try:
        ledger.apply_approved_proposal(
            proposal_service=service,
            proposal_id=proposal.proposal_id,
            executed_by="ops_carol",
            mode="live",
        )
    except LiveExecutionProhibitedError as err:
        print(f"  Live execution guard blocked live mode: {err.code}")

    receipt1 = ledger.apply_approved_proposal(
        proposal_service=service,
        proposal_id=proposal.proposal_id,
        executed_by="ops_carol",
        idempotency_key="demo_exec_1",
        executed_at_utc=snapshot.as_of_utc,
        mode="paper",
    )
    receipt2 = ledger.apply_approved_proposal(
        proposal_service=service,
        proposal_id=proposal.proposal_id,
        executed_by="ops_carol",
        idempotency_key="demo_exec_1",
        executed_at_utc=snapshot.as_of_utc,
        mode="paper",
    )
    print(
        f"  Paper receipt {receipt1.execution_id}: nav=${receipt1.total_nav_usd:,.2f}, "
        f"cash=${receipt1.cash_usd:,.2f}, cost=${receipt1.total_cost_usd:,.2f}"
    )
    print(f"  Idempotent replay confirmed: replay={receipt2.idempotent_replay}")

    chain = store.verify_chain()
    jsonl_count = store.export_jsonl(str(ROOT / "reports" / "demo_audit.jsonl"))
    print(
        f"\n[Audit] Hash chain verified: valid={chain['valid']}, "
        f"events={chain['event_count']}, exported_jsonl={jsonl_count}"
    )
    store.close()


if __name__ == "__main__":
    main()
