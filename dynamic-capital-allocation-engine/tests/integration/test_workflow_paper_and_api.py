"""Integration tests for proposal workflow, approval controls, paper ledger, audit chain, API, and walk-forward."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dcae.api import DCAEServiceApp
from dcae.audit import SQLiteAuditStore
from dcae.domain import EngineConfig, ProposalStatus
from dcae.errors import (
    ApprovalRequiredError,
    AuditIntegrityError,
    DualControlViolationError,
    LiveExecutionProhibitedError,
    ProposalStateError,
)
from dcae.evaluation import WalkForwardEvaluator
from dcae.fixtures import build_baseline_snapshot, generate_walkforward_strategies
from dcae.paper import PaperPortfolioLedger
from dcae.proposals import ProposalService


class TestProposalPaperAndAuditIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = EngineConfig()
        self.snap = build_baseline_snapshot(seed=20260409, n_days=60)
        self.store = SQLiteAuditStore(":memory:")
        self.service = ProposalService(self.cfg, self.store)
        self.ledger = PaperPortfolioLedger(
            initial_cash_usd=self.snap.positions.cash_usd,
            strategy_notional_usd=dict(self.snap.positions.strategy_notional_usd),
            as_of_utc=self.snap.as_of_utc,
        )

    def tearDown(self) -> None:
        self.store.close()

    def test_full_approval_gated_paper_execution_and_idempotency(self) -> None:
        prop = self.service.create_proposal(
            self.snap,
            proposer_id="alice",
            allocator_name="constrained_mv",
            idempotency_key="idem_key_01",
            now_utc=self.snap.as_of_utc,
        )
        self.assertEqual(prop.status, ProposalStatus.PENDING_APPROVAL)
        self.assertTrue(prop.feasibility_report.feasible)

        # Idempotent proposal creation returns same proposal_id
        prop_dup = self.service.create_proposal(
            self.snap,
            proposer_id="alice",
            allocator_name="constrained_mv",
            idempotency_key="idem_key_01",
            now_utc=self.snap.as_of_utc,
        )
        self.assertEqual(prop.proposal_id, prop_dup.proposal_id)

        # Cannot execute unapproved proposal
        with self.assertRaises(ApprovalRequiredError):
            self.ledger.apply_approved_proposal(
                proposal_service=self.service,
                proposal_id=prop.proposal_id,
                executed_by="ops_1",
            )

        # Dual control blocks self-approval
        with self.assertRaises(DualControlViolationError):
            self.service.review_proposal(
                prop.proposal_id,
                reviewer_id="alice",
                approve=True,
                reviewed_at_utc=self.snap.as_of_utc,
            )

        # Independent reviewer approves
        approved = self.service.review_proposal(
            prop.proposal_id,
            reviewer_id="risk_bob",
            approve=True,
            notes="Approved for paper execution",
            reviewed_at_utc=self.snap.as_of_utc,
        )
        self.assertEqual(approved["status"], ProposalStatus.APPROVED.value)

        # Live execution is strictly prohibited
        with self.assertRaises(LiveExecutionProhibitedError):
            self.ledger.apply_approved_proposal(
                proposal_service=self.service,
                proposal_id=prop.proposal_id,
                executed_by="ops_1",
                mode="live",
            )

        # Paper execution succeeds and replay is idempotent (no double cost)
        r1 = self.ledger.apply_approved_proposal(
            proposal_service=self.service,
            proposal_id=prop.proposal_id,
            executed_by="ops_1",
            idempotency_key="exec_idem_01",
            executed_at_utc=self.snap.as_of_utc,
        )
        nav_after_first = self.ledger.total_nav_usd
        cost_after_first = self.ledger.cumulative_transaction_cost_usd
        self.assertFalse(r1.idempotent_replay)

        r2 = self.ledger.apply_approved_proposal(
            proposal_service=self.service,
            proposal_id=prop.proposal_id,
            executed_by="ops_1",
            idempotency_key="exec_idem_01",
            executed_at_utc=self.snap.as_of_utc,
        )
        self.assertTrue(r2.idempotent_replay)
        self.assertEqual(r1.execution_id, r2.execution_id)
        self.assertEqual(self.ledger.total_nav_usd, nav_after_first)
        self.assertEqual(self.ledger.cumulative_transaction_cost_usd, cost_after_first)

        # Audit hash chain verifies cleanly
        chain = self.store.verify_chain()
        self.assertTrue(chain["valid"])
        self.assertGreaterEqual(chain["event_count"], 3)

    def test_expired_proposal_cannot_be_approved(self) -> None:
        prop = self.service.create_proposal(
            self.snap,
            proposer_id="alice",
            now_utc=self.snap.as_of_utc,
        )
        with self.assertRaises(ProposalStateError):
            self.service.review_proposal(
                prop.proposal_id,
                reviewer_id="risk_bob",
                approve=True,
                reviewed_at_utc="2099-01-01T00:00:00Z",
            )
        loaded = self.store.load_proposal(prop.proposal_id)
        self.assertEqual(loaded["status"], ProposalStatus.EXPIRED.value)

    def test_audit_tamper_detection_and_ledger_state_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "audit.sqlite3")
            ledger_path = str(Path(tmpdir) / "ledger.json")
            store = SQLiteAuditStore(db_path)
            svc = ProposalService(self.cfg, store)
            prop = svc.create_proposal(
                self.snap, proposer_id="alice", now_utc=self.snap.as_of_utc
            )
            svc.review_proposal(
                prop.proposal_id,
                reviewer_id="bob",
                approve=True,
                reviewed_at_utc=self.snap.as_of_utc,
            )
            self.ledger.apply_approved_proposal(
                proposal_service=svc,
                proposal_id=prop.proposal_id,
                executed_by="ops",
                executed_at_utc=self.snap.as_of_utc,
            )
            self.ledger.save_json(ledger_path)
            restored = PaperPortfolioLedger.load_json(ledger_path)
            self.assertAlmostEqual(restored.total_nav_usd, self.ledger.total_nav_usd, places=6)
            self.assertEqual(restored.rebalance_count, 1)

            # Tamper directly with sqlite audit payload -> verify_chain MUST fail!
            store._conn.execute(
                "UPDATE dcae_audit_log SET payload_json = '{}' WHERE seq = 1"
            )
            store._conn.commit()
            with self.assertRaises(AuditIntegrityError):
                store.verify_chain()
            store.close()


class TestHTTPAPIIntegration(unittest.TestCase):
    def test_http_api_end_to_end(self) -> None:
        cfg = EngineConfig()
        snap = build_baseline_snapshot(seed=20260409, n_days=60)
        store = SQLiteAuditStore(":memory:")
        ledger = PaperPortfolioLedger(
            initial_cash_usd=snap.positions.cash_usd,
            strategy_notional_usd=dict(snap.positions.strategy_notional_usd),
            as_of_utc=snap.as_of_utc,
        )
        app = DCAEServiceApp(cfg, store, ledger)

        def req(method: str, path: str, data=None):
            raw = None if data is None else json.dumps(data).encode("utf-8")
            status, resp_bytes = app.handle_json(method, path, raw)
            return status, json.loads(resp_bytes.decode("utf-8"))

        try:
            # Invalid JSON rejected cleanly with 400
            bad_code, bad_bytes = app.handle_json("POST", "/v1/snapshots", b"{not-json")
            self.assertEqual(bad_code, 400)
            self.assertEqual(json.loads(bad_bytes.decode("utf-8"))["error_code"], "INVALID_JSON")

            code, health = req("GET", "/health")
            self.assertEqual(code, 200)
            self.assertEqual(health["status"], "ok")
            self.assertFalse(health["live_execution_enabled"])

            code, prop = req(
                "POST",
                "/v1/allocations/propose",
                {
                    "snapshot": snap.to_dict(),
                    "proposer_id": "pm_1",
                    "allocator_name": "constrained_mv",
                    "now_utc": snap.as_of_utc,
                },
            )
            self.assertEqual(code, 201)
            pid = prop["proposal_id"]

            # Self-approval -> 403
            code, err = req(
                "POST",
                f"/v1/proposals/{pid}/review",
                {"reviewer_id": "pm_1", "approve": True, "reviewed_at_utc": snap.as_of_utc},
            )
            self.assertEqual(code, 403)
            self.assertEqual(err["error_code"], "DUAL_CONTROL_VIOLATION")

            # Independent approval -> 200
            code, rev = req(
                "POST",
                f"/v1/proposals/{pid}/review",
                {"reviewer_id": "risk_1", "approve": True, "reviewed_at_utc": snap.as_of_utc},
            )
            self.assertEqual(code, 200)
            self.assertEqual(rev["status"], "APPROVED")

            # Paper execute -> 200
            code, exec_res = req(
                "POST",
                f"/v1/proposals/{pid}/execute-paper",
                {"executed_by": "ops_1", "executed_at_utc": snap.as_of_utc},
            )
            self.assertEqual(code, 200)
            self.assertEqual(exec_res["proposal_id"], pid)

            code, metrics = req("GET", "/v1/metrics")
            self.assertEqual(code, 200)
            self.assertEqual(metrics["paper_executions"], 1)

            code, audit = req("GET", "/v1/audit")
            self.assertEqual(code, 200)
            self.assertTrue(audit["chain_integrity"]["valid"])
        finally:
            store.close()


class TestWalkForwardIntegration(unittest.TestCase):
    def test_walkforward_evaluates_all_allocators_over_identical_periods_and_costs(self) -> None:
        cfg = EngineConfig()
        strategies, prov = generate_walkforward_strategies(seed=20260409, n_days=120)
        wf = WalkForwardEvaluator(cfg)
        report = wf.evaluate(
            strategies=strategies,
            provenance=prov,
            initial_nav_usd=10_000_000.0,
            warmup_periods=30,
            rebalance_every=10,
            lookback_window=60,
        )
        summaries = report["summaries"]
        self.assertEqual(
            set(summaries.keys()),
            {
                "equal_weight",
                "inverse_volatility",
                "risk_budget",
                "constrained_mv",
                "constrained_mv_stressed",
            },
        )
        # All allocators evaluated on exact same periods & rebalance counts
        periods = {s["periods_evaluated"] for s in summaries.values()}
        rebs = {s["rebalance_steps"] for s in summaries.values()}
        self.assertEqual(len(periods), 1)
        self.assertEqual(len(rebs), 1)
        # Constrained allocators must have zero failed steps and zero verifier violations
        self.assertEqual(summaries["constrained_mv"]["failed_step_count"], 0)
        self.assertEqual(summaries["constrained_mv"]["verifier_violation_count"], 0)
        self.assertEqual(summaries["constrained_mv_stressed"]["verifier_violation_count"], 0)


if __name__ == "__main__":
    unittest.main()
