"""Integration tests for full pipeline, versioned storage, RBAC, audit chain, and HTTP REST API."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from pae.api.http_server import dispatch_request
from pae.api.service import AttributionService
from pae.config import AttributionConfig
from pae.domain.models import AttributionInputBatch
from pae.errors import AuthorizationError, WorkflowViolationError
from pae.security.rbac import Actor, Role
from pae.storage.sqlite_store import SQLiteAttributionStore

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "data" / "fixtures"


class TestPipelineStorageRBACAndAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.v1_dict = json.loads((FIXTURES_DIR / "multi_strategy_multi_ccy_baseline.json").read_text(encoding="utf-8"))
        self.v2_dict = json.loads((FIXTURES_DIR / "multi_strategy_fee_correction_v2.json").read_text(encoding="utf-8"))
        self.cfg = AttributionConfig()
        self.store = SQLiteAttributionStore(":memory:")
        self.svc = AttributionService(config=self.cfg, store=self.store)

    def tearDown(self) -> None:
        self.store.close()

    def test_versioned_lifecycle_rbac_dual_control_and_tamper_detection(self) -> None:
        viewer = Actor("viewer-1", Role.VIEWER)
        analyst = Actor("analyst-1", Role.ANALYST)
        reviewer = Actor("reviewer-1", Role.REVIEWER)
        approver = Actor("approver-1", Role.APPROVER)

        v1_batch = AttributionInputBatch.from_dict(self.v1_dict)
        v2_batch = AttributionInputBatch.from_dict(self.v2_dict)

        # Viewer cannot run attribution
        with self.assertRaises(AuthorizationError):
            self.svc.run_attribution(v1_batch, actor=viewer)

        # Analyst creates v1 DRAFT
        v1 = self.svc.run_attribution(v1_batch, actor=analyst)
        v1_id = v1["version_id"]
        self.assertEqual(v1["version_number"], 1)
        self.assertEqual(v1["reconciliation_status"], "RECONCILED")
        self.assertEqual(v1["report"]["brinson"]["status"], "AVAILABLE")
        self.assertEqual(v1["report"]["factors"]["status"], "AVAILABLE")

        # Viewer cannot read unapproved DRAFT
        with self.assertRaises(AuthorizationError):
            self.svc.get_report(v1_id, actor=viewer)

        # Approver cannot bypass review step
        with self.assertRaises(WorkflowViolationError):
            self.svc.approve_report(v1_id, actor=approver)

        # Author cannot review own report (even with ADMIN role)
        with self.assertRaises(WorkflowViolationError):
            self.svc.review_report(v1_id, actor=Actor("analyst-1", Role.ADMIN))

        # Independent reviewer reviews, then independent approver approves
        self.svc.review_report(v1_id, actor=reviewer, comment="Reconciled OK")
        approved_v1 = self.svc.approve_report(v1_id, actor=approver, comment="Approved")
        self.assertEqual(approved_v1["workflow"]["status"], "APPROVED")

        # Viewer can now read approved v1
        self.assertEqual(self.svc.get_report(v1_id, actor=viewer)["version_id"], v1_id)

        # Fee correction creates revised v2 superseding v1; v1 remains immutable
        v2 = self.svc.run_attribution(
            v2_batch,
            actor=analyst,
            report_id=v1["report_id"],
            correction_reason="Prime broker rebate applied",
        )
        self.assertEqual(v2["version_number"], 2)
        self.assertEqual(v2["supersedes_version_id"], v1_id)
        self.assertIsNotNone(v2["report"]["revision_diff"])
        # Day 5 has exact -$450 rebate; subsequent 5 days accrue tiny +$0.1339 mgmt fee on higher NAV
        d5_v1 = [d for d in v1["report"]["returns"]["daily_returns"] if d["trade_date"] == "2026-01-09"][0]
        d5_v2 = [d for d in v2["report"]["returns"]["daily_returns"] if d["trade_date"] == "2026-01-09"][0]
        self.assertAlmostEqual(d5_v2["total_fees_base"] - d5_v1["total_fees_base"], -450.0, places=6)
        self.assertLess(v2["report"]["revision_diff"]["delta_total_fees_base"], -449.0)

        # Verify v1 report hash and payload are unchanged
        v1_after = self.store.get_report_record(v1_id)
        self.assertEqual(v1_after["report_hash"], v1["report_hash"])
        self.assertEqual(v1_after["workflow"]["status"], "SUPERSEDED")

        # Verify audit chain valid, then simulate DB tamper and verify detection
        ok, count, err = self.store.verify_audit_chain()
        self.assertTrue(ok, msg=f"Audit chain failed: {err}")
        self.assertGreaterEqual(count, 4)

        self.store._conn.execute("UPDATE pae_audit_log SET outcome = 'TAMPERED' WHERE seq_id = 1")
        self.store._conn.commit()
        ok_after, _, err_after = self.store.verify_audit_chain()
        self.assertFalse(ok_after)
        self.assertIn("Tampered", err_after or "")

    def test_http_rest_api_and_dashboard_end_to_end(self) -> None:
        # 1. Health endpoint
        st, ctype, raw = dispatch_request(self.svc, "GET", "/api/v1/health")
        self.assertEqual(st, 200)
        self.assertEqual(ctype, "application/json")
        health = json.loads(raw)
        self.assertFalse(health["order_execution_enabled"])
        self.assertTrue(health["audit_chain_valid"])

        # 2. Viewer rejected on POST run (403)
        st, _, err_raw = dispatch_request(
            self.svc,
            "POST",
            "/api/v1/attribution/run",
            headers={"X-Actor-Id": "api-viewer", "X-Actor-Role": "VIEWER"},
            raw_body=json.dumps({"batch": self.v1_dict}),
        )
        self.assertEqual(st, 403)
        self.assertEqual(json.loads(err_raw)["error_code"], "AUTHORIZATION_ERROR")

        # 3. Analyst creates v1 report (201)
        st, _, raw = dispatch_request(
            self.svc,
            "POST",
            "/api/v1/attribution/run",
            headers={"X-Actor-Id": "api-analyst", "X-Actor-Role": "ANALYST"},
            raw_body=json.dumps({"batch": self.v1_dict}),
        )
        self.assertEqual(st, 201)
        vid = json.loads(raw)["version_id"]

        # 4. Review & approve via POST
        st, _, _ = dispatch_request(
            self.svc,
            "POST",
            f"/api/v1/reports/{vid}/review",
            headers={"X-Actor-Id": "api-reviewer", "X-Actor-Role": "REVIEWER"},
            raw_body=json.dumps({"comment": "LGTM"}),
        )
        self.assertEqual(st, 200)
        st, _, _ = dispatch_request(
            self.svc,
            "POST",
            f"/api/v1/reports/{vid}/approve",
            headers={"X-Actor-Id": "api-approver", "X-Actor-Role": "APPROVER"},
            raw_body=json.dumps({"comment": "Approved"}),
        )
        self.assertEqual(st, 200)

        # 5. Export Markdown, CSV, HTML, JSON, Dashboard index, and Audit log
        for fmt in ("md", "csv", "html", "json"):
            st, _, text = dispatch_request(
                self.svc,
                "GET",
                f"/api/v1/reports/{vid}/export?format={fmt}",
                headers={"X-Actor-Id": "api-viewer", "X-Actor-Role": "VIEWER"},
            )
            self.assertEqual(st, 200)
            self.assertGreater(len(text), 50)

        st, ctype, dash = dispatch_request(self.svc, "GET", "/dashboard")
        self.assertEqual(st, 200)
        self.assertEqual(ctype, "text/html")
        self.assertIn(vid, dash)

        st, _, audit_raw = dispatch_request(
            self.svc,
            "GET",
            "/api/v1/audit",
            headers={"X-Actor-Id": "api-approver", "X-Actor-Role": "APPROVER"},
        )
        self.assertEqual(st, 200)
        self.assertTrue(json.loads(audit_raw)["chain_valid"])


if __name__ == "__main__":
    unittest.main()
