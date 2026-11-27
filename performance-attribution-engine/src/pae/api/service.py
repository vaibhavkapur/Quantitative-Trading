"""Application service layer combining attribution pipeline, RBAC, versioning, audit, and telemetry."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from pae.config import AttributionConfig
from pae.domain.models import AttributionInputBatch
from pae.engine.pipeline import AttributionPipeline
from pae.errors import AuthorizationError
from pae.reporting.exporter import export_csv, export_html, export_json, export_markdown
from pae.security.rbac import Actor, Permission, check_permission, enforce_dual_control
from pae.storage.sqlite_store import SQLiteAttributionStore


class AttributionService:
    """Operational facade enforcing authorization, versioned revisions, audit trail, and health metrics."""

    def __init__(
        self,
        config: Optional[AttributionConfig] = None,
        store: Optional[SQLiteAttributionStore] = None,
    ) -> None:
        self.config = config or AttributionConfig()
        self.pipeline = AttributionPipeline(self.config)
        self.store = store or SQLiteAttributionStore(self.config.sqlite_path)
        self.metrics: Dict[str, Any] = {
            "runs_total": 0,
            "runs_failed": 0,
            "revisions_total": 0,
            "reviews_total": 0,
            "approvals_total": 0,
            "last_run_latency_ms": 0.0,
        }

    def health(self) -> Dict[str, Any]:
        chain_ok, audit_count, err = self.store.verify_audit_chain()
        return {
            "status": "OK" if chain_ok else "DEGRADED_AUDIT_INTEGRITY",
            "methodology_version": self.config.methodology_version,
            "config_version": self.config.config_version,
            "config_hash": self.config.config_hash(),
            "environment_mode": self.config.environment_mode,
            "order_execution_enabled": self.config.order_execution_enabled,
            "audit_chain_valid": chain_ok,
            "audit_entry_count": audit_count,
            "audit_error": err,
            "metrics": dict(self.metrics),
        }

    def run_attribution(
        self,
        batch: AttributionInputBatch,
        *,
        actor: Actor,
        report_id: Optional[str] = None,
        correction_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run attribution and persist an immutable report version (or revised version if report_id exists)."""
        required_perm = Permission.REVISE_REPORT if correction_reason else Permission.RUN_ATTRIBUTION
        check_permission(actor, required_perm)

        t0 = time.perf_counter()
        try:
            self.store.save_input_batch(
                batch_id=batch.batch_id,
                portfolio_id=batch.portfolio_id,
                as_of_utc=batch.as_of_utc,
                reporting_currency=batch.reporting_currency,
                content_hash=batch.content_hash(),
                payload=batch.to_dict(),
            )

            navs = sorted(batch.nav_snapshots, key=lambda n: n.trade_date)
            logical_id = report_id or (
                f"RPT-{batch.portfolio_id}-{navs[0].trade_date.replace('-', '')}-"
                f"{navs[-1].trade_date.replace('-', '')}"
            )
            next_ver, prior_report = self.store.next_version_info(logical_id)
            supersedes_id = prior_report["metadata"]["version_id"] if prior_report else None

            report = self.pipeline.run(
                batch,
                report_id=logical_id,
                version_number=next_ver,
                supersedes_version_id=supersedes_id,
                correction_reason=correction_reason,
                prior_report=prior_report,
            )
            saved = self.store.save_report(report, author_id=actor.actor_id)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.metrics["runs_total"] += 1
            if next_ver > 1:
                self.metrics["revisions_total"] += 1
            self.metrics["last_run_latency_ms"] = round(latency_ms, 3)

            self.store.append_audit(
                actor_id=actor.actor_id,
                actor_role=actor.role.value,
                action="REVISE_REPORT" if next_ver > 1 else "RUN_ATTRIBUTION",
                resource_id=report.version_id,
                outcome="SUCCESS",
                details={
                    "report_id": report.report_id,
                    "version_number": next_ver,
                    "supersedes_version_id": supersedes_id,
                    "correction_reason": correction_reason,
                    "reconciliation_status": report.reconciliation.status,
                    "latency_ms": round(latency_ms, 3),
                },
            )
            return saved
        except Exception as exc:
            self.metrics["runs_failed"] += 1
            self.store.append_audit(
                actor_id=actor.actor_id,
                actor_role=actor.role.value,
                action="RUN_ATTRIBUTION",
                resource_id=batch.batch_id,
                outcome="FAILURE",
                details={"error": str(exc)},
            )
            raise

    def list_reports(
        self,
        *,
        actor: Actor,
        portfolio_id: Optional[str] = None,
        report_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        check_permission(actor, Permission.READ_REPORT)
        approved_only = actor.role.value == "VIEWER"
        return self.store.list_reports(
            portfolio_id=portfolio_id,
            report_id=report_id,
            approved_only=approved_only,
        )

    def get_report(self, version_id: str, *, actor: Actor) -> Dict[str, Any]:
        check_permission(actor, Permission.READ_REPORT)
        rec = self.store.get_report_record(version_id)
        if actor.role.value == "VIEWER" and rec["workflow"]["status"] != "APPROVED":
            raise AuthorizationError(
                f"VIEWER role may only access APPROVED reports; {version_id} is {rec['workflow']['status']}."
            )
        return rec

    def export_report(self, version_id: str, fmt: str, *, actor: Actor) -> str:
        rec = self.get_report(version_id, actor=actor)
        payload = rec["report"]
        f = fmt.lower()
        if f == "json":
            return export_json(payload)
        if f in {"md", "markdown"}:
            return export_markdown(payload)
        if f == "csv":
            return export_csv(payload)
        if f == "html":
            return export_html(payload)
        raise ValueError(f"Unsupported export format: {fmt!r}")

    def review_report(self, version_id: str, *, actor: Actor, comment: str = "") -> Dict[str, Any]:
        check_permission(actor, Permission.REVIEW_REPORT)
        rec = self.store.get_report_record(version_id)
        wf = rec["workflow"]
        if self.config.require_dual_control_approval:
            enforce_dual_control(
                action="REVIEW",
                actor=actor,
                author_id=wf["author_id"],
                current_status=wf["status"],
            )
        updated = self.store.update_workflow_status(
            version_id, new_status="REVIEWED", reviewer_id=actor.actor_id
        )
        self.metrics["reviews_total"] += 1
        self.store.append_audit(
            actor_id=actor.actor_id,
            actor_role=actor.role.value,
            action="REVIEW_REPORT",
            resource_id=version_id,
            outcome="SUCCESS",
            details={"comment": comment},
        )
        return updated

    def approve_report(self, version_id: str, *, actor: Actor, comment: str = "") -> Dict[str, Any]:
        check_permission(actor, Permission.APPROVE_REPORT)
        rec = self.store.get_report_record(version_id)
        wf = rec["workflow"]
        if self.config.require_dual_control_approval:
            enforce_dual_control(
                action="APPROVE",
                actor=actor,
                author_id=wf["author_id"],
                reviewer_id=wf.get("reviewer_id"),
                current_status=wf["status"],
            )
        updated = self.store.update_workflow_status(
            version_id, new_status="APPROVED", approver_id=actor.actor_id
        )
        self.metrics["approvals_total"] += 1
        self.store.append_audit(
            actor_id=actor.actor_id,
            actor_role=actor.role.value,
            action="APPROVE_REPORT",
            resource_id=version_id,
            outcome="SUCCESS",
            details={"comment": comment},
        )
        return updated

    def get_audit_log(self, *, actor: Actor, limit: int = 200) -> Dict[str, Any]:
        check_permission(actor, Permission.READ_AUDIT)
        chain_ok, count, err = self.store.verify_audit_chain()
        return {
            "chain_valid": chain_ok,
            "entry_count": count,
            "error": err,
            "entries": self.store.list_audit_log(limit=limit),
        }
