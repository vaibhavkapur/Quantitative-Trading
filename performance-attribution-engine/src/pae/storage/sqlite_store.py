"""SQLite persistence for immutable report versions, provenance, and SHA-256 hash-chained audit logs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pae.engine.pipeline import AttributionReport
from pae.errors import IntegrityError, NotFoundError


GENESIS_HASH = "0" * 64


class SQLiteAttributionStore:
    """Zero-dependency local store implementing immutable versioned reports and hash-chained audit logs."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS pae_input_batches (
                batch_id           TEXT PRIMARY KEY,
                portfolio_id       TEXT NOT NULL,
                as_of_utc          TEXT NOT NULL,
                reporting_currency TEXT NOT NULL,
                content_hash       TEXT NOT NULL,
                payload_json       TEXT NOT NULL,
                created_at_utc     TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS pae_attribution_reports (
                version_id            TEXT PRIMARY KEY,
                report_id             TEXT NOT NULL,
                portfolio_id          TEXT NOT NULL,
                version_number        INTEGER NOT NULL,
                supersedes_version_id TEXT,
                correction_reason     TEXT,
                methodology_version   TEXT NOT NULL,
                config_hash           TEXT NOT NULL,
                input_hash            TEXT NOT NULL,
                report_hash           TEXT NOT NULL,
                period_start          TEXT NOT NULL,
                period_end            TEXT NOT NULL,
                reporting_currency    TEXT NOT NULL,
                environment_mode      TEXT NOT NULL,
                reconciliation_status TEXT NOT NULL,
                status                TEXT NOT NULL,
                author_id             TEXT NOT NULL,
                reviewer_id           TEXT,
                approver_id           TEXT,
                created_at_utc        TEXT NOT NULL,
                reviewed_at_utc       TEXT,
                approved_at_utc       TEXT,
                report_json           TEXT NOT NULL,
                UNIQUE (report_id, version_number)
            );

            CREATE TABLE IF NOT EXISTS pae_audit_log (
                seq_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id      TEXT UNIQUE NOT NULL,
                timestamp_utc TEXT NOT NULL,
                actor_id      TEXT NOT NULL,
                actor_role    TEXT NOT NULL,
                action        TEXT NOT NULL,
                resource_id   TEXT NOT NULL,
                outcome       TEXT NOT NULL,
                details_json  TEXT NOT NULL,
                prev_hash     TEXT NOT NULL,
                entry_hash    TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    def save_input_batch(
        self,
        batch_id: str,
        portfolio_id: str,
        as_of_utc: str,
        reporting_currency: str,
        content_hash: str,
        payload: Dict[str, Any],
    ) -> None:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._conn.execute(
            """
            INSERT OR REPLACE INTO pae_input_batches
                (batch_id, portfolio_id, as_of_utc, reporting_currency, content_hash, payload_json, created_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (batch_id, portfolio_id, as_of_utc, reporting_currency, content_hash, json.dumps(payload, sort_keys=True), now),
        )
        self._conn.commit()

    def next_version_info(self, report_id: str) -> Tuple[int, Optional[Dict[str, Any]]]:
        """Return (next_version_number, latest_prior_report_dict_or_None)."""
        row = self._conn.execute(
            """
            SELECT version_number, report_json
            FROM pae_attribution_reports
            WHERE report_id = ?
            ORDER BY version_number DESC
            LIMIT 1
            """,
            (report_id,),
        ).fetchone()
        if row is None:
            return 1, None
        return int(row["version_number"]) + 1, json.loads(row["report_json"])

    def save_report(self, report: AttributionReport, *, author_id: str) -> Dict[str, Any]:
        """Persist an immutable report version; mark any superseded version as SUPERSEDED without altering its payload."""
        report_dict = report.to_dict()
        now = report.generated_at_utc
        cur = self._conn.cursor()

        # Mark superseded version status if applicable (payload remains strictly immutable)
        if report.supersedes_version_id:
            cur.execute(
                """
                UPDATE pae_attribution_reports
                SET status = 'SUPERSEDED'
                WHERE version_id = ? AND status IN ('DRAFT', 'REVIEWED', 'APPROVED')
                """,
                (report.supersedes_version_id,),
            )

        cur.execute(
            """
            INSERT INTO pae_attribution_reports (
                version_id, report_id, portfolio_id, version_number,
                supersedes_version_id, correction_reason, methodology_version,
                config_hash, input_hash, report_hash, period_start, period_end,
                reporting_currency, environment_mode, reconciliation_status,
                status, author_id, created_at_utc, report_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'DRAFT', ?, ?, ?)
            """,
            (
                report.version_id,
                report.report_id,
                report.portfolio_id,
                report.version_number,
                report.supersedes_version_id,
                report.correction_reason,
                report.methodology_version,
                report.config_hash,
                report.input_hash,
                report.report_hash,
                report.period_start,
                report.period_end,
                report.reporting_currency,
                report.environment_mode,
                report.reconciliation.status,
                author_id,
                now,
                json.dumps(report_dict, sort_keys=True),
            ),
        )
        self._conn.commit()
        return self.get_report_record(report.version_id)

    def get_report_record(self, version_id: str) -> Dict[str, Any]:
        row = self._conn.execute(
            "SELECT * FROM pae_attribution_reports WHERE version_id = ?",
            (version_id,),
        ).fetchone()
        if row is None:
            raise NotFoundError(f"Report version {version_id!r} not found.")
        return self._row_to_record(row)

    def list_reports(
        self,
        *,
        portfolio_id: Optional[str] = None,
        report_id: Optional[str] = None,
        approved_only: bool = False,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if portfolio_id:
            clauses.append("portfolio_id = ?")
            params.append(portfolio_id)
        if report_id:
            clauses.append("report_id = ?")
            params.append(report_id)
        if approved_only:
            clauses.append("status = 'APPROVED'")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"""
            SELECT version_id, report_id, portfolio_id, version_number,
                   supersedes_version_id, correction_reason, methodology_version,
                   period_start, period_end, reporting_currency, environment_mode,
                   reconciliation_status, status, author_id, reviewer_id, approver_id,
                   created_at_utc, reviewed_at_utc, approved_at_utc, report_hash
            FROM pae_attribution_reports
            {where}
            ORDER BY report_id ASC, version_number DESC
            """,
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def update_workflow_status(
        self,
        version_id: str,
        *,
        new_status: str,
        reviewer_id: Optional[str] = None,
        approver_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if new_status == "REVIEWED":
            self._conn.execute(
                """
                UPDATE pae_attribution_reports
                SET status = 'REVIEWED', reviewer_id = ?, reviewed_at_utc = ?
                WHERE version_id = ?
                """,
                (reviewer_id, now, version_id),
            )
        elif new_status == "APPROVED":
            self._conn.execute(
                """
                UPDATE pae_attribution_reports
                SET status = 'APPROVED', approver_id = ?, approved_at_utc = ?
                WHERE version_id = ?
                """,
                (approver_id, now, version_id),
            )
        self._conn.commit()
        return self.get_report_record(version_id)

    def append_audit(
        self,
        *,
        actor_id: str,
        actor_role: str,
        action: str,
        resource_id: str,
        outcome: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        row = self._conn.execute(
            "SELECT entry_hash FROM pae_audit_log ORDER BY seq_id DESC LIMIT 1"
        ).fetchone()
        prev_hash = row["entry_hash"] if row is not None else GENESIS_HASH

        event_id = f"AUD-{uuid.uuid4().hex[:12]}"
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        details_str = json.dumps(details or {}, sort_keys=True, separators=(",", ":"))
        payload = f"{prev_hash}|{event_id}|{ts}|{actor_id}|{actor_role}|{action}|{resource_id}|{outcome}|{details_str}"
        entry_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        self._conn.execute(
            """
            INSERT INTO pae_audit_log
                (event_id, timestamp_utc, actor_id, actor_role, action, resource_id, outcome, details_json, prev_hash, entry_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (event_id, ts, actor_id, actor_role, action, resource_id, outcome, details_str, prev_hash, entry_hash),
        )
        self._conn.commit()
        return {
            "event_id": event_id,
            "timestamp_utc": ts,
            "actor_id": actor_id,
            "actor_role": actor_role,
            "action": action,
            "resource_id": resource_id,
            "outcome": outcome,
            "details": details or {},
            "prev_hash": prev_hash,
            "entry_hash": entry_hash,
        }

    def list_audit_log(self, limit: int = 200) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM pae_audit_log ORDER BY seq_id ASC LIMIT ?",
            (limit,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["details"] = json.loads(d.pop("details_json"))
            out.append(d)
        return out

    def verify_audit_chain(self) -> Tuple[bool, int, Optional[str]]:
        """Verify SHA-256 hash chain integrity across all audit entries."""
        rows = self._conn.execute("SELECT * FROM pae_audit_log ORDER BY seq_id ASC").fetchall()
        expected_prev = GENESIS_HASH
        for r in rows:
            if r["prev_hash"] != expected_prev:
                return False, len(rows), f"Chain break at seq_id={r['seq_id']}: prev_hash mismatch"
            payload = (
                f"{r['prev_hash']}|{r['event_id']}|{r['timestamp_utc']}|{r['actor_id']}|"
                f"{r['actor_role']}|{r['action']}|{r['resource_id']}|{r['outcome']}|{r['details_json']}"
            )
            recomputed = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            if recomputed != r["entry_hash"]:
                return False, len(rows), f"Tampered audit record at seq_id={r['seq_id']}"
            expected_prev = r["entry_hash"]
        return True, len(rows), None

    def _row_to_record(self, row: sqlite3.Row) -> Dict[str, Any]:
        report_payload = json.loads(row["report_json"])
        return {
            "version_id": row["version_id"],
            "report_id": row["report_id"],
            "portfolio_id": row["portfolio_id"],
            "version_number": row["version_number"],
            "supersedes_version_id": row["supersedes_version_id"],
            "correction_reason": row["correction_reason"],
            "methodology_version": row["methodology_version"],
            "config_hash": row["config_hash"],
            "input_hash": row["input_hash"],
            "report_hash": row["report_hash"],
            "period_start": row["period_start"],
            "period_end": row["period_end"],
            "reporting_currency": row["reporting_currency"],
            "environment_mode": row["environment_mode"],
            "reconciliation_status": row["reconciliation_status"],
            "workflow": {
                "status": row["status"],
                "author_id": row["author_id"],
                "reviewer_id": row["reviewer_id"],
                "approver_id": row["approver_id"],
                "created_at_utc": row["created_at_utc"],
                "reviewed_at_utc": row["reviewed_at_utc"],
                "approved_at_utc": row["approved_at_utc"],
            },
            "report": report_payload,
        }
