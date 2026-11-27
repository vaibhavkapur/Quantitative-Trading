"""Tamper-evident hash-chained SQLite + JSONL audit trail and state persistence."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dcae.domain.models import FundSnapshot, canonical_json, format_utc, sha256_hex
from dcae.errors import AuditIntegrityError

GENESIS_HASH = "0" * 64


@dataclass(frozen=True)
class AuditEvent:
    seq: int
    event_id: str
    event_type: str
    entity_id: str
    actor_id: str
    occurred_at_utc: str
    prev_hash: str
    entry_hash: str
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "entity_id": self.entity_id,
            "actor_id": self.actor_id,
            "occurred_at_utc": self.occurred_at_utc,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
            "payload": self.payload,
        }


class SQLiteAuditStore:
    """Append-only hash-chained SQLite store for snapshots, proposals, executions, and audit events."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        if db_path != ":memory:":
            parent = os.path.dirname(os.path.abspath(db_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS dcae_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    as_of_utc TEXT NOT NULL,
                    published_at_utc TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS dcae_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    decision_id TEXT NOT NULL,
                    snapshot_id TEXT NOT NULL,
                    config_version TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    allocator_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    proposer_id TEXT NOT NULL,
                    reviewer_id TEXT,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    expires_at_utc TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS dcae_audit_log (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    event_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    occurred_at_utc TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    entry_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS dcae_paper_executions (
                    execution_id TEXT PRIMARY KEY,
                    proposal_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    executed_by TEXT NOT NULL,
                    executed_at_utc TEXT NOT NULL,
                    total_nav_usd REAL NOT NULL,
                    cash_usd REAL NOT NULL,
                    total_cost_usd REAL NOT NULL,
                    receipt_json TEXT NOT NULL
                );
                """
            )

    def save_snapshot(self, snapshot: FundSnapshot) -> None:
        payload_json = canonical_json(snapshot.to_dict())
        prov_json = canonical_json(snapshot.provenance.to_dict())
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO dcae_snapshots
                    (snapshot_id, schema_version, as_of_utc, published_at_utc,
                     content_hash, provenance_json, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.snapshot_id,
                    snapshot.schema_version,
                    snapshot.as_of_utc,
                    snapshot.published_at_utc,
                    snapshot.content_hash(),
                    prov_json,
                    payload_json,
                ),
            )

    def load_snapshot(self, snapshot_id: str) -> Optional[FundSnapshot]:
        row = self._conn.execute(
            "SELECT payload_json FROM dcae_snapshots WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchone()
        if row is None:
            return None
        return FundSnapshot.from_dict(json.loads(row["payload_json"]))

    def save_proposal(self, proposal_dict: Dict[str, Any]) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO dcae_proposals
                    (proposal_id, idempotency_key, decision_id, snapshot_id,
                     config_version, config_hash, allocator_name, status,
                     proposer_id, reviewer_id, created_at_utc, updated_at_utc,
                     expires_at_utc, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    proposal_dict["proposal_id"],
                    proposal_dict["idempotency_key"],
                    proposal_dict["decision"]["decision_id"],
                    proposal_dict["decision"]["snapshot_id"],
                    proposal_dict["decision"]["config_version"],
                    proposal_dict["decision"]["config_hash"],
                    proposal_dict["decision"]["allocator_name"],
                    proposal_dict["status"],
                    proposal_dict["proposer_id"],
                    proposal_dict.get("reviewer_id"),
                    proposal_dict["created_at_utc"],
                    proposal_dict["updated_at_utc"],
                    proposal_dict["expires_at_utc"],
                    canonical_json(proposal_dict),
                ),
            )

    def load_proposal(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT payload_json FROM dcae_proposals WHERE proposal_id = ?",
            (proposal_id,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["payload_json"])

    def load_proposal_by_idempotency_key(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT payload_json FROM dcae_proposals WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["payload_json"])

    def list_proposals(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        if status is None:
            rows = self._conn.execute(
                "SELECT payload_json FROM dcae_proposals ORDER BY created_at_utc DESC"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT payload_json FROM dcae_proposals WHERE status = ? ORDER BY created_at_utc DESC",
                (status,),
            ).fetchall()
        return [json.loads(r["payload_json"]) for r in rows]

    def save_paper_execution(self, receipt: Dict[str, Any]) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO dcae_paper_executions
                    (execution_id, proposal_id, idempotency_key, executed_by,
                     executed_at_utc, total_nav_usd, cash_usd, total_cost_usd, receipt_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt["execution_id"],
                    receipt["proposal_id"],
                    receipt["idempotency_key"],
                    receipt["executed_by"],
                    receipt["executed_at_utc"],
                    float(receipt["total_nav_usd"]),
                    float(receipt["cash_usd"]),
                    float(receipt["total_cost_usd"]),
                    canonical_json(receipt),
                ),
            )

    def load_paper_execution_by_key(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT receipt_json FROM dcae_paper_executions WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["receipt_json"])

    def append_event(
        self,
        *,
        event_type: str,
        entity_id: str,
        actor_id: str,
        payload: Dict[str, Any],
        occurred_at_utc: Optional[str] = None,
    ) -> AuditEvent:
        ts = occurred_at_utc or format_utc(datetime.now(timezone.utc))
        event_id = f"evt_{uuid.uuid4().hex[:16]}"
        with self._conn:
            last = self._conn.execute(
                "SELECT entry_hash FROM dcae_audit_log ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            prev_hash = last["entry_hash"] if last is not None else GENESIS_HASH
            entry_material = {
                "event_id": event_id,
                "event_type": event_type,
                "entity_id": entity_id,
                "actor_id": actor_id,
                "occurred_at_utc": ts,
                "prev_hash": prev_hash,
                "payload": payload,
            }
            entry_hash = sha256_hex(entry_material)
            cur = self._conn.execute(
                """
                INSERT INTO dcae_audit_log
                    (event_id, event_type, entity_id, actor_id,
                     occurred_at_utc, prev_hash, entry_hash, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    event_type,
                    entity_id,
                    actor_id,
                    ts,
                    prev_hash,
                    entry_hash,
                    canonical_json(payload),
                ),
            )
            seq = int(cur.lastrowid)
        return AuditEvent(
            seq=seq,
            event_id=event_id,
            event_type=event_type,
            entity_id=entity_id,
            actor_id=actor_id,
            occurred_at_utc=ts,
            prev_hash=prev_hash,
            entry_hash=entry_hash,
            payload=payload,
        )

    def list_events(self, entity_id: Optional[str] = None) -> List[AuditEvent]:
        if entity_id is None:
            rows = self._conn.execute(
                "SELECT * FROM dcae_audit_log ORDER BY seq ASC"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM dcae_audit_log WHERE entity_id = ? ORDER BY seq ASC",
                (entity_id,),
            ).fetchall()
        return [
            AuditEvent(
                seq=int(r["seq"]),
                event_id=str(r["event_id"]),
                event_type=str(r["event_type"]),
                entity_id=str(r["entity_id"]),
                actor_id=str(r["actor_id"]),
                occurred_at_utc=str(r["occurred_at_utc"]),
                prev_hash=str(r["prev_hash"]),
                entry_hash=str(r["entry_hash"]),
                payload=json.loads(r["payload_json"]),
            )
            for r in rows
        ]

    def verify_chain(self) -> Dict[str, Any]:
        """Recompute every hash in the audit log and raise AuditIntegrityError on tamper."""
        events = self.list_events()
        expected_prev = GENESIS_HASH
        for ev in events:
            if ev.prev_hash != expected_prev:
                raise AuditIntegrityError(
                    f"Broken audit hash chain at seq={ev.seq}",
                    context={
                        "seq": ev.seq,
                        "expected_prev_hash": expected_prev,
                        "actual_prev_hash": ev.prev_hash,
                    },
                )
            material = {
                "event_id": ev.event_id,
                "event_type": ev.event_type,
                "entity_id": ev.entity_id,
                "actor_id": ev.actor_id,
                "occurred_at_utc": ev.occurred_at_utc,
                "prev_hash": ev.prev_hash,
                "payload": ev.payload,
            }
            recomputed = sha256_hex(material)
            if recomputed != ev.entry_hash:
                raise AuditIntegrityError(
                    f"Tampered audit entry payload at seq={ev.seq}",
                    context={
                        "seq": ev.seq,
                        "expected_entry_hash": recomputed,
                        "actual_entry_hash": ev.entry_hash,
                    },
                )
            expected_prev = ev.entry_hash
        return {
            "valid": True,
            "event_count": len(events),
            "head_hash": expected_prev,
        }

    def export_jsonl(self, path: str) -> int:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        events = self.list_events()
        with open(path, "w", encoding="utf-8") as fh:
            for ev in events:
                fh.write(canonical_json(ev.to_dict()) + "\n")
        return len(events)
