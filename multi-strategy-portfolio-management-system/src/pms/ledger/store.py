"""Immutable append-only event store (SQLite and in-memory) with hash chaining."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Protocol

from pms.domain.errors import DuplicateEventError, ImmutabilityViolationError
from pms.domain.models import LedgerEvent, StoredEvent, parse_ledger_event

GENESIS_HASH = "0" * 64


def canonical_event_json(event: LedgerEvent) -> str:
    return json.dumps(event.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def compute_event_hash(prev_hash: str, seq_no: int, payload_json: str) -> str:
    digest = hashlib.sha256()
    digest.update(f"{prev_hash}|{seq_no}|{payload_json}".encode())
    return digest.hexdigest()


class EventStore(Protocol):
    def append(self, event: LedgerEvent) -> StoredEvent: ...

    def list_events(self, *, as_of: datetime | None = None) -> list[StoredEvent]: ...

    def has_idempotency_key(self, idempotency_key: str) -> bool: ...

    def get_by_idempotency_key(self, idempotency_key: str) -> StoredEvent | None: ...

    def verify_hash_chain(self) -> tuple[bool, str]: ...


class InMemoryEventStore:
    """Fast deterministic in-memory append-only store for unit tests."""

    def __init__(self) -> None:
        self._events: list[StoredEvent] = []
        self._by_idempotency: dict[str, StoredEvent] = {}
        self._by_event_id: dict[str, StoredEvent] = {}

    def append(self, event: LedgerEvent) -> StoredEvent:
        if event.idempotency_key in self._by_idempotency:
            raise DuplicateEventError(
                f"Duplicate idempotency_key: {event.idempotency_key}",
                details={
                    "idempotency_key": event.idempotency_key,
                    "existing_event_id": self._by_idempotency[event.idempotency_key].event_id,
                    "incoming_event_id": event.event_id,
                },
            )
        if event.event_id in self._by_event_id:
            raise DuplicateEventError(
                f"Duplicate event_id: {event.event_id}",
                details={"event_id": event.event_id},
            )

        seq_no = len(self._events) + 1
        prev_hash = self._events[-1].event_hash if self._events else GENESIS_HASH
        payload_json = canonical_event_json(event)
        event_hash = compute_event_hash(prev_hash, seq_no, payload_json)
        stored = StoredEvent(
            seq_no=seq_no,
            event_id=event.event_id,
            idempotency_key=event.idempotency_key,
            event_type=event.event_type,
            event_time=event.event_time,
            recorded_time=event.recorded_time,
            prev_hash=prev_hash,
            event_hash=event_hash,
            event=event,
        )
        self._events.append(stored)
        self._by_idempotency[event.idempotency_key] = stored
        self._by_event_id[event.event_id] = stored
        return stored

    def list_events(self, *, as_of: datetime | None = None) -> list[StoredEvent]:
        if as_of is None:
            return list(self._events)
        return [item for item in self._events if item.event_time <= as_of]

    def has_idempotency_key(self, idempotency_key: str) -> bool:
        return idempotency_key in self._by_idempotency

    def get_by_idempotency_key(self, idempotency_key: str) -> StoredEvent | None:
        return self._by_idempotency.get(idempotency_key)

    def verify_hash_chain(self) -> tuple[bool, str]:
        prev_hash = GENESIS_HASH
        for stored in self._events:
            expected = compute_event_hash(
                prev_hash, stored.seq_no, canonical_event_json(stored.event)
            )
            if stored.prev_hash != prev_hash or stored.event_hash != expected:
                return False, f"Hash mismatch at seq_no={stored.seq_no}"
            prev_hash = stored.event_hash
        return True, prev_hash


class SQLiteEventStore:
    """SQLite-backed immutable append-only event store with DB-level immutability triggers."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS ledger_events (
                    seq_no INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    event_type TEXT NOT NULL,
                    event_time TEXT NOT NULL,
                    recorded_time TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TRIGGER IF NOT EXISTS prevent_ledger_events_update
                BEFORE UPDATE ON ledger_events
                BEGIN
                    SELECT RAISE(ABORT, 'IMMUTABILITY_VIOLATION: ledger_events cannot be updated');
                END;

                CREATE TRIGGER IF NOT EXISTS prevent_ledger_events_delete
                BEFORE DELETE ON ledger_events
                BEGIN
                    SELECT RAISE(ABORT, 'IMMUTABILITY_VIOLATION: ledger_events cannot be deleted');
                END;

                CREATE TABLE IF NOT EXISTS materialized_state_cache (
                    cache_key TEXT PRIMARY KEY,
                    last_seq_no INTEGER NOT NULL,
                    state_hash TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def append(self, event: LedgerEvent) -> StoredEvent:
        payload_json = canonical_event_json(event)
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT event_id FROM ledger_events WHERE idempotency_key = ?",
                (event.idempotency_key,),
            ).fetchone()
            if existing is not None:
                raise DuplicateEventError(
                    f"Duplicate idempotency_key: {event.idempotency_key}",
                    details={
                        "idempotency_key": event.idempotency_key,
                        "existing_event_id": existing["event_id"],
                        "incoming_event_id": event.event_id,
                    },
                )

            existing_id = conn.execute(
                "SELECT event_id FROM ledger_events WHERE event_id = ?",
                (event.event_id,),
            ).fetchone()
            if existing_id is not None:
                raise DuplicateEventError(
                    f"Duplicate event_id: {event.event_id}",
                    details={"event_id": event.event_id},
                )

            last_row = conn.execute(
                "SELECT seq_no, event_hash FROM ledger_events ORDER BY seq_no DESC LIMIT 1"
            ).fetchone()
            next_seq = 1 if last_row is None else int(last_row["seq_no"]) + 1
            prev_hash = GENESIS_HASH if last_row is None else str(last_row["event_hash"])
            event_hash = compute_event_hash(prev_hash, next_seq, payload_json)

            conn.execute(
                """
                INSERT INTO ledger_events (
                    seq_no, event_id, idempotency_key, event_type,
                    event_time, recorded_time, prev_hash, event_hash, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    next_seq,
                    event.event_id,
                    event.idempotency_key,
                    event.event_type.value,
                    event.event_time.isoformat(),
                    event.recorded_time.isoformat(),
                    prev_hash,
                    event_hash,
                    payload_json,
                ),
            )

        return StoredEvent(
            seq_no=next_seq,
            event_id=event.event_id,
            idempotency_key=event.idempotency_key,
            event_type=event.event_type,
            event_time=event.event_time,
            recorded_time=event.recorded_time,
            prev_hash=prev_hash,
            event_hash=event_hash,
            event=event,
        )

    def _row_to_stored(self, row: sqlite3.Row) -> StoredEvent:
        event = parse_ledger_event(json.loads(row["payload_json"]))
        return StoredEvent(
            seq_no=int(row["seq_no"]),
            event_id=str(row["event_id"]),
            idempotency_key=str(row["idempotency_key"]),
            event_type=event.event_type,
            event_time=event.event_time,
            recorded_time=event.recorded_time,
            prev_hash=str(row["prev_hash"]),
            event_hash=str(row["event_hash"]),
            event=event,
        )

    def list_events(self, *, as_of: datetime | None = None) -> list[StoredEvent]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM ledger_events ORDER BY seq_no ASC").fetchall()
        events = [self._row_to_stored(r) for r in rows]
        if as_of is not None:
            events = [item for item in events if item.event_time <= as_of]
        return events

    def has_idempotency_key(self, idempotency_key: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM ledger_events WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        return row is not None

    def get_by_idempotency_key(self, idempotency_key: str) -> StoredEvent | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM ledger_events WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        return None if row is None else self._row_to_stored(row)

    def verify_hash_chain(self) -> tuple[bool, str]:
        prev_hash = GENESIS_HASH
        for stored in self.list_events():
            expected = compute_event_hash(
                prev_hash, stored.seq_no, canonical_event_json(stored.event)
            )
            if stored.prev_hash != prev_hash or stored.event_hash != expected:
                return False, f"Hash mismatch at seq_no={stored.seq_no}"
            prev_hash = stored.event_hash
        return True, prev_hash

    def write_cache(self, cache_key: str, last_seq_no: int, state_hash: str, snapshot_json: str, updated_at: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO materialized_state_cache (cache_key, last_seq_no, state_hash, snapshot_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    last_seq_no = excluded.last_seq_no,
                    state_hash = excluded.state_hash,
                    snapshot_json = excluded.snapshot_json,
                    updated_at = excluded.updated_at
                """,
                (cache_key, last_seq_no, state_hash, snapshot_json, updated_at),
            )

    def clear_cache(self) -> None:
        """Drop materialized read-model cache to test recovery via full deterministic replay."""
        with self._connect() as conn:
            conn.execute("DELETE FROM materialized_state_cache")

    def attempt_illegal_mutation(self, event_id: str) -> None:
        """Helper used in tests to prove SQLite triggers block history tampering."""
        try:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE ledger_events SET payload_json = '{}' WHERE event_id = ?",
                    (event_id,),
                )
        except sqlite3.DatabaseError as exc:
            raise ImmutabilityViolationError(
                f"History mutation blocked by database trigger: {exc}",
                details={"event_id": event_id},
            ) from exc
