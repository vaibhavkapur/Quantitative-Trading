"""Application service facade wiring master data, subledger, valuation, OMS, and reconciliation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from pms.adapters.broker_feed import BrokerStatement
from pms.adapters.paper_oms import FillIngestionResult, PaperFillReport, PaperOMSAdapter
from pms.domain.enums import Role
from pms.domain.errors import AuthorizationError, ValidationError
from pms.domain.master import MasterRegistry
from pms.domain.models import LedgerEvent, StoredEvent, parse_ledger_event
from pms.ledger.engine import LedgerEngine, ReplaySnapshot
from pms.ledger.journal import JournalEntry
from pms.ledger.store import EventStore, InMemoryEventStore, SQLiteEventStore
from pms.reconciliation.engine import ReconciliationEngine
from pms.reconciliation.models import (
    ReconciliationBreak,
    ReconciliationRun,
    ReconciliationTolerances,
)
from pms.reporting.cod import CODReportPackage, build_cod_report
from pms.valuation.consolidation import FundValuationReport, ValuationEngine
from pms.valuation.pricing import MarketSnapshot

ZERO = Decimal("0")


class PortfolioService:
    """High-level orchestration boundary for API, CLI, and validation scripts."""

    def __init__(
        self,
        *,
        master: MasterRegistry,
        store: EventStore,
        settings: dict[str, Any] | None = None,
    ) -> None:
        self.master = master
        self.store = store
        self.settings = settings or {}

        controls = self.settings.get("controls", {})
        tol_cfg = self.settings.get("reconciliation_tolerances", {})
        tolerances = ReconciliationTolerances(
            quantity=Decimal(str(tol_cfg.get("quantity", "0.000001"))),
            cash=Decimal(str(tol_cfg.get("cash", "0.01"))),
            price_pct=Decimal(str(tol_cfg.get("price_pct", "0.005"))),
        )

        self.ledger = LedgerEngine(master, store)
        self.valuation = ValuationEngine(
            master,
            mark_staleness_seconds=int(self.settings.get("mark_staleness_seconds", 86400)),
        )
        self.paper_oms = PaperOMSAdapter(
            self.ledger,
            paper_oms_enabled=bool(controls.get("paper_oms_enabled", True)),
            live_order_execution_enabled=bool(controls.get("live_order_execution_enabled", False)),
        )
        self.reconciliation = ReconciliationEngine(tolerances=tolerances)

        self._market_snapshots: dict[str, MarketSnapshot] = {}
        self._latest_snapshot_id: str | None = None
        self._latest_broker_statement: BrokerStatement | None = None

        self.metrics: dict[str, int] = {
            "events_ingested": 0,
            "duplicates_rejected": 0,
            "replays_executed": 0,
            "recon_runs": 0,
            "adjustments_approved": 0,
        }

    @classmethod
    def from_workspace(
        cls,
        root: Path | str,
        *,
        sqlite_path: Path | str | None = None,
        in_memory: bool = False,
        load_fixtures: bool = True,
    ) -> PortfolioService:
        base = Path(root)
        settings = json.loads((base / "config" / "settings.json").read_text(encoding="utf-8"))
        master = MasterRegistry.from_file(base / "config" / "accounts_and_instruments.json")

        if in_memory:
            store: EventStore = InMemoryEventStore()
        else:
            db_target = Path(sqlite_path) if sqlite_path else (base / settings["storage"]["sqlite_path"])
            store = SQLiteEventStore(db_target)

        svc = cls(master=master, store=store, settings=settings)

        if load_fixtures:
            events_file = base / "data" / "fixtures" / "baseline_events.json"
            marks_file = base / "data" / "fixtures" / "market_marks.json"
            broker_file = base / "data" / "fixtures" / "broker_statement.json"
            if events_file.exists():
                raw = json.loads(events_file.read_text(encoding="utf-8"))
                for item in raw["events"]:
                    evt = parse_ledger_event(item)
                    if not svc.store.has_idempotency_key(evt.idempotency_key):
                        svc.ingest_event(evt, role=Role.OPERATOR)
            if marks_file.exists():
                svc.register_market_snapshot(MarketSnapshot.from_file(marks_file), role=Role.OPERATOR)
            if broker_file.exists():
                stmt = BrokerStatement.from_file(broker_file)
                svc._latest_broker_statement = stmt
                svc.run_reconciliation(stmt, role=Role.OPERATOR)

        return svc

    @staticmethod
    def _require_role(role: Role, allowed: set[Role], action: str) -> None:
        if role not in allowed:
            raise AuthorizationError(
                f"Role {role.value} is not permitted to perform {action}",
                details={"role": role.value, "action": action, "allowed": sorted(r.value for r in allowed)},
            )

    def ingest_event(
        self, payload: dict[str, Any] | LedgerEvent, *, role: Role = Role.OPERATOR
    ) -> StoredEvent:
        self._require_role(role, {Role.OPERATOR, Role.APPROVER}, "ingest_event")
        event = parse_ledger_event(payload)
        try:
            stored = self.ledger.record_event(event)
            self.metrics["events_ingested"] += 1
            return stored
        except Exception:
            if self.store.has_idempotency_key(event.idempotency_key):
                self.metrics["duplicates_rejected"] += 1
            raise

    def ingest_paper_fill(
        self, report: PaperFillReport, *, role: Role = Role.OPERATOR
    ) -> FillIngestionResult:
        self._require_role(role, {Role.OPERATOR, Role.APPROVER}, "ingest_paper_fill")
        res = self.paper_oms.ingest_paper_fill(report)
        if res.status == "BOOKED":
            self.metrics["events_ingested"] += 1
        elif res.status == "DUPLICATE_IGNORED":
            self.metrics["duplicates_rejected"] += 1
        return res

    def register_market_snapshot(
        self, snapshot: MarketSnapshot, *, role: Role = Role.OPERATOR
    ) -> MarketSnapshot:
        self._require_role(role, {Role.OPERATOR, Role.APPROVER}, "register_market_snapshot")
        self._market_snapshots[snapshot.snapshot_id] = snapshot
        self._latest_snapshot_id = snapshot.snapshot_id
        return snapshot

    def get_market_snapshot(self, snapshot_id: str | None = None) -> MarketSnapshot:
        sid = snapshot_id or self._latest_snapshot_id
        if sid is None or sid not in self._market_snapshots:
            raise ValidationError(
                f"No market snapshot available (requested={snapshot_id!r})"
            )
        return self._market_snapshots[sid]

    def replay(
        self, *, as_of: datetime | None = None, include_journals: bool = False
    ) -> tuple[ReplaySnapshot, list[JournalEntry]]:
        self.metrics["replays_executed"] += 1
        snap, journals = self.ledger.replay(as_of=as_of, include_journals=include_journals)
        if isinstance(self.store, SQLiteEventStore) and as_of is None:
            self.store.write_cache(
                cache_key="latest_replay",
                last_seq_no=snap.last_seq_no,
                state_hash=snap.state_hash,
                snapshot_json=json.dumps(snap.model_dump(mode="json"), sort_keys=True),
                updated_at=datetime.now(UTC).isoformat(),
            )
        return snap, journals

    def get_valuation(
        self, *, snapshot_id: str | None = None, as_of: datetime | None = None
    ) -> FundValuationReport:
        market = self.get_market_snapshot(snapshot_id)
        replay_snap, _ = self.replay(as_of=as_of)
        return self.valuation.value_fund(replay_snap, market)

    def run_reconciliation(
        self,
        statement: BrokerStatement | None = None,
        *,
        snapshot_id: str | None = None,
        role: Role = Role.OPERATOR,
    ) -> ReconciliationRun:
        self._require_role(role, {Role.OPERATOR, Role.APPROVER}, "run_reconciliation")
        stmt = statement or self._latest_broker_statement
        if stmt is None:
            raise ValidationError("No broker statement supplied or loaded")
        self._latest_broker_statement = stmt
        replay_snap, _ = self.replay()
        val: FundValuationReport | None = None
        if self._latest_snapshot_id is not None:
            val = self.valuation.value_fund(replay_snap, self.get_market_snapshot(snapshot_id))
        run = self.reconciliation.reconcile(replay_snap, stmt, valuation=val)
        self.metrics["recon_runs"] += 1
        return run

    def resolve_break(
        self,
        *,
        break_id: str,
        actor: str,
        role: Role,
        reason: str,
        unit_cost: Decimal = ZERO,
    ) -> tuple[ReconciliationBreak, StoredEvent]:
        resolved, stored = self.reconciliation.resolve_break_with_adjustment(
            break_id=break_id,
            ledger=self.ledger,
            actor=actor,
            role=role,
            reason=reason,
            unit_cost=unit_cost,
        )
        self.metrics["adjustments_approved"] += 1
        self.metrics["events_ingested"] += 1
        return resolved, stored

    def build_cod_report(self, *, snapshot_id: str | None = None) -> CODReportPackage:
        replay_snap, _ = self.replay()
        market = self.get_market_snapshot(snapshot_id)
        val = self.valuation.value_fund(replay_snap, market)
        chain_valid, _ = self.store.verify_hash_chain()
        latest_recon = self.reconciliation.list_runs()[-1] if self.reconciliation.list_runs() else None
        return build_cod_report(
            replay=replay_snap,
            valuation=val,
            hash_chain_valid=chain_valid,
            reconciliation=latest_recon,
        )

    def verify_recovery_via_replay(self) -> dict[str, Any]:
        """Clear materialized cache, replay twice from raw event log, and verify bit-for-bit equality."""
        before_snap, _ = self.replay()
        cache_cleared = False
        if isinstance(self.store, SQLiteEventStore):
            self.store.clear_cache()
            cache_cleared = True
        after_snap_1, _ = self.replay()
        after_snap_2, _ = self.replay()
        chain_valid, tip_hash = self.store.verify_hash_chain()

        identical = (
            before_snap.state_hash == after_snap_1.state_hash == after_snap_2.state_hash
            and before_snap.accounts == after_snap_1.accounts == after_snap_2.accounts
        )
        return {
            "recovered_identically": identical,
            "cache_cleared": cache_cleared,
            "hash_chain_valid": chain_valid,
            "tip_event_hash": tip_hash,
            "before_state_hash": before_snap.state_hash,
            "after_replay_1_state_hash": after_snap_1.state_hash,
            "after_replay_2_state_hash": after_snap_2.state_hash,
            "event_count": after_snap_1.event_count,
            "last_seq_no": after_snap_1.last_seq_no,
        }

    def health(self) -> dict[str, Any]:
        chain_valid, tip_hash = self.store.verify_hash_chain()
        runs = self.reconciliation.list_runs()
        open_breaks = runs[-1].open_break_count if runs else 0
        unpriced_count = 0
        if self._latest_snapshot_id is not None:
            val = self.get_valuation()
            unpriced_count = len(val.unpriced_instruments)
        return {
            "status": "OK" if chain_valid else "DEGRADED",
            "system_name": self.settings.get("system_name", "SAAT PMS"),
            "environment": self.settings.get("environment", "local_prototype"),
            "fund_id": self.master.fund.fund_id,
            "base_currency": self.master.fund.base_currency,
            "hash_chain_valid": chain_valid,
            "tip_event_hash": tip_hash,
            "event_count": len(self.store.list_events()),
            "latest_snapshot_id": self._latest_snapshot_id,
            "open_reconciliation_breaks": open_breaks,
            "unpriced_instruments_count": unpriced_count,
            "controls": {
                "live_order_execution_enabled": self.paper_oms.live_order_execution_enabled,
                "paper_oms_enabled": self.paper_oms.paper_oms_enabled,
                "allow_history_mutation": False,
            },
            "metrics": dict(self.metrics),
        }
