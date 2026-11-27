"""Reconciliation contracts and break records."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import Field

from pms.domain.enums import BreakSeverity, BreakStatus, BreakType
from pms.domain.models import FrozenModel


class ReconciliationTolerances(FrozenModel):
    quantity: Decimal = Decimal("0.000001")
    cash: Decimal = Decimal("0.01")
    price_pct: Decimal = Decimal("0.005")


class ReconciliationBreak(FrozenModel):
    break_id: str
    run_id: str
    break_type: BreakType
    severity: BreakSeverity
    status: BreakStatus = BreakStatus.OPEN
    account_id: str
    instrument_id: str | None = None
    currency: str
    internal_value: Decimal
    broker_value: Decimal
    difference: Decimal
    description: str
    resolution_event_id: str | None = None
    resolved_by: str | None = None
    resolution_note: str | None = None


class ReconciliationRun(FrozenModel):
    run_id: str
    statement_id: str
    broker_name: str
    executed_at: datetime
    ledger_last_seq_no: int
    ledger_state_hash: str
    positions_checked: int
    cash_balances_checked: int
    breaks: tuple[ReconciliationBreak, ...] = Field(default_factory=tuple)
    mutated_history: bool = False  # Always False; reconciliation is strictly read-only

    @property
    def is_clean(self) -> bool:
        return len(self.breaks) == 0

    @property
    def open_break_count(self) -> int:
        return sum(1 for b in self.breaks if b.status in {BreakStatus.OPEN, BreakStatus.INVESTIGATING})
