"""Read-only reconciliation engine and approver-gated adjustment workflow."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from pms.adapters.broker_feed import BrokerStatement
from pms.domain.enums import AdjustmentType, BreakSeverity, BreakStatus, BreakType, Role
from pms.domain.errors import AuthorizationError, ValidationError
from pms.domain.models import AdjustmentEvent, StoredEvent
from pms.ledger.engine import LedgerEngine, ReplaySnapshot
from pms.reconciliation.models import (
    ReconciliationBreak,
    ReconciliationRun,
    ReconciliationTolerances,
)
from pms.valuation.consolidation import FundValuationReport

ZERO = Decimal("0")


class ReconciliationEngine:
    """Compares internal ledger state to broker statements without ever mutating history."""

    def __init__(self, tolerances: ReconciliationTolerances | None = None) -> None:
        self.tolerances = tolerances or ReconciliationTolerances()
        self._runs: list[ReconciliationRun] = []
        self._breaks_by_id: dict[str, ReconciliationBreak] = {}

    def list_runs(self) -> list[ReconciliationRun]:
        return list(self._runs)

    def get_break(self, break_id: str) -> ReconciliationBreak:
        if break_id not in self._breaks_by_id:
            raise ValidationError(f"Unknown reconciliation break_id: {break_id}")
        return self._breaks_by_id[break_id]

    def reconcile(
        self,
        replay: ReplaySnapshot,
        statement: BrokerStatement,
        *,
        valuation: FundValuationReport | None = None,
        executed_at: datetime | None = None,
    ) -> ReconciliationRun:
        run_num = len(self._runs) + 1
        run_id = f"RECON-{run_num:04d}-{statement.statement_id}"
        ts = executed_at or datetime.now(UTC)

        # Build lookup maps from internal replay snapshot
        internal_pos: dict[tuple[str, str], tuple[Decimal, str]] = {}
        for acct_id, acct_state in replay.accounts.items():
            for inst_id, pos in acct_state.positions.items():
                if pos.quantity != ZERO:
                    internal_pos[(acct_id, inst_id)] = (pos.quantity, pos.currency)

        internal_cash: dict[tuple[str, str], Decimal] = {}
        for acct_id, acct_state in replay.accounts.items():
            for ccy, bal in acct_state.cash_balances.items():
                if bal != ZERO:
                    internal_cash[(acct_id, ccy)] = bal

        internal_marks: dict[tuple[str, str], Decimal] = {}
        if valuation is not None:
            for acct_val in (*valuation.strategy_valuations, *valuation.unallocated_valuations):
                for vpos in acct_val.positions:
                    if vpos.mark_price is not None:
                        internal_marks[(acct_val.account_id, vpos.instrument_id)] = vpos.mark_price

        broker_pos = {
            (p.account_id, p.instrument_id): p
            for p in statement.positions
            if p.quantity != ZERO
        }
        broker_cash = {
            (c.account_id, c.currency): c.balance
            for c in statement.cash_balances
            if c.balance != ZERO
        }

        breaks: list[ReconciliationBreak] = []
        break_idx = 0

        # 1. Reconcile positions
        all_pos_keys = sorted(set(internal_pos) | set(broker_pos))
        for key in all_pos_keys:
            acct_id, inst_id = key
            in_rec = internal_pos.get(key)
            br_rec = broker_pos.get(key)
            in_qty = in_rec[0] if in_rec else ZERO
            br_qty = br_rec.quantity if br_rec else ZERO
            ccy = in_rec[1] if in_rec else (br_rec.currency if br_rec else "USD")
            diff = br_qty - in_qty

            if in_rec is None and br_rec is not None:
                break_idx += 1
                breaks.append(
                    ReconciliationBreak(
                        break_id=f"{run_id}-BRK-{break_idx:03d}",
                        run_id=run_id,
                        break_type=BreakType.MISSING_INTERNAL_POSITION,
                        severity=BreakSeverity.HIGH,
                        account_id=acct_id,
                        instrument_id=inst_id,
                        currency=ccy,
                        internal_value=ZERO,
                        broker_value=br_qty,
                        difference=diff,
                        description=(
                            f"Broker reports {br_qty} of {inst_id} in {acct_id}, "
                            "missing from internal ledger."
                        ),
                    )
                )
            elif br_rec is None and in_rec is not None:
                break_idx += 1
                breaks.append(
                    ReconciliationBreak(
                        break_id=f"{run_id}-BRK-{break_idx:03d}",
                        run_id=run_id,
                        break_type=BreakType.MISSING_BROKER_POSITION,
                        severity=BreakSeverity.HIGH,
                        account_id=acct_id,
                        instrument_id=inst_id,
                        currency=ccy,
                        internal_value=in_qty,
                        broker_value=ZERO,
                        difference=diff,
                        description=(
                            f"Internal ledger holds {in_qty} of {inst_id} in {acct_id}, "
                            "missing from broker statement."
                        ),
                    )
                )
            elif abs(diff) > self.tolerances.quantity:
                break_idx += 1
                breaks.append(
                    ReconciliationBreak(
                        break_id=f"{run_id}-BRK-{break_idx:03d}",
                        run_id=run_id,
                        break_type=BreakType.POSITION_QUANTITY_BREAK,
                        severity=BreakSeverity.HIGH,
                        account_id=acct_id,
                        instrument_id=inst_id,
                        currency=ccy,
                        internal_value=in_qty,
                        broker_value=br_qty,
                        difference=diff,
                        description=(
                            f"Position quantity mismatch for {inst_id} in {acct_id}: "
                            f"internal={in_qty}, broker={br_qty}, diff={diff}."
                        ),
                    )
                )

            # Price discrepancy check when both sides report a mark price
            if (
                br_rec is not None
                and br_rec.broker_mark_price is not None
                and key in internal_marks
            ):
                in_px = internal_marks[key]
                br_px = br_rec.broker_mark_price
                if in_px > ZERO and abs(br_px - in_px) / in_px > self.tolerances.price_pct:
                    break_idx += 1
                    breaks.append(
                        ReconciliationBreak(
                            break_id=f"{run_id}-BRK-{break_idx:03d}",
                            run_id=run_id,
                            break_type=BreakType.PRICE_DISCREPANCY,
                            severity=BreakSeverity.MEDIUM,
                            account_id=acct_id,
                            instrument_id=inst_id,
                            currency=ccy,
                            internal_value=in_px,
                            broker_value=br_px,
                            difference=br_px - in_px,
                            description=(
                                f"Mark price discrepancy for {inst_id}: "
                                f"internal={in_px}, broker={br_px}."
                            ),
                        )
                    )

        # 2. Reconcile cash balances
        all_cash_keys = sorted(set(internal_cash) | set(broker_cash))
        for key in all_cash_keys:
            acct_id, ccy = key
            in_bal = internal_cash.get(key, ZERO)
            br_bal = broker_cash.get(key, ZERO)
            diff = br_bal - in_bal
            if abs(diff) > self.tolerances.cash:
                break_idx += 1
                breaks.append(
                    ReconciliationBreak(
                        break_id=f"{run_id}-BRK-{break_idx:03d}",
                        run_id=run_id,
                        break_type=BreakType.CASH_BALANCE_BREAK,
                        severity=BreakSeverity.HIGH if abs(diff) >= Decimal("100") else BreakSeverity.MEDIUM,
                        account_id=acct_id,
                        instrument_id=None,
                        currency=ccy,
                        internal_value=in_bal,
                        broker_value=br_bal,
                        difference=diff,
                        description=(
                            f"Cash balance mismatch in {acct_id} ({ccy}): "
                            f"internal={in_bal}, broker={br_bal}, diff={diff}."
                        ),
                    )
                )

        run = ReconciliationRun(
            run_id=run_id,
            statement_id=statement.statement_id,
            broker_name=statement.broker_name,
            executed_at=ts,
            ledger_last_seq_no=replay.last_seq_no,
            ledger_state_hash=replay.state_hash,
            positions_checked=len(all_pos_keys),
            cash_balances_checked=len(all_cash_keys),
            breaks=tuple(breaks),
            mutated_history=False,
        )
        self._runs.append(run)
        for brk in breaks:
            self._breaks_by_id[brk.break_id] = brk
        return run

    def resolve_break_with_adjustment(
        self,
        *,
        break_id: str,
        ledger: LedgerEngine,
        actor: str,
        role: Role,
        reason: str,
        unit_cost: Decimal = ZERO,
        effective_time: datetime | None = None,
    ) -> tuple[ReconciliationBreak, StoredEvent]:
        """Resolve an open break by appending an explicit, approver-authorized AdjustmentEvent.

        Never edits or deletes historical events.
        """
        if role != Role.APPROVER:
            raise AuthorizationError(
                "Only an APPROVER role can authorize a reconciliation adjustment event",
                details={"actor": actor, "role": role.value, "break_id": break_id},
            )
        brk = self.get_break(break_id)
        if brk.status not in {BreakStatus.OPEN, BreakStatus.INVESTIGATING}:
            raise ValidationError(
                f"Break {break_id} is already in terminal state {brk.status.value}",
                details={"break_id": break_id, "status": brk.status.value},
            )

        ts = effective_time or datetime.now(UTC)
        event_id = f"EVT-ADJ-{break_id}"
        idem_key = f"ADJ-{break_id}"

        if brk.break_type == BreakType.CASH_BALANCE_BREAK:
            adj_event = AdjustmentEvent(
                event_id=event_id,
                idempotency_key=idem_key,
                event_time=ts,
                recorded_time=ts,
                account_id=brk.account_id,
                adjustment_type=AdjustmentType.CASH,
                instrument_id=None,
                currency=brk.currency,
                cash_delta=brk.difference,
                quantity_delta=ZERO,
                unit_cost=ZERO,
                reconciliation_break_id=brk.break_id,
                approved_by=actor,
                reason=reason,
            )
        elif brk.break_type in {
            BreakType.POSITION_QUANTITY_BREAK,
            BreakType.MISSING_INTERNAL_POSITION,
            BreakType.MISSING_BROKER_POSITION,
        }:
            adj_event = AdjustmentEvent(
                event_id=event_id,
                idempotency_key=idem_key,
                event_time=ts,
                recorded_time=ts,
                account_id=brk.account_id,
                adjustment_type=AdjustmentType.POSITION,
                instrument_id=brk.instrument_id,
                currency=brk.currency,
                cash_delta=ZERO,
                quantity_delta=brk.difference,
                unit_cost=unit_cost,
                reconciliation_break_id=brk.break_id,
                approved_by=actor,
                reason=reason,
            )
        else:
            raise ValidationError(
                f"Break type {brk.break_type.value} cannot be resolved via ledger adjustment",
                details={"break_id": break_id, "break_type": brk.break_type.value},
            )

        stored = ledger.record_event(adj_event)
        resolved = brk.model_copy(
            update={
                "status": BreakStatus.RESOLVED_BY_ADJUSTMENT,
                "resolution_event_id": stored.event_id,
                "resolved_by": actor,
                "resolution_note": reason,
            }
        )
        self._breaks_by_id[break_id] = resolved
        # Update break inside stored runs
        updated_runs: list[ReconciliationRun] = []
        for run in self._runs:
            if run.run_id == brk.run_id:
                new_breaks = tuple(resolved if b.break_id == break_id else b for b in run.breaks)
                updated_runs.append(run.model_copy(update={"breaks": new_breaks}))
            else:
                updated_runs.append(run)
        self._runs = updated_runs
        return resolved, stored
