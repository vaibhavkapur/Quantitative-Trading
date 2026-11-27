"""Integration tests for Paper OMS controls and Approver-gated broker reconciliation."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from pms.adapters.paper_oms import PaperFillReport
from pms.domain.enums import BreakStatus, BreakType, Role, Side
from pms.domain.errors import AuthorizationError, ControlViolationError
from pms.service import PortfolioService


def test_live_order_execution_is_blocked_by_control_policy(
    service_with_fixtures: PortfolioService,
) -> None:
    with pytest.raises(ControlViolationError):
        service_with_fixtures.paper_oms.submit_live_order(symbol="AAPL", qty=100)


def test_paper_oms_fill_ingestion_and_duplicate_deduplication(
    service_with_fixtures: PortfolioService,
) -> None:
    fill = PaperFillReport(
        order_id="ORD-PAPER-900",
        execution_id="EXEC-PAPER-900-1",
        account_id="ACCT-STRAT-MOM",
        instrument_id="EQ-AAPL-US",
        side=Side.BUY,
        quantity=Decimal("50"),
        price=Decimal("98.50"),
        currency="USD",
        fee=Decimal("1.00"),
        fill_time=datetime(2026, 1, 15, 20, 30, 0, tzinfo=UTC),
    )
    first = service_with_fixtures.ingest_paper_fill(fill, role=Role.OPERATOR)
    assert first.status == "BOOKED"
    assert first.seq_no == 18

    second = service_with_fixtures.ingest_paper_fill(fill, role=Role.OPERATOR)
    assert second.status == "DUPLICATE_IGNORED"
    assert second.seq_no == 18
    assert len(service_with_fixtures.store.list_events()) == 18


def test_reconciliation_never_mutates_history_and_requires_approver_to_resolve(
    service_with_fixtures: PortfolioService,
) -> None:
    events_before = service_with_fixtures.store.list_events()
    runs = service_with_fixtures.reconciliation.list_runs()
    assert len(runs) == 1
    run = runs[0]
    assert run.mutated_history is False
    assert len(run.breaks) == 2

    break_types = {b.break_type for b in run.breaks}
    assert break_types == {BreakType.POSITION_QUANTITY_BREAK, BreakType.CASH_BALANCE_BREAK}

    # Verify history was completely untouched by reconciliation
    assert service_with_fixtures.store.list_events() == events_before

    # OPERATOR cannot approve an adjustment
    first_break = run.breaks[0]
    with pytest.raises(AuthorizationError):
        service_with_fixtures.resolve_break(
            break_id=first_break.break_id,
            actor="junior_ops",
            role=Role.OPERATOR,
            reason="Attempted unauthorized adjustment",
            unit_cost=Decimal("200.00"),
        )

    # APPROVER resolves both breaks by appending new audited AdjustmentEvents
    for brk in run.breaks:
        resolved, stored = service_with_fixtures.resolve_break(
            break_id=brk.break_id,
            actor="chief_risk_officer",
            role=Role.APPROVER,
            reason="Broker confirmed break via T+0 ops ticket",
            unit_cost=Decimal("200.00") if brk.instrument_id else Decimal("0"),
        )
        assert resolved.status == BreakStatus.RESOLVED_BY_ADJUSTMENT
        assert stored.event_id.startswith("EVT-ADJ-")

    # Historical events 1..17 remain intact; 2 new immutable events appended -> 19 total
    events_after = service_with_fixtures.store.list_events()
    assert len(events_after) == 19
    assert events_after[:17] == events_before

    # Re-running reconciliation now produces zero breaks and P&L remains 100% explained
    clean_run = service_with_fixtures.run_reconciliation(role=Role.OPERATOR)
    assert clean_run.is_clean is True
    assert len(clean_run.breaks) == 0

    val_after = service_with_fixtures.get_valuation()
    assert val_after.consolidation_check.passed is True
    assert val_after.pnl.unexplained_residual_base == Decimal("0")
