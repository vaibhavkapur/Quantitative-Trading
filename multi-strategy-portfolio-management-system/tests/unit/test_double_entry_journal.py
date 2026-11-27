"""Unit tests for double-entry journal postings and trial balance invariants."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from pms.domain.errors import InvariantViolationError
from pms.ledger.journal import JournalEntry, JournalLine
from pms.service import PortfolioService


def test_unbalanced_journal_entry_is_rejected() -> None:
    bad_entry = JournalEntry(
        event_id="EVT-BAD",
        seq_no=1,
        event_time=datetime(2026, 1, 15, 15, 0, 0, tzinfo=UTC),
        lines=(
            JournalLine(
                account_id="ACCT-STRAT-MOM",
                gl_account="ASSET:CASH",
                currency="USD",
                debit=Decimal("100.00"),
                credit=Decimal("0"),
            ),
            JournalLine(
                account_id="ACCT-STRAT-MOM",
                gl_account="EQUITY:CAPITAL_CONTRIBUTED",
                currency="USD",
                debit=Decimal("0"),
                credit=Decimal("99.00"),
            ),
        ),
    )
    with pytest.raises(InvariantViolationError):
        bad_entry.assert_balanced()


def test_all_baseline_events_produce_balanced_trial_balance_per_currency(
    service_with_fixtures: PortfolioService,
) -> None:
    snap, journals = service_with_fixtures.replay(include_journals=True)
    assert snap.event_count == 17
    assert len(journals) == 17
    assert set(snap.trial_balance.keys()) == {"USD", "EUR", "GBP"}
    for ccy, totals in snap.trial_balance.items():
        assert totals["debits"] > Decimal("0"), f"Expected positive debits in {ccy}"
        assert totals["debits"] == totals["credits"], f"Out of balance in {ccy}"
