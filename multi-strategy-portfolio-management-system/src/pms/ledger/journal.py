"""Balanced double-entry journal postings with strict per-currency verification."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from decimal import Decimal

from pydantic import Field

from pms.domain.errors import InvariantViolationError
from pms.domain.models import FrozenModel

ZERO = Decimal("0")


class JournalLine(FrozenModel):
    account_id: str
    gl_account: str
    currency: str
    debit: Decimal = ZERO
    credit: Decimal = ZERO
    instrument_id: str | None = None
    memo: str = ""


class JournalEntry(FrozenModel):
    event_id: str
    seq_no: int
    event_time: datetime
    lines: tuple[JournalLine, ...] = Field(default_factory=tuple)

    def assert_balanced(self) -> None:
        """Verify sum(debits) == sum(credits) independently for every currency."""
        debits: dict[str, Decimal] = defaultdict(lambda: ZERO)
        credits: dict[str, Decimal] = defaultdict(lambda: ZERO)

        for line in self.lines:
            if line.debit < ZERO or line.credit < ZERO:
                raise InvariantViolationError(
                    f"Negative debit/credit in journal entry {self.event_id}",
                    details={"event_id": self.event_id, "line": line.model_dump(mode="json")},
                )
            debits[line.currency] += line.debit
            credits[line.currency] += line.credit

        for ccy in set(debits) | set(credits):
            if debits[ccy] != credits[ccy]:
                raise InvariantViolationError(
                    f"Unbalanced journal entry {self.event_id} in {ccy}: "
                    f"debits={debits[ccy]} credits={credits[ccy]}",
                    details={
                        "event_id": self.event_id,
                        "currency": ccy,
                        "debits": str(debits[ccy]),
                        "credits": str(credits[ccy]),
                    },
                )


def canonical_decimal(value: Decimal) -> Decimal:
    """Normalize trailing zeros without scientific notation or precision loss."""
    normalized = value.normalize()
    if normalized == ZERO:
        return ZERO
    return Decimal(format(normalized, "f"))


def verify_trial_balance(entries: list[JournalEntry]) -> dict[str, dict[str, Decimal]]:
    """Compute per-currency trial balance totals and assert strict double-entry equality."""
    totals: dict[str, dict[str, Decimal]] = {}
    for entry in entries:
        entry.assert_balanced()
        for line in entry.lines:
            bucket = totals.setdefault(line.currency, {"debits": ZERO, "credits": ZERO})
            bucket["debits"] += line.debit
            bucket["credits"] += line.credit

    for ccy, bucket in totals.items():
        if bucket["debits"] != bucket["credits"]:
            raise InvariantViolationError(
                f"Trial balance out of balance in {ccy}",
                details={"currency": ccy, "debits": str(bucket["debits"]), "credits": str(bucket["credits"])},
            )
        bucket["debits"] = canonical_decimal(bucket["debits"])
        bucket["credits"] = canonical_decimal(bucket["credits"])
    return totals
