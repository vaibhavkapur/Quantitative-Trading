"""Paper OMS fill ingestion adapter with live-execution safety guard and deduplication."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from pydantic import Field

from pms.domain.enums import Side
from pms.domain.errors import ControlViolationError, DuplicateEventError
from pms.domain.models import FrozenModel, StoredEvent, TradeFillEvent
from pms.ledger.engine import LedgerEngine

ZERO = Decimal("0")


class PaperFillReport(FrozenModel):
    order_id: str = Field(min_length=1)
    execution_id: str = Field(min_length=1)
    account_id: str = Field(min_length=1)
    instrument_id: str = Field(min_length=1)
    side: Side
    quantity: Decimal
    price: Decimal
    currency: str
    fee: Decimal = ZERO
    venue: str = "PAPER_SIM"
    fill_time: datetime
    idempotency_key: str | None = None


class FillIngestionResult(FrozenModel):
    status: str  # "BOOKED" | "DUPLICATE_IGNORED"
    execution_id: str
    idempotency_key: str
    event_id: str
    seq_no: int | None
    message: str


class PaperOMSAdapter:
    """Adapter translating paper OMS execution reports into immutable TradeFillEvents.

    Enforces Section 7 operational controls: live order execution is disabled by default.
    """

    def __init__(
        self,
        ledger: LedgerEngine,
        *,
        paper_oms_enabled: bool = True,
        live_order_execution_enabled: bool = False,
    ) -> None:
        self.ledger = ledger
        self.paper_oms_enabled = paper_oms_enabled
        self.live_order_execution_enabled = live_order_execution_enabled

    def submit_live_order(self, **_: object) -> None:
        if not self.live_order_execution_enabled:
            raise ControlViolationError(
                "Live order execution is disabled by operational control policy. "
                "Real-capital trading requires separate formal approval."
            )
        raise ControlViolationError("No live broker gateway is attached to this prototype.")

    def ingest_paper_fill(
        self,
        report: PaperFillReport,
        *,
        recorded_time: datetime | None = None,
    ) -> FillIngestionResult:
        if not self.paper_oms_enabled:
            raise ControlViolationError("Paper OMS ingestion is disabled in configuration.")

        idem_key = report.idempotency_key or f"OMS-EXEC-{report.execution_id}"
        event_id = f"EVT-OMS-{report.execution_id}"
        now_utc = recorded_time or datetime.now(UTC)

        event = TradeFillEvent(
            event_id=event_id,
            idempotency_key=idem_key,
            event_time=report.fill_time,
            recorded_time=now_utc,
            account_id=report.account_id,
            instrument_id=report.instrument_id,
            order_id=report.order_id,
            execution_id=report.execution_id,
            side=report.side,
            quantity=report.quantity,
            price=report.price,
            currency=report.currency,
            fee=report.fee,
            venue=report.venue,
        )

        try:
            stored: StoredEvent = self.ledger.record_event(event)
            return FillIngestionResult(
                status="BOOKED",
                execution_id=report.execution_id,
                idempotency_key=idem_key,
                event_id=stored.event_id,
                seq_no=stored.seq_no,
                message=f"Paper fill booked at seq_no={stored.seq_no}",
            )
        except DuplicateEventError as exc:
            existing = self.ledger.store.get_by_idempotency_key(idem_key)
            return FillIngestionResult(
                status="DUPLICATE_IGNORED",
                execution_id=report.execution_id,
                idempotency_key=idem_key,
                event_id=existing.event_id if existing else event_id,
                seq_no=existing.seq_no if existing else None,
                message=f"Duplicate fill safely ignored: {exc.message}",
            )
