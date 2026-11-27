"""Unit tests for FIFO/LIFO/AVG_COST lot accounting, flips, multipliers, and splits."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from pms.domain.enums import LotMethod, Side
from pms.ledger.lots import LotBook


def test_fifo_vs_lifo_vs_avg_cost_realized_pnl() -> None:
    t0 = datetime(2026, 1, 15, 14, 0, 0, tzinfo=UTC)
    results: dict[LotMethod, Decimal] = {}

    for method in (LotMethod.FIFO, LotMethod.LIFO, LotMethod.AVG_COST):
        book = LotBook(
            account_id="A1",
            instrument_id="EQ-AAPL-US",
            currency="USD",
            multiplier=Decimal("1"),
            lot_method=method,
        )
        book.apply_trade(seq_no=1, event_id="E1", event_time=t0, side=Side.BUY, quantity=Decimal("100"), price=Decimal("100"))
        book.apply_trade(seq_no=2, event_id="E2", event_time=t0, side=Side.BUY, quantity=Decimal("100"), price=Decimal("120"))
        out = book.apply_trade(seq_no=3, event_id="E3", event_time=t0, side=Side.SELL, quantity=Decimal("100"), price=Decimal("130"))
        results[method] = out.total_realized_pnl

    assert results[LotMethod.FIFO] == Decimal("3000")      # sold 100 @ 130 against 100 cost
    assert results[LotMethod.LIFO] == Decimal("1000")      # sold 100 @ 130 against 120 cost
    assert results[LotMethod.AVG_COST] == Decimal("2000")  # sold 100 @ 130 against 110 avg


def test_long_to_short_and_short_to_long_flips_with_futures_multiplier() -> None:
    t0 = datetime(2026, 1, 15, 14, 0, 0, tzinfo=UTC)
    book = LotBook(
        account_id="A1",
        instrument_id="FUT-ES-202606",
        currency="USD",
        multiplier=Decimal("50"),
        lot_method=LotMethod.FIFO,
    )
    book.apply_trade(seq_no=1, event_id="E1", event_time=t0, side=Side.BUY, quantity=Decimal("2"), price=Decimal("5900"))
    flip1 = book.apply_trade(seq_no=2, event_id="E2", event_time=t0, side=Side.SELL, quantity=Decimal("5"), price=Decimal("5910"))
    assert flip1.flipped is True
    assert flip1.closed_long_qty == Decimal("2")
    assert flip1.opened_short_qty == Decimal("3")
    assert flip1.total_realized_pnl == Decimal("1000")  # 2 * (5910 - 5900) * 50
    assert book.net_quantity == Decimal("-3")

    flip2 = book.apply_trade(seq_no=3, event_id="E3", event_time=t0, side=Side.BUY, quantity=Decimal("4"), price=Decimal("5900"))
    assert flip2.flipped is True
    assert flip2.closed_short_qty == Decimal("3")
    assert flip2.opened_long_qty == Decimal("1")
    assert flip2.total_realized_pnl == Decimal("1500")  # 3 * (5910 - 5900) * 50
    assert book.net_quantity == Decimal("1")
    assert book.average_unit_cost == Decimal("5900")


def test_split_preserves_exact_total_cost_basis_even_for_three_for_one() -> None:
    t0 = datetime(2026, 1, 15, 14, 0, 0, tzinfo=UTC)
    book = LotBook(
        account_id="A1",
        instrument_id="EQ-AAPL-US",
        currency="USD",
        multiplier=Decimal("1"),
    )
    book.apply_trade(seq_no=1, event_id="E1", event_time=t0, side=Side.BUY, quantity=Decimal("100"), price=Decimal("100"))
    before_basis = book.signed_cost_basis
    book.apply_split(Decimal("3"), Decimal("1"))
    assert book.net_quantity == Decimal("300")
    assert book.signed_cost_basis == before_basis  # Exact $10,000.00 preserved!

    # Closing all 300 shares at 100/3 yields exact 0 realized P&L when sold at exact basis
    close_out = book.apply_trade(seq_no=2, event_id="E2", event_time=t0, side=Side.SELL, quantity=Decimal("300"), price=Decimal("40"))
    assert close_out.closed_long_cost == Decimal("10000")
    assert close_out.total_realized_pnl == Decimal("2000")  # 12000 - 10000
