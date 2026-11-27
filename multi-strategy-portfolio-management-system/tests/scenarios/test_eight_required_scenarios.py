"""Verification tests covering all 8 required scenarios from Section 5 of plan.md."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from pms.adapters.broker_feed import BrokerCashRecord, BrokerPositionRecord, BrokerStatement
from pms.adapters.paper_oms import PaperFillReport
from pms.domain.enums import (
    BreakStatus,
    BreakType,
    CashDirection,
    CorporateActionType,
    PricingStatus,
    Role,
    Side,
)
from pms.domain.errors import DuplicateEventError
from pms.domain.master import MasterRegistry
from pms.domain.models import (
    CashTransferEvent,
    CorporateActionEvent,
    FeeEvent,
    FxConversionEvent,
    StrategyAllocationEvent,
    TradeFillEvent,
)
from pms.ledger.store import InMemoryEventStore
from pms.service import PortfolioService
from pms.valuation.pnl import compute_pnl_bridge
from pms.valuation.pricing import FxRateMark, MarketSnapshot, PriceMark


def _fresh_service(master: MasterRegistry) -> PortfolioService:
    svc = PortfolioService(master=master, store=InMemoryEventStore())
    t0 = datetime(2026, 1, 15, 13, 0, 0, tzinfo=UTC)
    svc.ingest_event(
        CashTransferEvent(
            event_id="EVT-DEP-1",
            idempotency_key="DEP-1",
            event_time=t0,
            recorded_time=t0,
            account_id="ACCT-FUND-UNALLOC",
            currency="USD",
            amount=Decimal("500000.00"),
            direction=CashDirection.DEPOSIT,
        )
    )
    svc.ingest_event(
        StrategyAllocationEvent(
            event_id="EVT-ALLOC-1",
            idempotency_key="ALLOC-1",
            event_time=t0,
            recorded_time=t0,
            from_account_id="ACCT-FUND-UNALLOC",
            to_account_id="ACCT-STRAT-MOM",
            currency="USD",
            amount=Decimal("300000.00"),
        )
    )
    return svc


def test_scenario_1_duplicate_fill(master: MasterRegistry) -> None:
    """Scenario 1: Duplicate fill is rejected/deduplicated without double-counting cash, qty, or fees."""
    svc = _fresh_service(master)
    t1 = datetime(2026, 1, 15, 14, 0, 0, tzinfo=UTC)
    fill = TradeFillEvent(
        event_id="EVT-FILL-DUP-1",
        idempotency_key="EXEC-DUP-001",
        event_time=t1,
        recorded_time=t1,
        account_id="ACCT-STRAT-MOM",
        instrument_id="EQ-AAPL-US",
        order_id="ORD-DUP",
        execution_id="EXEC-DUP-001",
        side=Side.BUY,
        quantity=Decimal("100"),
        price=Decimal("200.00"),
        currency="USD",
        fee=Decimal("5.00"),
    )
    svc.ingest_event(fill)
    snap_after_first, _ = svc.replay()

    with pytest.raises(DuplicateEventError):
        svc.ingest_event(fill.model_copy(update={"event_id": "EVT-FILL-DUP-2"}))

    # Also test via Paper OMS idempotent ingestion path
    oms_res = svc.ingest_paper_fill(
        PaperFillReport(
            order_id="ORD-DUP",
            execution_id="EXEC-DUP-001",
            idempotency_key="EXEC-DUP-001",
            account_id="ACCT-STRAT-MOM",
            instrument_id="EQ-AAPL-US",
            side=Side.BUY,
            quantity=Decimal("100"),
            price=Decimal("200.00"),
            currency="USD",
            fee=Decimal("5.00"),
            fill_time=t1,
        )
    )
    assert oms_res.status == "DUPLICATE_IGNORED"

    snap_after_dup, _ = svc.replay()
    assert snap_after_dup.state_hash == snap_after_first.state_hash
    assert snap_after_dup.accounts["ACCT-STRAT-MOM"].positions["EQ-AAPL-US"].quantity == Decimal("100")
    assert snap_after_dup.accounts["ACCT-STRAT-MOM"].cash_balances["USD"] == Decimal("279995.00")


def test_scenario_2_position_flip(master: MasterRegistry) -> None:
    """Scenario 2: Position flip from Long 200 -> Short 100 realizes exact P&L and opens short lot."""
    svc = _fresh_service(master)
    t1 = datetime(2026, 1, 15, 14, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 1, 15, 15, 0, 0, tzinfo=UTC)
    svc.ingest_event(
        TradeFillEvent(
            event_id="EVT-FLIP-1",
            idempotency_key="FLIP-1",
            event_time=t1,
            recorded_time=t1,
            account_id="ACCT-STRAT-MOM",
            instrument_id="EQ-MSFT-US",
            order_id="O-FLIP-1",
            execution_id="X-FLIP-1",
            side=Side.BUY,
            quantity=Decimal("200"),
            price=Decimal("400.00"),
            currency="USD",
            fee=Decimal("2.00"),
        )
    )
    svc.ingest_event(
        TradeFillEvent(
            event_id="EVT-FLIP-2",
            idempotency_key="FLIP-2",
            event_time=t2,
            recorded_time=t2,
            account_id="ACCT-STRAT-MOM",
            instrument_id="EQ-MSFT-US",
            order_id="O-FLIP-2",
            execution_id="X-FLIP-2",
            side=Side.SELL,
            quantity=Decimal("300"),
            price=Decimal("410.00"),
            currency="USD",
            fee=Decimal("3.00"),
        )
    )
    snap, _ = svc.replay()
    acct = snap.accounts["ACCT-STRAT-MOM"]
    pos = acct.positions["EQ-MSFT-US"]
    assert pos.quantity == Decimal("-100")
    assert pos.average_unit_cost == Decimal("410.00")
    assert acct.realized_trading_pnl["USD"] == Decimal("2000.00")
    assert acct.cash_balances["USD"] == Decimal("342995.00")  # 300k - 80002 + 123000 - 3


def test_scenario_3_partial_fill(master: MasterRegistry) -> None:
    """Scenario 3: Order filled in 3 partial fills tracks cumulative fill qty, FIFO lots, and avg cost."""
    svc = _fresh_service(master)
    t1 = datetime(2026, 1, 15, 14, 0, 0, tzinfo=UTC)
    fills = [
        ("P1", Decimal("100"), Decimal("100.00"), Decimal("1.00")),
        ("P2", Decimal("200"), Decimal("103.00"), Decimal("2.00")),
        ("P3", Decimal("100"), Decimal("106.00"), Decimal("1.00")),
    ]
    for suffix, qty, px, fee in fills:
        svc.ingest_event(
            TradeFillEvent(
                event_id=f"EVT-{suffix}",
                idempotency_key=f"KEY-{suffix}",
                event_time=t1,
                recorded_time=t1,
                account_id="ACCT-STRAT-MOM",
                instrument_id="EQ-AAPL-US",
                order_id="ORD-PARTIAL-1",
                execution_id=f"EXEC-{suffix}",
                side=Side.BUY,
                quantity=qty,
                price=px,
                currency="USD",
                fee=fee,
            )
        )
    snap, _ = svc.replay()
    assert snap.order_fill_tracker["ORD-PARTIAL-1"] == Decimal("400")
    pos = snap.accounts["ACCT-STRAT-MOM"].positions["EQ-AAPL-US"]
    assert pos.quantity == Decimal("400")
    assert len(pos.lots) == 3
    assert pos.average_unit_cost == Decimal("103.00")  # (10000 + 20600 + 10600) / 400


def test_scenario_4_multi_currency_cash(master: MasterRegistry) -> None:
    """Scenario 4: Multi-currency FX conversion + EUR trade decomposes price P&L vs FX P&L exactly."""
    svc = _fresh_service(master)
    t1 = datetime(2026, 1, 15, 14, 0, 0, tzinfo=UTC)
    svc.ingest_event(
        FxConversionEvent(
            event_id="EVT-FX-1",
            idempotency_key="FX-1",
            event_time=t1,
            recorded_time=t1,
            account_id="ACCT-STRAT-MOM",
            sell_currency="USD",
            sell_amount=Decimal("110000.00"),
            buy_currency="EUR",
            buy_amount=Decimal("100000.00"),
            fx_rate=Decimal("1.1000"),
            fee_currency="USD",
            fee_amount=Decimal("10.00"),
        )
    )
    svc.ingest_event(
        TradeFillEvent(
            event_id="EVT-EUR-1",
            idempotency_key="EUR-1",
            event_time=t1,
            recorded_time=t1,
            account_id="ACCT-STRAT-MOM",
            instrument_id="EQ-SAP-DE",
            order_id="O-EUR-1",
            execution_id="X-EUR-1",
            side=Side.BUY,
            quantity=Decimal("400"),
            price=Decimal("200.00"),
            currency="EUR",
            fee=Decimal("15.00"),
        )
    )
    market = MarketSnapshot(
        snapshot_id="SNAP-MCY",
        as_of_time=datetime(2026, 1, 15, 21, 0, 0, tzinfo=UTC),
        availability_time=datetime(2026, 1, 15, 21, 1, 0, tzinfo=UTC),
        prices=(
            PriceMark(
                instrument_id="EQ-SAP-DE",
                price=Decimal("206.00"),
                currency="EUR",
                price_timestamp=datetime(2026, 1, 15, 21, 0, 0, tzinfo=UTC),
                availability_timestamp=datetime(2026, 1, 15, 21, 1, 0, tzinfo=UTC),
                source="SYNTHETIC",
            ),
        ),
        fx_rates=(
            FxRateMark(
                from_currency="EUR",
                to_currency="USD",
                rate=Decimal("1.1200"),
                rate_timestamp=datetime(2026, 1, 15, 21, 0, 0, tzinfo=UTC),
                availability_timestamp=datetime(2026, 1, 15, 21, 1, 0, tzinfo=UTC),
                source="SYNTHETIC",
            ),
        ),
    )
    svc.register_market_snapshot(market)
    report = svc.get_valuation()

    assert report.pnl.fx_translation_pnl_base == Decimal("2000.00")   # 100k * (1.12 - 1.10)
    assert report.pnl.unrealized_price_pnl_base == Decimal("2688.00") # 2400 EUR * 1.12
    assert report.pnl.fees_and_commissions_base == Decimal("26.80")   # 10 USD + 15 EUR * 1.12
    assert report.pnl.explained_total_pnl_base == Decimal("4661.20")
    assert report.pnl.unexplained_residual_base == Decimal("0")


def test_scenario_5_late_fee(master: MasterRegistry) -> None:
    """Scenario 5: Late fee posted after initial EOD snapshot appends immutably and bridges P&L exactly."""
    svc = _fresh_service(master)
    market = MarketSnapshot(
        snapshot_id="SNAP-LATE-1",
        as_of_time=datetime(2026, 1, 15, 21, 0, 0, tzinfo=UTC),
        availability_time=datetime(2026, 1, 15, 21, 1, 0, tzinfo=UTC),
    )
    svc.register_market_snapshot(market)
    before = svc.get_valuation()
    events_before = svc.store.list_events()

    # Late fee arrives at 23:15 UTC with earlier event_time 19:00 UTC
    svc.ingest_event(
        FeeEvent(
            event_id="EVT-LATE-FEE-1",
            idempotency_key="LATE-FEE-1",
            event_time=datetime(2026, 1, 15, 19, 0, 0, tzinfo=UTC),
            recorded_time=datetime(2026, 1, 15, 23, 15, 0, tzinfo=UTC),
            account_id="ACCT-STRAT-MOM",
            instrument_id="EQ-MSFT-US",
            currency="USD",
            amount=Decimal("75.25"),
            fee_category="SHORT_BORROW_FEE",
            is_late_posting=True,
            description="Late borrow fee from prime broker",
        )
    )
    after = svc.get_valuation()
    assert svc.store.list_events()[: len(events_before)] == events_before

    bridge = compute_pnl_bridge(before, after)
    assert bridge.delta_nav_base == Decimal("-75.25")
    assert bridge.delta_late_fees_base == Decimal("75.25")
    assert bridge.unexplained_residual_base == Decimal("0")


def test_scenario_6_split_and_dividend(master: MasterRegistry) -> None:
    """Scenario 6: 2-for-1 split preserves exact basis with zero phantom P&L; dividend credits net cash & tax."""
    svc = _fresh_service(master)
    t1 = datetime(2026, 1, 15, 14, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 1, 15, 16, 0, 0, tzinfo=UTC)
    t3 = datetime(2026, 1, 15, 17, 0, 0, tzinfo=UTC)
    svc.ingest_event(
        TradeFillEvent(
            event_id="EVT-CA-BUY",
            idempotency_key="CA-BUY",
            event_time=t1,
            recorded_time=t1,
            account_id="ACCT-STRAT-MOM",
            instrument_id="EQ-AAPL-US",
            order_id="O-CA",
            execution_id="X-CA",
            side=Side.BUY,
            quantity=Decimal("100"),
            price=Decimal("200.00"),
            currency="USD",
            fee=Decimal("0"),
        )
    )
    svc.ingest_event(
        CorporateActionEvent(
            event_id="EVT-CA-SPLIT",
            idempotency_key="CA-SPLIT",
            event_time=t2,
            recorded_time=t2,
            account_id="ACCT-STRAT-MOM",
            instrument_id="EQ-AAPL-US",
            action_type=CorporateActionType.SPLIT,
            ratio_numerator=Decimal("2"),
            ratio_denominator=Decimal("1"),
            currency="USD",
        )
    )
    svc.ingest_event(
        CorporateActionEvent(
            event_id="EVT-CA-DIV",
            idempotency_key="CA-DIV",
            event_time=t3,
            recorded_time=t3,
            account_id="ACCT-STRAT-MOM",
            instrument_id="EQ-AAPL-US",
            action_type=CorporateActionType.CASH_DIVIDEND,
            cash_per_share=Decimal("0.50"),
            withholding_tax_rate=Decimal("0.15"),
            currency="USD",
        )
    )
    market = MarketSnapshot(
        snapshot_id="SNAP-CA",
        as_of_time=datetime(2026, 1, 15, 21, 0, 0, tzinfo=UTC),
        availability_time=datetime(2026, 1, 15, 21, 1, 0, tzinfo=UTC),
        prices=(
            PriceMark(
                instrument_id="EQ-AAPL-US",
                price=Decimal("100.00"),  # exact half of 200 -> 0 price P&L!
                currency="USD",
                price_timestamp=datetime(2026, 1, 15, 21, 0, 0, tzinfo=UTC),
                availability_timestamp=datetime(2026, 1, 15, 21, 1, 0, tzinfo=UTC),
                source="SYNTHETIC",
            ),
        ),
    )
    svc.register_market_snapshot(market)
    report = svc.get_valuation()

    mom = next(v for v in report.strategy_valuations if v.account_id == "ACCT-STRAT-MOM")
    assert mom.positions[0].quantity == Decimal("200")
    assert mom.positions[0].average_unit_cost == Decimal("100.00")
    assert mom.pnl.unrealized_price_pnl_base == Decimal("0")
    assert mom.pnl.dividend_income_base == Decimal("100.00")  # 200 * 0.50 gross
    assert mom.pnl.fees_and_commissions_base == Decimal("15.00")  # 15% withholding tax
    assert mom.cash_balances_local["USD"] == Decimal("280085.00")  # 300k - 20k + 85 net div
    assert report.pnl.unexplained_residual_base == Decimal("0")


def test_scenario_7_missing_mark(master: MasterRegistry) -> None:
    """Scenario 7: Asset lacking a market price is explicitly flagged as MISSING_MARK, never silently zeroed."""
    svc = _fresh_service(master)
    t1 = datetime(2026, 1, 15, 14, 0, 0, tzinfo=UTC)
    svc.ingest_event(
        TradeFillEvent(
            event_id="EVT-UNP-1",
            idempotency_key="UNP-1",
            event_time=t1,
            recorded_time=t1,
            account_id="ACCT-STRAT-MOM",
            instrument_id="EQ-UNPRICED-US",
            order_id="O-UNP",
            execution_id="X-UNP",
            side=Side.BUY,
            quantity=Decimal("100"),
            price=Decimal("50.00"),
            currency="USD",
            fee=Decimal("0"),
        )
    )
    market = MarketSnapshot(
        snapshot_id="SNAP-EMPTY",
        as_of_time=datetime(2026, 1, 15, 21, 0, 0, tzinfo=UTC),
        availability_time=datetime(2026, 1, 15, 21, 1, 0, tzinfo=UTC),
        prices=(),
    )
    svc.register_market_snapshot(market)
    report = svc.get_valuation()

    assert report.valuation_complete is False
    assert report.unpriced_instruments == ("EQ-UNPRICED-US",)
    assert report.unpriced_cost_basis_base == Decimal("5000.00")
    assert report.fund_nav_base == Decimal("500000.00")  # Not falsely dropped to $495,000!
    mom = next(v for v in report.strategy_valuations if v.account_id == "ACCT-STRAT-MOM")
    assert mom.positions[0].pricing_status == PricingStatus.MISSING_MARK
    assert mom.positions[0].is_priced is False


def test_scenario_8_reconciliation_break(master: MasterRegistry) -> None:
    """Scenario 8: Reconciliation flags position and cash breaks without modifying history, then resolves via approved adjustment."""
    svc = _fresh_service(master)
    t1 = datetime(2026, 1, 15, 14, 0, 0, tzinfo=UTC)
    svc.ingest_event(
        TradeFillEvent(
            event_id="EVT-REC-1",
            idempotency_key="REC-1",
            event_time=t1,
            recorded_time=t1,
            account_id="ACCT-STRAT-MOM",
            instrument_id="EQ-AAPL-US",
            order_id="O-REC",
            execution_id="X-REC",
            side=Side.BUY,
            quantity=Decimal("100"),
            price=Decimal("100.00"),
            currency="USD",
            fee=Decimal("0"),
        )
    )
    events_before = svc.store.list_events()
    stmt = BrokerStatement(
        statement_id="STMT-BRK-1",
        broker_name="PRIME_TEST",
        as_of_time=datetime(2026, 1, 15, 22, 0, 0, tzinfo=UTC),
        positions=(
            BrokerPositionRecord(
                account_id="ACCT-STRAT-MOM",
                instrument_id="EQ-AAPL-US",
                quantity=Decimal("95"),  # -5 share break
                currency="USD",
            ),
        ),
        cash_balances=(
            BrokerCashRecord(
                account_id="ACCT-FUND-UNALLOC",
                currency="USD",
                balance=Decimal("200000.00"),
            ),
            BrokerCashRecord(
                account_id="ACCT-STRAT-MOM",
                currency="USD",
                balance=Decimal("289950.00"),  # -$50 cash break vs 290,000
            ),
        ),
    )

    run1 = svc.run_reconciliation(stmt, role=Role.OPERATOR)
    assert run1.mutated_history is False
    assert svc.store.list_events() == events_before
    assert len(run1.breaks) == 2
    assert {b.break_type for b in run1.breaks} == {
        BreakType.POSITION_QUANTITY_BREAK,
        BreakType.CASH_BALANCE_BREAK,
    }

    for brk in run1.breaks:
        res, _ = svc.resolve_break(
            break_id=brk.break_id,
            actor="head_of_ops",
            role=Role.APPROVER,
            reason="Confirmed broker break",
            unit_cost=Decimal("100.00") if brk.instrument_id else Decimal("0"),
        )
        assert res.status == BreakStatus.RESOLVED_BY_ADJUSTMENT

    # History intact + 2 new audited adjustment events
    assert svc.store.list_events()[: len(events_before)] == events_before
    assert len(svc.store.list_events()) == len(events_before) + 2

    run2 = svc.run_reconciliation(stmt, role=Role.OPERATOR)
    assert run2.is_clean is True
