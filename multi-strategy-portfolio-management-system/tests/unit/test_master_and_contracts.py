"""Unit tests for instrument/account master registry and strict domain contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError as PydanticValidationError

from pms.domain.enums import AccountType, Side
from pms.domain.errors import MasterDataError
from pms.domain.master import MasterRegistry
from pms.domain.models import Account, TradeFillEvent


def test_master_loads_canonical_accounts_and_instruments(master: MasterRegistry) -> None:
    assert master.fund.fund_id == "FUND-SAAT-MAIN"
    assert master.fund.base_currency == "USD"
    assert len(master.list_strategy_accounts()) == 3
    assert len(master.list_unallocated_accounts()) == 1
    assert master.get_instrument("EQ-AAPL-US").currency == "USD"
    assert master.get_instrument("FUT-ES-202606").multiplier == Decimal("50")


def test_account_strategy_binding_rules() -> None:
    with pytest.raises(PydanticValidationError):
        Account(
            account_id="BAD-1",
            name="Missing strategy id",
            account_type=AccountType.STRATEGY,
            fund_id="FUND-SAAT-MAIN",
            strategy_id=None,
        )
    with pytest.raises(PydanticValidationError):
        Account(
            account_id="BAD-2",
            name="Unallocated with strategy id",
            account_type=AccountType.FUND_UNALLOCATED,
            fund_id="FUND-SAAT-MAIN",
            strategy_id="STRAT-X",
        )


def test_naive_timestamp_and_bad_currency_rejected() -> None:
    with pytest.raises(PydanticValidationError):
        TradeFillEvent(
            event_id="E1",
            idempotency_key="K1",
            event_time=datetime(2026, 1, 15, 14, 0, 0),  # naive timestamp
            recorded_time=datetime(2026, 1, 15, 14, 0, 1, tzinfo=UTC),
            account_id="ACCT-STRAT-MOM",
            instrument_id="EQ-AAPL-US",
            order_id="O1",
            execution_id="X1",
            side=Side.BUY,
            quantity=Decimal("10"),
            price=Decimal("100"),
            currency="USD",
        )

    with pytest.raises(PydanticValidationError):
        TradeFillEvent(
            event_id="E2",
            idempotency_key="K2",
            event_time=datetime(2026, 1, 15, 14, 0, 0, tzinfo=UTC),
            recorded_time=datetime(2026, 1, 15, 14, 0, 1, tzinfo=UTC),
            account_id="ACCT-STRAT-MOM",
            instrument_id="EQ-AAPL-US",
            order_id="O1",
            execution_id="X1",
            side=Side.BUY,
            quantity=Decimal("10"),
            price=Decimal("100"),
            currency="US",  # invalid ISO-4217
        )


def test_master_rejects_trade_into_unallocated_and_currency_mismatch(
    master: MasterRegistry,
) -> None:
    now = datetime(2026, 1, 15, 14, 0, 0, tzinfo=UTC)
    unalloc_trade = TradeFillEvent(
        event_id="E-UNALLOC",
        idempotency_key="K-UNALLOC",
        event_time=now,
        recorded_time=now,
        account_id="ACCT-FUND-UNALLOC",
        instrument_id="EQ-AAPL-US",
        order_id="O1",
        execution_id="X1",
        side=Side.BUY,
        quantity=Decimal("10"),
        price=Decimal("100"),
        currency="USD",
    )
    with pytest.raises(MasterDataError):
        master.validate_event(unalloc_trade)

    wrong_ccy_trade = TradeFillEvent(
        event_id="E-CCY",
        idempotency_key="K-CCY",
        event_time=now,
        recorded_time=now,
        account_id="ACCT-STRAT-STATARB",
        instrument_id="EQ-SAP-DE",  # EUR instrument
        order_id="O2",
        execution_id="X2",
        side=Side.BUY,
        quantity=Decimal("10"),
        price=Decimal("200"),
        currency="USD",  # mismatch
    )
    with pytest.raises(MasterDataError):
        master.validate_event(wrong_ccy_trade)
