"""Versioned Instrument and Account Master registry."""

from __future__ import annotations

import json
from pathlib import Path

from pms.domain.enums import AccountType
from pms.domain.errors import MasterDataError
from pms.domain.models import (
    Account,
    AdjustmentEvent,
    CashTransferEvent,
    CorporateActionEvent,
    FeeEvent,
    Fund,
    FxConversionEvent,
    Instrument,
    LedgerEvent,
    StrategyAllocationEvent,
    TradeFillEvent,
)


class MasterRegistry:
    """Owns canonical fund, account, and instrument master contracts."""

    def __init__(
        self,
        *,
        schema_version: str,
        effective_at: str,
        fund: Fund,
        accounts: list[Account],
        instruments: list[Instrument],
    ) -> None:
        self.schema_version = schema_version
        self.effective_at = effective_at
        self.fund = fund
        self._accounts: dict[str, Account] = {}
        self._instruments: dict[str, Instrument] = {}

        for account in accounts:
            self.register_account(account)
        for instrument in instruments:
            self.register_instrument(instrument)

    @classmethod
    def from_file(cls, path: Path | str) -> MasterRegistry:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        fund = Fund.model_validate(raw["fund"])
        accounts = [Account.model_validate(item) for item in raw["accounts"]]
        instruments = [Instrument.model_validate(item) for item in raw["instruments"]]
        return cls(
            schema_version=raw.get("schema_version", "1.0.0"),
            effective_at=raw.get("effective_at", "2026-01-01T00:00:00Z"),
            fund=fund,
            accounts=accounts,
            instruments=instruments,
        )

    def register_account(self, account: Account) -> None:
        if account.account_id in self._accounts:
            raise MasterDataError(
                f"Duplicate account_id: {account.account_id}",
                details={"account_id": account.account_id},
            )
        if account.fund_id != self.fund.fund_id:
            raise MasterDataError(
                f"Account {account.account_id} references unknown fund {account.fund_id}",
                details={"account_id": account.account_id, "fund_id": account.fund_id},
            )
        self._accounts[account.account_id] = account

    def register_instrument(self, instrument: Instrument) -> None:
        if instrument.instrument_id in self._instruments:
            raise MasterDataError(
                f"Duplicate instrument_id: {instrument.instrument_id}",
                details={"instrument_id": instrument.instrument_id},
            )
        self._instruments[instrument.instrument_id] = instrument

    def get_account(self, account_id: str) -> Account:
        if account_id not in self._accounts:
            raise MasterDataError(
                f"Unknown account_id: {account_id}",
                details={"account_id": account_id},
            )
        return self._accounts[account_id]

    def get_instrument(self, instrument_id: str) -> Instrument:
        if instrument_id not in self._instruments:
            raise MasterDataError(
                f"Unknown instrument_id: {instrument_id}",
                details={"instrument_id": instrument_id},
            )
        return self._instruments[instrument_id]

    def list_accounts(self) -> list[Account]:
        return list(self._accounts.values())

    def list_strategy_accounts(self) -> list[Account]:
        return [a for a in self._accounts.values() if a.account_type == AccountType.STRATEGY]

    def list_unallocated_accounts(self) -> list[Account]:
        return [
            a for a in self._accounts.values() if a.account_type == AccountType.FUND_UNALLOCATED
        ]

    def list_instruments(self) -> list[Instrument]:
        return list(self._instruments.values())

    def validate_event(self, event: LedgerEvent) -> None:
        """Validate account, instrument, and currency bindings against master data."""
        if isinstance(event, TradeFillEvent):
            account = self.get_account(event.account_id)
            if account.account_type == AccountType.FUND_UNALLOCATED:
                raise MasterDataError(
                    "Trade fills must book into a STRATEGY account, not FUND_UNALLOCATED",
                    details={"account_id": event.account_id, "event_id": event.event_id},
                )
            inst = self.get_instrument(event.instrument_id)
            if inst.currency != event.currency:
                raise MasterDataError(
                    f"Trade currency {event.currency} does not match instrument currency {inst.currency}",
                    details={
                        "instrument_id": event.instrument_id,
                        "expected_currency": inst.currency,
                        "actual_currency": event.currency,
                    },
                )
        elif isinstance(event, CashTransferEvent):
            self.get_account(event.account_id)
            if event.counterparty_account_id is not None:
                self.get_account(event.counterparty_account_id)
        elif isinstance(event, StrategyAllocationEvent):
            self.get_account(event.from_account_id)
            self.get_account(event.to_account_id)
        elif isinstance(event, FxConversionEvent):
            self.get_account(event.account_id)
        elif isinstance(event, FeeEvent):
            self.get_account(event.account_id)
            if event.instrument_id is not None:
                self.get_instrument(event.instrument_id)
        elif isinstance(event, CorporateActionEvent):
            self.get_account(event.account_id)
            inst = self.get_instrument(event.instrument_id)
            if inst.currency != event.currency:
                raise MasterDataError(
                    f"Corporate action currency {event.currency} does not match instrument currency {inst.currency}",
                    details={
                        "instrument_id": event.instrument_id,
                        "expected_currency": inst.currency,
                        "actual_currency": event.currency,
                    },
                )
        elif isinstance(event, AdjustmentEvent):
            self.get_account(event.account_id)
            if event.instrument_id is not None:
                inst = self.get_instrument(event.instrument_id)
                if inst.currency != event.currency:
                    raise MasterDataError(
                        f"Adjustment currency {event.currency} does not match instrument currency {inst.currency}",
                        details={"instrument_id": event.instrument_id},
                    )
