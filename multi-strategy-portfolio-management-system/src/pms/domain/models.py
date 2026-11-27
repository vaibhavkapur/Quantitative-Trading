"""Immutable domain contracts for master data and transaction events."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from pms.domain.enums import (
    AccountType,
    AdjustmentType,
    AssetClass,
    CashDirection,
    CorporateActionType,
    EventType,
    InstrumentStatus,
    LotMethod,
    Side,
)

ZERO = Decimal("0")


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("Timestamp must include explicit timezone (UTC required)")
    return value.astimezone(UTC)


def normalize_currency(value: str) -> str:
    code = value.strip().upper()
    if len(code) != 3 or not code.isalpha():
        raise ValueError(f"Invalid ISO-4217 currency code: {value!r}")
    return code


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Fund(FrozenModel):
    fund_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    base_currency: str = "USD"

    @field_validator("base_currency")
    @classmethod
    def _val_ccy(cls, v: str) -> str:
        return normalize_currency(v)


class Account(FrozenModel):
    account_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    account_type: AccountType
    fund_id: str = Field(min_length=1)
    strategy_id: str | None = None
    base_currency: str = "USD"
    lot_method: LotMethod = LotMethod.FIFO
    broker_account_ref: str | None = None

    @field_validator("base_currency")
    @classmethod
    def _val_ccy(cls, v: str) -> str:
        return normalize_currency(v)

    @model_validator(mode="after")
    def _validate_strategy_binding(self) -> Account:
        if self.account_type == AccountType.STRATEGY and not self.strategy_id:
            raise ValueError("STRATEGY account must specify strategy_id")
        if self.account_type == AccountType.FUND_UNALLOCATED and self.strategy_id is not None:
            raise ValueError("FUND_UNALLOCATED account cannot have strategy_id")
        return self


class Instrument(FrozenModel):
    instrument_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    name: str = Field(min_length=1)
    asset_class: AssetClass
    currency: str
    exchange: str = Field(min_length=1)
    multiplier: Decimal = Decimal("1")
    tick_size: Decimal = Decimal("0.01")
    status: InstrumentStatus = InstrumentStatus.ACTIVE

    @field_validator("currency")
    @classmethod
    def _val_ccy(cls, v: str) -> str:
        return normalize_currency(v)

    @field_validator("multiplier", "tick_size")
    @classmethod
    def _val_positive(cls, v: Decimal) -> Decimal:
        if v <= ZERO:
            raise ValueError("multiplier and tick_size must be strictly positive")
        return v


class BaseLedgerEvent(FrozenModel):
    event_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    event_time: datetime
    recorded_time: datetime

    @field_validator("event_time", "recorded_time")
    @classmethod
    def _val_utc(cls, v: datetime) -> datetime:
        return ensure_utc(v)


class TradeFillEvent(BaseLedgerEvent):
    event_type: Literal[EventType.TRADE_FILL] = EventType.TRADE_FILL
    account_id: str = Field(min_length=1)
    instrument_id: str = Field(min_length=1)
    order_id: str = Field(min_length=1)
    execution_id: str = Field(min_length=1)
    side: Side
    quantity: Decimal
    price: Decimal
    currency: str
    fee: Decimal = ZERO
    venue: str = "SYNTHETIC"

    @field_validator("currency")
    @classmethod
    def _val_ccy(cls, v: str) -> str:
        return normalize_currency(v)

    @field_validator("quantity", "price")
    @classmethod
    def _val_pos(cls, v: Decimal) -> Decimal:
        if v <= ZERO:
            raise ValueError("quantity and price must be strictly positive")
        return v

    @field_validator("fee")
    @classmethod
    def _val_fee(cls, v: Decimal) -> Decimal:
        if v < ZERO:
            raise ValueError("fee cannot be negative")
        return v


class CashTransferEvent(BaseLedgerEvent):
    event_type: Literal[EventType.CASH_TRANSFER] = EventType.CASH_TRANSFER
    account_id: str = Field(min_length=1)
    currency: str
    amount: Decimal
    direction: CashDirection
    counterparty_account_id: str | None = None
    description: str = ""

    @field_validator("currency")
    @classmethod
    def _val_ccy(cls, v: str) -> str:
        return normalize_currency(v)

    @field_validator("amount")
    @classmethod
    def _val_amount(cls, v: Decimal) -> Decimal:
        if v <= ZERO:
            raise ValueError("cash transfer amount must be strictly positive")
        return v


class StrategyAllocationEvent(BaseLedgerEvent):
    event_type: Literal[EventType.STRATEGY_ALLOCATION] = EventType.STRATEGY_ALLOCATION
    from_account_id: str = Field(min_length=1)
    to_account_id: str = Field(min_length=1)
    currency: str
    amount: Decimal
    allocation_reason: str = ""

    @field_validator("currency")
    @classmethod
    def _val_ccy(cls, v: str) -> str:
        return normalize_currency(v)

    @field_validator("amount")
    @classmethod
    def _val_amount(cls, v: Decimal) -> Decimal:
        if v <= ZERO:
            raise ValueError("allocation amount must be strictly positive")
        return v

    @model_validator(mode="after")
    def _distinct_accounts(self) -> StrategyAllocationEvent:
        if self.from_account_id == self.to_account_id:
            raise ValueError("from_account_id and to_account_id must be distinct")
        return self


class FxConversionEvent(BaseLedgerEvent):
    event_type: Literal[EventType.FX_CONVERSION] = EventType.FX_CONVERSION
    account_id: str = Field(min_length=1)
    sell_currency: str
    sell_amount: Decimal
    buy_currency: str
    buy_amount: Decimal
    fx_rate: Decimal
    fee_currency: str = "USD"
    fee_amount: Decimal = ZERO

    @field_validator("sell_currency", "buy_currency", "fee_currency")
    @classmethod
    def _val_ccy(cls, v: str) -> str:
        return normalize_currency(v)

    @field_validator("sell_amount", "buy_amount", "fx_rate")
    @classmethod
    def _val_pos(cls, v: Decimal) -> Decimal:
        if v <= ZERO:
            raise ValueError("FX amounts and rate must be strictly positive")
        return v

    @field_validator("fee_amount")
    @classmethod
    def _val_non_neg(cls, v: Decimal) -> Decimal:
        if v < ZERO:
            raise ValueError("FX fee_amount cannot be negative")
        return v

    @model_validator(mode="after")
    def _distinct_currencies(self) -> FxConversionEvent:
        if self.sell_currency == self.buy_currency:
            raise ValueError("sell_currency and buy_currency must differ")
        return self


class FeeEvent(BaseLedgerEvent):
    event_type: Literal[EventType.FEE] = EventType.FEE
    account_id: str = Field(min_length=1)
    instrument_id: str | None = None
    currency: str
    amount: Decimal
    fee_category: str = Field(min_length=1)
    is_late_posting: bool = False
    description: str = ""

    @field_validator("currency")
    @classmethod
    def _val_ccy(cls, v: str) -> str:
        return normalize_currency(v)

    @field_validator("amount")
    @classmethod
    def _val_amount(cls, v: Decimal) -> Decimal:
        if v <= ZERO:
            raise ValueError("fee amount must be strictly positive")
        return v


class CorporateActionEvent(BaseLedgerEvent):
    event_type: Literal[EventType.CORPORATE_ACTION] = EventType.CORPORATE_ACTION
    account_id: str = Field(min_length=1)
    instrument_id: str = Field(min_length=1)
    action_type: CorporateActionType
    ratio_numerator: Decimal = Decimal("1")
    ratio_denominator: Decimal = Decimal("1")
    cash_per_share: Decimal = ZERO
    withholding_tax_rate: Decimal = ZERO
    currency: str
    description: str = ""

    @field_validator("currency")
    @classmethod
    def _val_ccy(cls, v: str) -> str:
        return normalize_currency(v)

    @field_validator("ratio_numerator", "ratio_denominator")
    @classmethod
    def _val_ratio(cls, v: Decimal) -> Decimal:
        if v <= ZERO:
            raise ValueError("split ratio numerator and denominator must be positive")
        return v

    @field_validator("cash_per_share")
    @classmethod
    def _val_cash(cls, v: Decimal) -> Decimal:
        if v < ZERO:
            raise ValueError("cash_per_share cannot be negative")
        return v

    @field_validator("withholding_tax_rate")
    @classmethod
    def _val_tax(cls, v: Decimal) -> Decimal:
        if v < ZERO or v >= Decimal("1"):
            raise ValueError("withholding_tax_rate must be in [0, 1)")
        return v

    @model_validator(mode="after")
    def _validate_action_fields(self) -> CorporateActionEvent:
        if self.action_type == CorporateActionType.CASH_DIVIDEND and self.cash_per_share <= ZERO:
            raise ValueError("CASH_DIVIDEND requires positive cash_per_share")
        if self.action_type == CorporateActionType.SPLIT and (
            self.ratio_numerator == self.ratio_denominator
        ):
            raise ValueError("SPLIT ratio must change share count")
        return self


class AdjustmentEvent(BaseLedgerEvent):
    """Explicit audited adjustment event appended to resolve a reconciliation break."""

    event_type: Literal[EventType.ADJUSTMENT] = EventType.ADJUSTMENT
    account_id: str = Field(min_length=1)
    adjustment_type: AdjustmentType
    instrument_id: str | None = None
    currency: str
    cash_delta: Decimal = ZERO
    quantity_delta: Decimal = ZERO
    unit_cost: Decimal = ZERO
    reconciliation_break_id: str = Field(min_length=1)
    approved_by: str = Field(min_length=1)
    reason: str = Field(min_length=1)

    @field_validator("currency")
    @classmethod
    def _val_ccy(cls, v: str) -> str:
        return normalize_currency(v)

    @model_validator(mode="after")
    def _validate_adjustment(self) -> AdjustmentEvent:
        if self.adjustment_type == AdjustmentType.CASH and self.cash_delta == ZERO:
            raise ValueError("CASH adjustment requires non-zero cash_delta")
        if self.adjustment_type == AdjustmentType.POSITION:
            if not self.instrument_id:
                raise ValueError("POSITION adjustment requires instrument_id")
            if self.quantity_delta == ZERO:
                raise ValueError("POSITION adjustment requires non-zero quantity_delta")
            if self.unit_cost < ZERO:
                raise ValueError("POSITION adjustment unit_cost cannot be negative")
        return self


LedgerEvent = Annotated[
    TradeFillEvent
    | CashTransferEvent
    | StrategyAllocationEvent
    | FxConversionEvent
    | FeeEvent
    | CorporateActionEvent
    | AdjustmentEvent,
    Field(discriminator="event_type"),
]

LedgerEventAdapter: TypeAdapter[LedgerEvent] = TypeAdapter(LedgerEvent)


def parse_ledger_event(payload: dict[str, Any] | LedgerEvent) -> LedgerEvent:
    if isinstance(
        payload,
        (
            TradeFillEvent,
            CashTransferEvent,
            StrategyAllocationEvent,
            FxConversionEvent,
            FeeEvent,
            CorporateActionEvent,
            AdjustmentEvent,
        ),
    ):
        return payload
    return LedgerEventAdapter.validate_python(payload)


class StoredEvent(FrozenModel):
    seq_no: int
    event_id: str
    idempotency_key: str
    event_type: EventType
    event_time: datetime
    recorded_time: datetime
    prev_hash: str
    event_hash: str
    event: LedgerEvent
