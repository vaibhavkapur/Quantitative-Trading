"""Enumerations for the Multi-Strategy Portfolio Management System."""

from __future__ import annotations

from enum import StrEnum


class AssetClass(StrEnum):
    EQUITY = "EQUITY"
    ETF = "ETF"
    FUTURE = "FUTURE"
    FX = "FX"
    BOND = "BOND"
    CASH = "CASH"


class InstrumentStatus(StrEnum):
    ACTIVE = "ACTIVE"
    HALTED = "HALTED"
    DELISTED = "DELISTED"


class AccountType(StrEnum):
    FUND_UNALLOCATED = "FUND_UNALLOCATED"
    STRATEGY = "STRATEGY"


class LotMethod(StrEnum):
    FIFO = "FIFO"
    LIFO = "LIFO"
    AVG_COST = "AVG_COST"


class EventType(StrEnum):
    TRADE_FILL = "TRADE_FILL"
    CASH_TRANSFER = "CASH_TRANSFER"
    STRATEGY_ALLOCATION = "STRATEGY_ALLOCATION"
    FX_CONVERSION = "FX_CONVERSION"
    FEE = "FEE"
    CORPORATE_ACTION = "CORPORATE_ACTION"
    ADJUSTMENT = "ADJUSTMENT"


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class CashDirection(StrEnum):
    DEPOSIT = "DEPOSIT"
    WITHDRAWAL = "WITHDRAWAL"


class CorporateActionType(StrEnum):
    SPLIT = "SPLIT"
    CASH_DIVIDEND = "CASH_DIVIDEND"


class AdjustmentType(StrEnum):
    CASH = "CASH"
    POSITION = "POSITION"


class PricingStatus(StrEnum):
    PRICED = "PRICED"
    MISSING_MARK = "MISSING_MARK"
    STALE_MARK = "STALE_MARK"
    MISSING_FX = "MISSING_FX"


class BreakType(StrEnum):
    POSITION_QUANTITY_BREAK = "POSITION_QUANTITY_BREAK"
    CASH_BALANCE_BREAK = "CASH_BALANCE_BREAK"
    MISSING_INTERNAL_POSITION = "MISSING_INTERNAL_POSITION"
    MISSING_BROKER_POSITION = "MISSING_BROKER_POSITION"
    PRICE_DISCREPANCY = "PRICE_DISCREPANCY"


class BreakSeverity(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class BreakStatus(StrEnum):
    OPEN = "OPEN"
    INVESTIGATING = "INVESTIGATING"
    RESOLVED_BY_ADJUSTMENT = "RESOLVED_BY_ADJUSTMENT"
    WAIVED = "WAIVED"


class Role(StrEnum):
    VIEWER = "VIEWER"
    OPERATOR = "OPERATOR"
    APPROVER = "APPROVER"
