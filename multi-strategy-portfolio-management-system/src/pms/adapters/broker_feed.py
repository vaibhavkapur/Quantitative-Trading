"""External broker statement ingestion contract."""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from pydantic import Field, field_validator

from pms.domain.models import FrozenModel, ensure_utc, normalize_currency


class BrokerPositionRecord(FrozenModel):
    account_id: str = Field(min_length=1)
    instrument_id: str = Field(min_length=1)
    quantity: Decimal
    currency: str
    broker_mark_price: Decimal | None = None

    @field_validator("currency")
    @classmethod
    def _val_ccy(cls, v: str) -> str:
        return normalize_currency(v)


class BrokerCashRecord(FrozenModel):
    account_id: str = Field(min_length=1)
    currency: str
    balance: Decimal

    @field_validator("currency")
    @classmethod
    def _val_ccy(cls, v: str) -> str:
        return normalize_currency(v)


class BrokerStatement(FrozenModel):
    statement_id: str = Field(min_length=1)
    broker_name: str = Field(min_length=1)
    as_of_time: datetime
    positions: tuple[BrokerPositionRecord, ...] = Field(default_factory=tuple)
    cash_balances: tuple[BrokerCashRecord, ...] = Field(default_factory=tuple)

    @field_validator("as_of_time")
    @classmethod
    def _val_utc(cls, v: datetime) -> datetime:
        return ensure_utc(v)

    @classmethod
    def from_file(cls, path: Path | str) -> BrokerStatement:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(raw)
