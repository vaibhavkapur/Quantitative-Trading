"""Versioned point-in-time market price and FX rate snapshots."""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from pydantic import Field, field_validator

from pms.domain.models import FrozenModel, ensure_utc, normalize_currency

ZERO = Decimal("0")


class PriceMark(FrozenModel):
    instrument_id: str = Field(min_length=1)
    price: Decimal
    currency: str
    price_timestamp: datetime
    availability_timestamp: datetime
    source: str = Field(min_length=1)

    @field_validator("currency")
    @classmethod
    def _val_ccy(cls, v: str) -> str:
        return normalize_currency(v)

    @field_validator("price")
    @classmethod
    def _val_price(cls, v: Decimal) -> Decimal:
        if v <= ZERO:
            raise ValueError("mark price must be strictly positive")
        return v

    @field_validator("price_timestamp", "availability_timestamp")
    @classmethod
    def _val_utc(cls, v: datetime) -> datetime:
        return ensure_utc(v)


class FxRateMark(FrozenModel):
    from_currency: str
    to_currency: str
    rate: Decimal
    rate_timestamp: datetime
    availability_timestamp: datetime
    source: str = Field(min_length=1)

    @field_validator("from_currency", "to_currency")
    @classmethod
    def _val_ccy(cls, v: str) -> str:
        return normalize_currency(v)

    @field_validator("rate")
    @classmethod
    def _val_rate(cls, v: Decimal) -> Decimal:
        if v <= ZERO:
            raise ValueError("FX rate must be strictly positive")
        return v

    @field_validator("rate_timestamp", "availability_timestamp")
    @classmethod
    def _val_utc(cls, v: datetime) -> datetime:
        return ensure_utc(v)


class MarketSnapshot(FrozenModel):
    snapshot_id: str = Field(min_length=1)
    as_of_time: datetime
    availability_time: datetime
    base_currency: str = "USD"
    source: str = "SYNTHETIC"
    prices: tuple[PriceMark, ...] = Field(default_factory=tuple)
    fx_rates: tuple[FxRateMark, ...] = Field(default_factory=tuple)

    @field_validator("base_currency")
    @classmethod
    def _val_ccy(cls, v: str) -> str:
        return normalize_currency(v)

    @field_validator("as_of_time", "availability_time")
    @classmethod
    def _val_utc(cls, v: datetime) -> datetime:
        return ensure_utc(v)

    @classmethod
    def from_file(cls, path: Path | str) -> MarketSnapshot:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(raw)

    def get_price(self, instrument_id: str) -> PriceMark | None:
        for mark in self.prices:
            if mark.instrument_id == instrument_id:
                return mark
        return None

    def get_fx_rate(self, from_currency: str, to_currency: str | None = None) -> FxRateMark | None:
        target = to_currency or self.base_currency
        if from_currency == target:
            return FxRateMark(
                from_currency=from_currency,
                to_currency=target,
                rate=Decimal("1"),
                rate_timestamp=self.as_of_time,
                availability_timestamp=self.availability_time,
                source="IDENTITY",
            )
        for mark in self.fx_rates:
            if mark.from_currency == from_currency and mark.to_currency == target:
                return mark
        return None
