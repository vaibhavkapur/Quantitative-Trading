"""Canonical domain types and versioned schemas for alpha research."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from alpha_engine.errors import LookAheadViolationError


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("Timestamp must be timezone-aware UTC.")
    return value.astimezone(UTC)


class MarketBar(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str = Field(min_length=1)
    event_ts_utc: datetime
    available_at_utc: datetime
    open: float = Field(gt=0.0)
    high: float = Field(gt=0.0)
    low: float = Field(gt=0.0)
    close: float = Field(gt=0.0)
    adjusted_close: float = Field(gt=0.0)
    volume: float = Field(ge=0.0)
    currency: str = "USD"
    split_factor: float = Field(gt=0.0, default=1.0)
    dividend_cash: float = Field(ge=0.0, default=0.0)

    @field_validator("event_ts_utc", "available_at_utc")
    @classmethod
    def _validate_utc(cls, value: datetime) -> datetime:
        return _ensure_utc(value)

    @model_validator(mode="after")
    def _validate_causality_and_ohlc(self) -> MarketBar:
        if self.available_at_utc < self.event_ts_utc:
            raise LookAheadViolationError(
                f"MarketBar for {self.symbol} has available_at_utc before event_ts_utc.",
                details={
                    "symbol": self.symbol,
                    "event_ts_utc": self.event_ts_utc.isoformat(),
                    "available_at_utc": self.available_at_utc.isoformat(),
                },
            )
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) cannot be less than low ({self.low}).")
        return self


class FundamentalRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str = Field(min_length=1)
    period_end_utc: datetime
    event_ts_utc: datetime
    available_at_utc: datetime
    earnings_ttm_usd: float
    book_value_usd: float = Field(gt=0.0)
    market_cap_usd: float = Field(gt=0.0)
    roe: float
    currency: str = "USD"

    @field_validator("period_end_utc", "event_ts_utc", "available_at_utc")
    @classmethod
    def _validate_utc(cls, value: datetime) -> datetime:
        return _ensure_utc(value)

    @model_validator(mode="after")
    def _validate_filing_causality(self) -> FundamentalRecord:
        if self.available_at_utc < self.period_end_utc or self.available_at_utc < self.event_ts_utc:
            raise LookAheadViolationError(
                f"FundamentalRecord for {self.symbol} violates filing availability causality.",
                details={
                    "symbol": self.symbol,
                    "period_end_utc": self.period_end_utc.isoformat(),
                    "available_at_utc": self.available_at_utc.isoformat(),
                },
            )
        return self


class CorporateAction(BaseModel):
    model_config = ConfigDict(frozen=True)

    action_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    action_type: Literal["split", "reverse_split", "dividend", "delisting"]
    ex_date_utc: datetime
    available_at_utc: datetime
    ratio: float = Field(gt=0.0, default=1.0)
    cash_amount_usd: float = Field(ge=0.0, default=0.0)
    currency: str = "USD"

    @field_validator("ex_date_utc", "available_at_utc")
    @classmethod
    def _validate_utc(cls, value: datetime) -> datetime:
        return _ensure_utc(value)


class UniverseMembership(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str = Field(min_length=1)
    sector: str = Field(min_length=1)
    entry_ts_utc: datetime
    exit_ts_utc: datetime | None = None
    exit_reason: str | None = None

    @field_validator("entry_ts_utc")
    @classmethod
    def _validate_entry(cls, value: datetime) -> datetime:
        return _ensure_utc(value)

    @field_validator("exit_ts_utc")
    @classmethod
    def _validate_exit(cls, value: datetime | None) -> datetime | None:
        return _ensure_utc(value) if value is not None else None


class DatasetManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    manifest_version: str = "1.0.0"
    dataset_id: str
    generated_at_utc: datetime
    source_type: Literal["synthetic", "permitted_historical"] = "synthetic"
    entitlements: str = "internal_synthetic_research_only"
    license: str = "proprietary_synthetic_fixture"
    seed: int
    n_symbols: int
    n_delisted_symbols: int
    n_trading_days: int
    start_date_utc: datetime
    end_date_utc: datetime
    file_checksums: dict[str, str]
    manifest_digest: str = ""

    @field_validator("generated_at_utc", "start_date_utc", "end_date_utc")
    @classmethod
    def _validate_utc(cls, value: datetime) -> datetime:
        return _ensure_utc(value)

    def compute_digest(self) -> str:
        payload = self.model_dump(mode="json", exclude={"manifest_digest"})
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


class SignalDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    signal_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    family: Literal[
        "momentum",
        "reversal",
        "volatility",
        "fundamental",
        "statistical_ml",
        "adapter_legacy",
    ]
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    lookback_bars: int = Field(ge=1)
    horizon_bars: int = Field(ge=1, default=5)
    inputs_required: list[str] = Field(default_factory=lambda: ["market_bars"])
    causal_guarantee: str = "strict_point_in_time_available_at_utc"

    @property
    def definition_hash(self) -> str:
        payload = self.model_dump(mode="json")
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]


class SignalRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    signal_id: str
    signal_version: str
    symbol: str
    event_ts_utc: datetime
    available_at_utc: datetime
    raw_value: float
    neutralized_value: float | None = None

    @field_validator("event_ts_utc", "available_at_utc")
    @classmethod
    def _validate_utc(cls, value: datetime) -> datetime:
        return _ensure_utc(value)

    @model_validator(mode="after")
    def _validate_causal_availability(self) -> SignalRecord:
        if self.available_at_utc < self.event_ts_utc:
            raise LookAheadViolationError(
                f"Signal {self.signal_id} for {self.symbol} has available_at_utc < event_ts_utc.",
                details={
                    "signal_id": self.signal_id,
                    "symbol": self.symbol,
                    "event_ts_utc": self.event_ts_utc.isoformat(),
                    "available_at_utc": self.available_at_utc.isoformat(),
                },
            )
        return self


class CostSensitivityPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    cost_bps: float = Field(ge=0.0)
    gross_annualized_return: float
    net_annualized_return: float
    gross_sharpe: float
    net_sharpe: float
    avg_daily_turnover: float = Field(ge=0.0)
    total_cost_drag_bps: float = Field(ge=0.0)
    max_drawdown: float


class FoldMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    fold_id: int
    train_start_utc: datetime
    train_end_utc: datetime
    test_start_utc: datetime
    test_end_utc: datetime
    n_test_days: int
    oos_spearman_ic: float
    oos_pearson_ic: float
    oos_icir: float
    hit_rate: float
    avg_daily_turnover: float
    max_abs_sector_exposure: float
    coefficients: dict[str, float] = Field(default_factory=dict)


class StabilityReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    coefficient_sign_flip_rate: float = Field(ge=0.0, le=1.0)
    coefficient_cv: float = Field(ge=0.0)
    ic_sign_consistency: float = Field(ge=0.0, le=1.0)
    max_fold_ic_drop: float
    is_stable: bool
    reasons: list[str] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    trial_id: str
    signal_id: str
    signal_version: str
    definition_hash: str
    evaluated_at_utc: datetime
    dataset_manifest_digest: str
    sample_days: int
    universe_size: int
    n_folds: int
    oos_mean_ic: float
    oos_std_ic: float
    oos_icir: float
    oos_ic_tstat: float
    oos_ic_pvalue: float
    bh_adjusted_pvalue: float
    bootstrap_ic_ci_low: float
    bootstrap_ic_ci_high: float
    avg_daily_turnover: float
    max_abs_sector_exposure: float
    max_peer_correlation: float
    most_correlated_peer: str | None = None
    stability: StabilityReport
    folds: list[FoldMetrics]
    cost_sensitivity: list[CostSensitivityPoint]
    base_cost_bps: float
    net_sharpe_at_base_cost: float
    passed_gates: bool
    gate_failures: list[str] = Field(default_factory=list)
    negative_result_notes: str = ""


class TrialRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    trial_id: str
    signal_id: str
    signal_version: str
    definition_hash: str
    recorded_at_utc: datetime
    dataset_manifest_digest: str
    oos_mean_ic: float
    oos_icir: float
    oos_ic_pvalue: float
    bh_adjusted_pvalue: float
    avg_daily_turnover: float
    max_abs_sector_exposure: float
    max_peer_correlation: float
    net_sharpe_at_base_cost: float
    passed_gates: bool
    gate_failures: list[str] = Field(default_factory=list)
    status: Literal["candidate", "promoted", "rejected", "failed"] = "candidate"
    notes: str = ""


class CatalogEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    signal_definition: SignalDefinition
    latest_trial_id: str
    status: Literal["candidate", "promoted", "rejected"]
    promoted_at_utc: datetime | None = None
    promoted_by: str | None = None
    evaluation_summary: dict[str, Any] = Field(default_factory=dict)


class PaperSignalExport(BaseModel):
    model_config = ConfigDict(frozen=True)

    export_id: str
    exported_at_utc: datetime
    mode: Literal["paper_only"] = "paper_only"
    execution_enabled: Literal[False] = False
    signal_id: str
    signal_version: str
    definition_hash: str
    as_of_utc: datetime
    records_count: int
    gross_exposure: float
    net_exposure: float
    provenance: dict[str, Any]
    weights: list[dict[str, Any]]
    sha256_digest: str
