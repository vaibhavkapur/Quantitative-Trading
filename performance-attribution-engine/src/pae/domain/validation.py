"""Canonical domain validation with point-in-time, currency, and ID invariant checks."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import List, Set, Tuple

from pae.domain.models import AttributionInputBatch, DataQualityException
from pae.errors import ValidationError


def parse_utc(ts: str) -> datetime:
    """Parse an ISO-8601 UTC timestamp string (ending with Z or +00:00)."""
    if not isinstance(ts, str) or not ts:
        raise ValidationError(f"Timestamp must be a non-empty ISO-8601 UTC string, got {ts!r}")
    normalized = ts[:-1] + "+00:00" if ts.endswith("Z") else ts
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValidationError(f"Invalid ISO-8601 timestamp: {ts!r}") from exc
    if dt.tzinfo is None:
        raise ValidationError(f"Timestamp must include UTC timezone offset or 'Z': {ts!r}")
    return dt.astimezone(timezone.utc)


def _check_currency(ccy: str, field_name: str) -> None:
    if not isinstance(ccy, str) or len(ccy) != 3 or not ccy.isalpha() or ccy != ccy.upper():
        raise ValidationError(
            f"Invalid ISO-4217 uppercase 3-letter currency for {field_name}: {ccy!r}"
        )


def _check_finite(val: float, field_name: str) -> None:
    if not isinstance(val, (int, float)) or not math.isfinite(val):
        raise ValidationError(f"Non-finite numeric value for {field_name}: {val!r}")


def _check_date(d: str, field_name: str) -> None:
    try:
        datetime.strptime(d, "%Y-%m-%d")
    except (ValueError, TypeError) as exc:
        raise ValidationError(f"Invalid YYYY-MM-DD date for {field_name}: {d!r}") from exc


def validate_input_batch(batch: AttributionInputBatch) -> List[DataQualityException]:
    """Validate structural invariants of an AttributionInputBatch.

    Raises ValidationError on hard schema/contract violations.
    Returns non-fatal DataQualityException warnings (e.g., look-ahead records, fee corrections).
    """
    exceptions: List[DataQualityException] = []

    if not batch.batch_id or not batch.portfolio_id:
        raise ValidationError("batch_id and portfolio_id are required.")
    _check_currency(batch.reporting_currency, "batch.reporting_currency")
    run_as_of = parse_utc(batch.as_of_utc)

    if not batch.nav_snapshots:
        raise ValidationError("At least one DailyNavSnapshot is required in nav_snapshots.")

    # Validate NAV snapshots
    nav_dates: Set[str] = set()
    nav_ids: Set[str] = set()
    for nav in batch.nav_snapshots:
        _check_date(nav.trade_date, "nav.trade_date")
        _check_currency(nav.reporting_currency, "nav.reporting_currency")
        if nav.nav_id in nav_ids:
            raise ValidationError(f"Duplicate nav_id: {nav.nav_id}")
        nav_ids.add(nav.nav_id)
        if nav.trade_date in nav_dates:
            raise ValidationError(f"Duplicate DailyNavSnapshot for trade_date {nav.trade_date}")
        nav_dates.add(nav.trade_date)
        if nav.portfolio_id != batch.portfolio_id:
            raise ValidationError(
                f"NAV snapshot {nav.nav_id} portfolio_id {nav.portfolio_id} != batch {batch.portfolio_id}"
            )
        if nav.reporting_currency != batch.reporting_currency:
            raise ValidationError(
                f"NAV snapshot {nav.nav_id} currency {nav.reporting_currency} != batch {batch.reporting_currency}"
            )
        for fname in ("start_nav_base", "end_nav_base", "start_cash_base", "end_cash_base", "cash_interest_base"):
            _check_finite(getattr(nav, fname), f"nav.{fname}")
        avail = parse_utc(nav.availability_utc)
        parse_utc(nav.as_of_utc)
        if avail > run_as_of:
            exceptions.append(
                DataQualityException(
                    code="LOOKAHEAD_VIOLATION",
                    severity="WARNING",
                    trade_date=nav.trade_date,
                    entity_id=nav.nav_id,
                    message=(
                        f"NAV snapshot {nav.nav_id} availability_utc ({nav.availability_utc}) "
                        f"is after batch as_of_utc ({batch.as_of_utc})."
                    ),
                )
            )

    # Validate positions
    pos_ids: Set[str] = set()
    pos_keys: Set[Tuple[str, str, str]] = set()
    for pos in batch.positions:
        _check_date(pos.trade_date, "position.trade_date")
        _check_currency(pos.local_currency, "position.local_currency")
        if pos.position_id in pos_ids:
            raise ValidationError(f"Duplicate position_id: {pos.position_id}")
        pos_ids.add(pos.position_id)
        key = (pos.trade_date, pos.strategy_id, pos.instrument_id)
        if key in pos_keys:
            raise ValidationError(
                f"Duplicate position for (trade_date, strategy_id, instrument_id)={key}"
            )
        pos_keys.add(key)
        if pos.portfolio_id != batch.portfolio_id:
            raise ValidationError(
                f"Position {pos.position_id} portfolio_id {pos.portfolio_id} != batch {batch.portfolio_id}"
            )
        for fname in (
            "start_quantity",
            "end_quantity",
            "start_price_local",
            "end_price_local",
            "income_local",
            "explicit_fees_local",
        ):
            _check_finite(getattr(pos, fname), f"position.{fname}")
        if pos.start_price_local < 0 or pos.end_price_local < 0:
            raise ValidationError(f"Position {pos.position_id} has negative price.")
        if pos.start_market_value_base is not None:
            _check_finite(pos.start_market_value_base, "position.start_market_value_base")
        if pos.end_market_value_base is not None:
            _check_finite(pos.end_market_value_base, "position.end_market_value_base")
        parse_utc(pos.as_of_utc)
        avail = parse_utc(pos.availability_utc)
        if avail > run_as_of:
            exceptions.append(
                DataQualityException(
                    code="LOOKAHEAD_VIOLATION",
                    severity="WARNING",
                    trade_date=pos.trade_date,
                    entity_id=pos.position_id,
                    message=f"Position {pos.position_id} became available after batch as_of_utc.",
                )
            )

    # Validate trades
    trade_ids: Set[str] = set()
    for tr in batch.trades:
        _check_date(tr.trade_date, "trade.trade_date")
        _check_currency(tr.local_currency, "trade.local_currency")
        if tr.trade_id in trade_ids:
            raise ValidationError(f"Duplicate trade_id: {tr.trade_id}")
        trade_ids.add(tr.trade_id)
        if tr.side.upper() not in {"BUY", "SELL"}:
            raise ValidationError(f"Trade {tr.trade_id} invalid side: {tr.side!r}")
        if tr.quantity <= 0 or tr.order_quantity < tr.quantity:
            raise ValidationError(
                f"Trade {tr.trade_id} invalid quantities: quantity={tr.quantity}, order_quantity={tr.order_quantity}"
            )
        for fname in (
            "decision_price_local",
            "arrival_price_local",
            "execution_price_local",
            "close_price_local",
            "commission_local",
            "exchange_fee_local",
            "tax_local",
        ):
            val = getattr(tr, fname)
            _check_finite(val, f"trade.{fname}")
            if fname.endswith("_local") and "price" in fname and val <= 0:
                raise ValidationError(f"Trade {tr.trade_id} non-positive {fname}: {val}")
        if tr.benchmark_vwap_local is not None:
            _check_finite(tr.benchmark_vwap_local, "trade.benchmark_vwap_local")
        parse_utc(tr.execution_ts_utc)
        parse_utc(tr.availability_utc)

    # Validate cash flows
    cf_ids: Set[str] = set()
    for cf in batch.cash_flows:
        _check_date(cf.trade_date, "cash_flow.trade_date")
        _check_currency(cf.currency, "cash_flow.currency")
        if cf.cash_flow_id in cf_ids:
            raise ValidationError(f"Duplicate cash_flow_id: {cf.cash_flow_id}")
        cf_ids.add(cf.cash_flow_id)
        if cf.timing not in {"START_OF_DAY", "END_OF_DAY"}:
            raise ValidationError(f"CashFlow {cf.cash_flow_id} invalid timing: {cf.timing!r}")
        _check_finite(cf.amount_local, "cash_flow.amount_local")
        parse_utc(cf.availability_utc)

    # Validate fee adjustments
    fee_ids: Set[str] = set()
    for fee in batch.fee_adjustments:
        _check_date(fee.trade_date, "fee.trade_date")
        _check_currency(fee.currency, "fee.currency")
        if fee.fee_id in fee_ids:
            raise ValidationError(f"Duplicate fee_id: {fee.fee_id}")
        fee_ids.add(fee.fee_id)
        _check_finite(fee.amount_local, "fee.amount_local")
        parse_utc(fee.availability_utc)
        if fee.is_correction:
            exceptions.append(
                DataQualityException(
                    code="FEE_CORRECTION_APPLIED",
                    severity="INFO",
                    trade_date=fee.trade_date,
                    entity_id=fee.fee_id,
                    message=(
                        f"Fee correction {fee.fee_id} ({fee.amount_local:+.4f} {fee.currency}) applied"
                        + (f" superseding {fee.supersedes_fee_id}" if fee.supersedes_fee_id else "")
                        + (f": {fee.correction_reason}" if fee.correction_reason else "")
                    ),
                    details={
                        "fee_id": fee.fee_id,
                        "supersedes_fee_id": fee.supersedes_fee_id,
                        "amount_local": fee.amount_local,
                        "currency": fee.currency,
                        "reason": fee.correction_reason,
                    },
                )
            )

    # Validate FX observations
    for fx in batch.fx_rates:
        _check_date(fx.trade_date, "fx.trade_date")
        _check_currency(fx.base_currency, "fx.base_currency")
        _check_currency(fx.quote_currency, "fx.quote_currency")
        _check_finite(fx.rate, "fx.rate")
        if fx.rate <= 0:
            raise ValidationError(f"FX rate for {fx.pair} on {fx.trade_date} must be > 0.")
        parse_utc(fx.as_of_utc)
        parse_utc(fx.availability_utc)

    # Validate benchmark observations
    bm_dates: Set[str] = set()
    for bm in batch.benchmark:
        _check_date(bm.trade_date, "benchmark.trade_date")
        _check_currency(bm.currency, "benchmark.currency")
        _check_finite(bm.total_return, "benchmark.total_return")
        if bm.trade_date in bm_dates:
            raise ValidationError(f"Duplicate benchmark observation for {bm.trade_date}")
        bm_dates.add(bm.trade_date)
        parse_utc(bm.as_of_utc)
        parse_utc(bm.availability_utc)

    # Validate factor observations
    for fac in batch.factors:
        _check_date(fac.trade_date, "factor.trade_date")
        _check_currency(fac.currency, "factor.currency")
        _check_finite(fac.risk_free_rate, "factor.risk_free_rate")
        for fname, fval in fac.factor_returns.items():
            _check_finite(fval, f"factor.{fname}")
        parse_utc(fac.availability_utc)

    return exceptions
