"""Return, FX conversion, cash-flow adjustment, and aligned benchmark calculation engine."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from pae.config import AttributionConfig
from pae.domain.models import (
    AttributionInputBatch,
    BenchmarkObservation,
    CashFlow,
    DailyNavSnapshot,
    DataQualityException,
    FeeAdjustment,
    FxRateObservation,
    PositionSnapshot,
    TradeExecution,
)
from pae.domain.validation import parse_utc
from pae.errors import ZeroNavError


class FxConverter:
    """Point-in-time FX lookup supporting direct, inverse, and staleness checks."""

    def __init__(
        self,
        observations: List[FxRateObservation],
        reporting_currency: str,
        max_staleness_hours: float = 48.0,
    ) -> None:
        self.reporting_currency = reporting_currency.upper()
        self.max_staleness_hours = max_staleness_hours
        # Key: (base_ccy, quote_ccy, trade_date) -> FxRateObservation
        self._direct: Dict[Tuple[str, str, str], FxRateObservation] = {}
        self._by_pair: Dict[Tuple[str, str], List[FxRateObservation]] = {}
        for obs in observations:
            b = obs.base_currency.upper()
            q = obs.quote_currency.upper()
            self._direct[(b, q, obs.trade_date)] = obs
            self._by_pair.setdefault((b, q), []).append(obs)
        for pair_list in self._by_pair.values():
            pair_list.sort(key=lambda x: x.trade_date)

    def get_rate(
        self,
        local_currency: str,
        trade_date: str,
        *,
        exceptions: Optional[List[DataQualityException]] = None,
        entity_id: Optional[str] = None,
        as_of_utc: Optional[str] = None,
    ) -> Optional[float]:
        """Return FX rate converting 1 unit of local_currency into reporting_currency."""
        loc = local_currency.upper()
        rep = self.reporting_currency
        if loc == rep:
            return 1.0

        # Exact date direct match
        obs = self._direct.get((loc, rep, trade_date))
        if obs is not None:
            self._check_staleness(obs, as_of_utc, trade_date, exceptions)
            return obs.rate

        # Exact date inverse match
        inv = self._direct.get((rep, loc, trade_date))
        if inv is not None and inv.rate > 0:
            self._check_staleness(inv, as_of_utc, trade_date, exceptions)
            return 1.0 / inv.rate

        # Fallback: latest prior date (with explicit STALE_FX_RATE warning)
        candidates = [x for x in self._by_pair.get((loc, rep), []) if x.trade_date < trade_date]
        if candidates:
            latest = candidates[-1]
            if exceptions is not None:
                exceptions.append(
                    DataQualityException(
                        code="STALE_FX_RATE",
                        severity="WARNING",
                        trade_date=trade_date,
                        entity_id=entity_id or f"{loc}/{rep}",
                        message=(
                            f"No FX fixing for {loc}/{rep} on {trade_date}; "
                            f"carried forward rate {latest.rate} from {latest.trade_date}."
                        ),
                        details={"pair": f"{loc}/{rep}", "fallback_date": latest.trade_date, "rate": latest.rate},
                    )
                )
            return latest.rate

        inv_candidates = [x for x in self._by_pair.get((rep, loc), []) if x.trade_date < trade_date]
        if inv_candidates:
            latest = inv_candidates[-1]
            if exceptions is not None:
                exceptions.append(
                    DataQualityException(
                        code="STALE_FX_RATE",
                        severity="WARNING",
                        trade_date=trade_date,
                        entity_id=entity_id or f"{loc}/{rep}",
                        message=(
                            f"No FX fixing for {loc}/{rep} on {trade_date}; "
                            f"carried forward inverse rate from {latest.trade_date}."
                        ),
                    )
                )
            return 1.0 / latest.rate

        if exceptions is not None:
            exceptions.append(
                DataQualityException(
                    code="MISSING_FX_RATE",
                    severity="ERROR",
                    trade_date=trade_date,
                    entity_id=entity_id or f"{loc}/{rep}",
                    message=f"Missing FX rate for {loc}/{rep} on {trade_date}.",
                )
            )
        return None

    def get_prior_rate(
        self,
        local_currency: str,
        trade_date: str,
        prior_date: Optional[str],
        *,
        exceptions: Optional[List[DataQualityException]] = None,
        entity_id: Optional[str] = None,
    ) -> Optional[float]:
        """Get start-of-day FX rate (prior trading day close, or explicit prior fixing)."""
        loc = local_currency.upper()
        rep = self.reporting_currency
        if loc == rep:
            return 1.0

        if prior_date is not None:
            exact_prior = self._direct.get((loc, rep, prior_date))
            if exact_prior is not None:
                return exact_prior.rate
            inv_prior = self._direct.get((rep, loc, prior_date))
            if inv_prior is not None and inv_prior.rate > 0:
                return 1.0 / inv_prior.rate

        # Search latest observation strictly before trade_date
        candidates = [x for x in self._by_pair.get((loc, rep), []) if x.trade_date < trade_date]
        if candidates:
            return candidates[-1].rate
        inv_candidates = [x for x in self._by_pair.get((rep, loc), []) if x.trade_date < trade_date]
        if inv_candidates:
            return 1.0 / inv_candidates[-1].rate

        # Last resort: same-day rate (e.g. first day of new position entered intraday)
        return self.get_rate(local_currency, trade_date, exceptions=exceptions, entity_id=entity_id)

    def _check_staleness(
        self,
        obs: FxRateObservation,
        as_of_utc: Optional[str],
        trade_date: str,
        exceptions: Optional[List[DataQualityException]],
    ) -> None:
        if as_of_utc is None or exceptions is None:
            return
        try:
            age_hours = (parse_utc(as_of_utc) - parse_utc(obs.as_of_utc)).total_seconds() / 3600.0
        except Exception:
            return
        if age_hours > self.max_staleness_hours:
            exceptions.append(
                DataQualityException(
                    code="STALE_FX_RATE",
                    severity="WARNING",
                    trade_date=trade_date,
                    entity_id=obs.pair,
                    message=(
                        f"FX observation {obs.pair} on {trade_date} is {age_hours:.1f}h old "
                        f"(exceeds max_staleness_hours={self.max_staleness_hours})."
                    ),
                    details={"age_hours": round(age_hours, 2), "max_staleness_hours": self.max_staleness_hours},
                )
            )


@dataclass(frozen=True)
class DailyReturnRecord:
    """Daily portfolio & aligned benchmark return record with strict Gross vs Net separation."""

    trade_date: str
    reporting_currency: str
    start_nav_base: float
    end_nav_base: float
    cash_flow_sod_base: float
    cash_flow_eod_base: float
    capital_base: float
    gross_pnl_base: float
    total_fees_base: float
    net_pnl_base: float
    gross_return: float
    fee_drag_return: float
    net_return: float
    is_zero_nav_day: bool = False
    benchmark_id: Optional[str] = None
    benchmark_original_currency: Optional[str] = None
    benchmark_return_local: Optional[float] = None
    benchmark_fx_return: Optional[float] = None
    benchmark_return_aligned: Optional[float] = None
    active_gross_return: Optional[float] = None
    active_net_return: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReturnSummary:
    """Period-level return & benchmark summary with Carino linking weights."""

    methodology_version: str
    reporting_currency: str
    period_start: str
    period_end: str
    trading_days: int
    valid_return_days: int
    zero_nav_days: int
    cumulative_gross_return: float
    cumulative_net_return: float
    cumulative_fee_drag: float
    modified_dietz_net_return: Optional[float]
    total_external_cash_flows_base: float
    total_gross_pnl_base: float
    total_fees_base: float
    total_net_pnl_base: float
    benchmark_id: Optional[str]
    benchmark_aligned_days: int
    benchmark_missing_days: int
    cumulative_benchmark_return: Optional[float]
    cumulative_active_gross_return: Optional[float]
    cumulative_active_net_return: Optional[float]
    carino_weights_net: Dict[str, float] = field(default_factory=dict)
    carino_weights_gross: Dict[str, float] = field(default_factory=dict)
    daily_returns: List[DailyReturnRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        return d


def compute_carino_weights(daily_returns: List[Tuple[str, float]]) -> Dict[str, float]:
    """Compute exact Carino logarithmic smoothing weights so sum(w_t * r_t) == R_cum.

    For each day t:
        k_t = ln(1 + r_t) / r_t   (limit 1.0 as r_t -> 0)
    For total period R = prod(1 + r_t) - 1:
        k = ln(1 + R) / R         (limit 1.0 as R -> 0)
    Weight w_t = k_t / k.
    """
    if not daily_returns:
        return {}
    compounded = 1.0
    for _, r in daily_returns:
        if 1.0 + r <= 0.0:
            # Wipeout (-100% return): fall back to proportional geometric weights
            return {d: 1.0 for d, _ in daily_returns}
        compounded *= 1.0 + r
    total_r = compounded - 1.0
    k_total = 1.0 if abs(total_r) < 1e-12 else math.log1p(total_r) / total_r

    weights: Dict[str, float] = {}
    for trade_date, r in daily_returns:
        k_t = 1.0 if abs(r) < 1e-12 else math.log1p(r) / r
        weights[trade_date] = k_t / k_total
    return weights


def compute_returns_and_benchmark(
    batch: AttributionInputBatch,
    config: AttributionConfig,
    fx: FxConverter,
    exceptions: List[DataQualityException],
) -> ReturnSummary:
    """Calculate daily TWR gross & net returns, Modified Dietz cross-check, and aligned benchmark returns."""
    navs = sorted(batch.nav_snapshots, key=lambda n: n.trade_date)
    dates = [n.trade_date for n in navs]

    # Index records by date
    cfs_by_date: Dict[str, List[CashFlow]] = {}
    for cf in batch.cash_flows:
        cfs_by_date.setdefault(cf.trade_date, []).append(cf)

    pos_by_date: Dict[str, List[PositionSnapshot]] = {}
    for pos in batch.positions:
        pos_by_date.setdefault(pos.trade_date, []).append(pos)

    trades_by_date: Dict[str, List[TradeExecution]] = {}
    for tr in batch.trades:
        trades_by_date.setdefault(tr.trade_date, []).append(tr)

    fees_by_date: Dict[str, List[FeeAdjustment]] = {}
    for fee in batch.fee_adjustments:
        fees_by_date.setdefault(fee.trade_date, []).append(fee)

    bm_by_date: Dict[str, BenchmarkObservation] = {b.trade_date: b for b in batch.benchmark}

    daily_records: List[DailyReturnRecord] = []
    valid_net_pairs: List[Tuple[str, float]] = []
    valid_gross_pairs: List[Tuple[str, float]] = []
    bm_aligned_returns: List[float] = []

    total_cf_base = 0.0
    total_gross_pnl = 0.0
    total_fees_all = 0.0
    total_net_pnl = 0.0
    zero_nav_days = 0
    bm_missing_days = 0

    # For Modified Dietz cross-check
    n_days = len(navs)
    weighted_cf_sum = 0.0

    for idx, nav in enumerate(navs):
        d = nav.trade_date
        prior_d = dates[idx - 1] if idx > 0 else None

        # Convert external cash flows to reporting currency
        cf_sod = 0.0
        cf_eod = 0.0
        for cf in cfs_by_date.get(d, []):
            if cf.timing == "START_OF_DAY":
                rate = fx.get_prior_rate(cf.currency, d, prior_d, exceptions=exceptions, entity_id=cf.cash_flow_id)
                if rate is None:
                    rate = fx.get_rate(cf.currency, d, exceptions=exceptions, entity_id=cf.cash_flow_id) or 1.0
                amt = cf.amount_local * rate
                cf_sod += amt
                # SOD on day idx has (n_days - idx) / n_days remaining in period
                w_dietz = (n_days - idx) / n_days
                weighted_cf_sum += w_dietz * amt
            else:
                rate = fx.get_rate(cf.currency, d, exceptions=exceptions, entity_id=cf.cash_flow_id, as_of_utc=nav.as_of_utc)
                if rate is None:
                    rate = 1.0
                amt = cf.amount_local * rate
                cf_eod += amt
                # EOD on day idx has (n_days - idx - 1) / n_days remaining in period
                w_dietz = (n_days - idx - 1) / n_days
                weighted_cf_sum += w_dietz * amt
            total_cf_base += amt

        # Total explicit fees in reporting currency on day d:
        # 1) position non-trade fees (borrow/custody)
        # 2) trade explicit fees (commission + exchange_fee + tax)
        # 3) fund / strategy fee adjustments (management, admin, corrections)
        fees_today = 0.0
        for pos in pos_by_date.get(d, []):
            if pos.explicit_fees_local != 0.0:
                rate = fx.get_rate(pos.local_currency, d, exceptions=exceptions, entity_id=pos.position_id, as_of_utc=nav.as_of_utc) or 1.0
                fees_today += pos.explicit_fees_local * rate
        for tr in trades_by_date.get(d, []):
            if tr.total_explicit_fee_local != 0.0:
                rate = fx.get_rate(tr.local_currency, d, exceptions=exceptions, entity_id=tr.trade_id, as_of_utc=nav.as_of_utc) or 1.0
                fees_today += tr.total_explicit_fee_local * rate
        for fee in fees_by_date.get(d, []):
            rate = fx.get_rate(fee.currency, d, exceptions=exceptions, entity_id=fee.fee_id, as_of_utc=nav.as_of_utc) or 1.0
            fees_today += fee.amount_local * rate

        capital_base = nav.start_nav_base + cf_sod
        end_before_eod_cf = nav.end_nav_base - cf_eod
        net_pnl = end_before_eod_cf - capital_base
        gross_pnl = net_pnl + fees_today

        total_gross_pnl += gross_pnl
        total_fees_all += fees_today
        total_net_pnl += net_pnl

        is_zero_nav = capital_base <= 1e-12
        if is_zero_nav:
            zero_nav_days += 1
            msg = (
                f"Capital base (start_nav + start_of_day_cf) is {capital_base:.6f} <= 0 on {d}; "
                "return cannot be defined by division."
            )
            if config.fail_on_zero_nav:
                raise ZeroNavError(msg, details={"trade_date": d, "capital_base": capital_base})
            exceptions.append(
                DataQualityException(
                    code="ZERO_NAV",
                    severity="CRITICAL",
                    trade_date=d,
                    entity_id=nav.nav_id,
                    message=msg,
                    details={
                        "start_nav_base": nav.start_nav_base,
                        "cash_flow_sod_base": cf_sod,
                        "capital_base": capital_base,
                        "end_nav_base": nav.end_nav_base,
                    },
                )
            )
            gross_ret = 0.0
            fee_drag_ret = 0.0
            net_ret = 0.0
        else:
            gross_ret = gross_pnl / capital_base
            fee_drag_ret = fees_today / capital_base
            net_ret = net_pnl / capital_base
            if nav.end_nav_base <= 1e-12:
                exceptions.append(
                    DataQualityException(
                        code="ZERO_END_NAV",
                        severity="WARNING",
                        trade_date=d,
                        entity_id=nav.nav_id,
                        message=f"Portfolio end_nav_base reached {nav.end_nav_base:.6f} on {d}.",
                        details={"end_nav_base": nav.end_nav_base, "net_return": net_ret},
                    )
                )
            valid_net_pairs.append((d, net_ret))
            valid_gross_pairs.append((d, gross_ret))

        # Benchmark date & currency alignment
        bm = bm_by_date.get(d)
        bm_id = batch.benchmark_id
        bm_ccy: Optional[str] = None
        bm_ret_local: Optional[float] = None
        bm_fx_ret: Optional[float] = None
        bm_ret_aligned: Optional[float] = None
        active_gross: Optional[float] = None
        active_net: Optional[float] = None

        if bm is None:
            if batch.benchmark_id is not None or len(batch.benchmark) > 0:
                bm_missing_days += 1
                exceptions.append(
                    DataQualityException(
                        code="MISSING_BENCHMARK",
                        severity="WARNING",
                        trade_date=d,
                        entity_id=batch.benchmark_id or "BENCHMARK",
                        message=f"No benchmark observation available for aligned date {d}; active return omitted.",
                    )
                )
        else:
            bm_id = bm.benchmark_id
            bm_ccy = bm.currency.upper()
            bm_ret_local = bm.total_return
            if bm_ccy == batch.reporting_currency.upper():
                bm_fx_ret = 0.0
                bm_ret_aligned = bm_ret_local
            else:
                fx_0 = fx.get_prior_rate(bm_ccy, d, prior_d, exceptions=exceptions, entity_id=bm.benchmark_id)
                fx_1 = fx.get_rate(bm_ccy, d, exceptions=exceptions, entity_id=bm.benchmark_id, as_of_utc=bm.as_of_utc)
                if fx_0 is None or fx_1 is None or fx_0 <= 0:
                    bm_missing_days += 1
                    exceptions.append(
                        DataQualityException(
                            code="BENCHMARK_CURRENCY_MISMATCH",
                            severity="ERROR",
                            trade_date=d,
                            entity_id=bm.benchmark_id,
                            message=(
                                f"Benchmark {bm.benchmark_id} is denominated in {bm_ccy} vs portfolio "
                                f"{batch.reporting_currency} and FX rates are unavailable for {d}."
                            ),
                        )
                    )
                else:
                    bm_fx_ret = (fx_1 / fx_0) - 1.0
                    bm_ret_aligned = (1.0 + bm_ret_local) * (1.0 + bm_fx_ret) - 1.0

            if bm_ret_aligned is not None and not is_zero_nav:
                bm_aligned_returns.append(bm_ret_aligned)
                active_gross = gross_ret - bm_ret_aligned
                active_net = net_ret - bm_ret_aligned

        daily_records.append(
            DailyReturnRecord(
                trade_date=d,
                reporting_currency=batch.reporting_currency,
                start_nav_base=nav.start_nav_base,
                end_nav_base=nav.end_nav_base,
                cash_flow_sod_base=cf_sod,
                cash_flow_eod_base=cf_eod,
                capital_base=capital_base,
                gross_pnl_base=gross_pnl,
                total_fees_base=fees_today,
                net_pnl_base=net_pnl,
                gross_return=gross_ret,
                fee_drag_return=fee_drag_ret,
                net_return=net_ret,
                is_zero_nav_day=is_zero_nav,
                benchmark_id=bm_id,
                benchmark_original_currency=bm_ccy,
                benchmark_return_local=bm_ret_local,
                benchmark_fx_return=bm_fx_ret,
                benchmark_return_aligned=bm_ret_aligned,
                active_gross_return=active_gross,
                active_net_return=active_net,
            )
        )

    # Compound multi-period returns
    cum_gross = 1.0
    for _, r in valid_gross_pairs:
        cum_gross *= 1.0 + r
    cum_gross -= 1.0

    cum_net = 1.0
    for _, r in valid_net_pairs:
        cum_net *= 1.0 + r
    cum_net -= 1.0

    cum_fee_drag = cum_gross - cum_net

    # Modified Dietz cross-check across the period
    dietz_net: Optional[float] = None
    if config.compute_modified_dietz_crosscheck and navs:
        dietz_denom = navs[0].start_nav_base + weighted_cf_sum
        if dietz_denom > 1e-12:
            dietz_net = (navs[-1].end_nav_base - navs[0].start_nav_base - total_cf_base) / dietz_denom

    # Cumulative benchmark return only if ALL valid return days have aligned benchmark data
    cum_bm: Optional[float] = None
    cum_active_gross: Optional[float] = None
    cum_active_net: Optional[float] = None
    if len(bm_aligned_returns) == len(valid_net_pairs) and len(bm_aligned_returns) > 0 and bm_missing_days == 0:
        c = 1.0
        for r in bm_aligned_returns:
            c *= 1.0 + r
        cum_bm = c - 1.0
        cum_active_gross = cum_gross - cum_bm
        cum_active_net = cum_net - cum_bm
    elif len(bm_aligned_returns) > 0 and bm_missing_days > 0:
        exceptions.append(
            DataQualityException(
                code="INCOMPLETE_BENCHMARK_SERIES",
                severity="WARNING",
                entity_id=batch.benchmark_id or "BENCHMARK",
                message=(
                    f"Benchmark has {bm_missing_days} missing/unaligned day(s) out of {len(navs)}; "
                    "cumulative benchmark and active returns are withheld to avoid date misalignment."
                ),
                details={"aligned_days": len(bm_aligned_returns), "missing_days": bm_missing_days},
            )
        )

    carino_net = compute_carino_weights(valid_net_pairs)
    carino_gross = compute_carino_weights(valid_gross_pairs)

    return ReturnSummary(
        methodology_version=config.methodology_version,
        reporting_currency=batch.reporting_currency,
        period_start=dates[0],
        period_end=dates[-1],
        trading_days=len(navs),
        valid_return_days=len(valid_net_pairs),
        zero_nav_days=zero_nav_days,
        cumulative_gross_return=cum_gross,
        cumulative_net_return=cum_net,
        cumulative_fee_drag=cum_fee_drag,
        modified_dietz_net_return=dietz_net,
        total_external_cash_flows_base=total_cf_base,
        total_gross_pnl_base=total_gross_pnl,
        total_fees_base=total_fees_all,
        total_net_pnl_base=total_net_pnl,
        benchmark_id=batch.benchmark_id or (batch.benchmark[0].benchmark_id if batch.benchmark else None),
        benchmark_aligned_days=len(bm_aligned_returns),
        benchmark_missing_days=bm_missing_days,
        cumulative_benchmark_return=cum_bm,
        cumulative_active_gross_return=cum_active_gross,
        cumulative_active_net_return=cum_active_net,
        carino_weights_net=carino_net,
        carino_weights_gross=carino_gross,
        daily_returns=daily_records,
    )
