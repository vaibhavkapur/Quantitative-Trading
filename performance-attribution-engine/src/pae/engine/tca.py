"""Execution-cost attribution (Perold Implementation Shortfall & VWAP slippage)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from pae.config import AttributionConfig
from pae.domain.models import AttributionInputBatch, DataQualityException
from pae.engine.returns import FxConverter, ReturnSummary


@dataclass(frozen=True)
class TradeTCARecord:
    """Implementation Shortfall decomposition for a single executed trade."""

    trade_id: str
    order_id: str
    trade_date: str
    strategy_id: str
    instrument_id: str
    sector: str
    venue: str
    side: str
    local_currency: str
    fx_rate: float
    order_quantity: float
    executed_quantity: float
    fill_rate: float
    notional_decision_base: float
    notional_executed_base: float
    delay_cost_base: float
    market_impact_cost_base: float
    opportunity_cost_base: float
    explicit_cost_base: float
    total_is_cost_base: float
    total_is_bps_of_notional: float
    total_is_bps_of_nav: float
    vwap_slippage_base: Optional[float]
    vwap_slippage_bps: Optional[float]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AggregatedTCABucket:
    """Execution-cost totals grouped by strategy, instrument, venue, or side."""

    group_type: str
    group_key: str
    trade_count: int
    total_executed_notional_base: float
    delay_cost_base: float
    market_impact_cost_base: float
    opportunity_cost_base: float
    explicit_cost_base: float
    total_is_cost_base: float
    is_bps_of_notional: float
    avg_fill_rate: float
    trade_ids: List[str] = field(default_factory=list)
    order_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionCostReport:
    """Portfolio-wide execution-cost attribution report linked to trade provenance."""

    benchmark_type: str
    trade_count: int
    total_executed_notional_base: float
    total_delay_cost_base: float
    total_market_impact_cost_base: float
    total_opportunity_cost_base: float
    total_explicit_cost_base: float
    total_implementation_shortfall_base: float
    total_is_bps_of_notional: float
    total_is_return_drag: float
    implicit_cost_return_drag: float
    explicit_trade_cost_return_drag: float
    by_strategy: List[AggregatedTCABucket] = field(default_factory=list)
    by_instrument: List[AggregatedTCABucket] = field(default_factory=list)
    by_venue: List[AggregatedTCABucket] = field(default_factory=list)
    trades: List[TradeTCARecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def compute_execution_cost_attribution(
    batch: AttributionInputBatch,
    config: AttributionConfig,
    fx: FxConverter,
    returns: ReturnSummary,
    exceptions: List[DataQualityException],
) -> ExecutionCostReport:
    """Decompose trade-level Implementation Shortfall into delay, impact, opportunity, and explicit costs."""
    ret_by_date = {r.trade_date: r for r in returns.daily_returns}
    w_net = returns.carino_weights_net

    trade_records: List[TradeTCARecord] = []
    tot_exec_notional = 0.0
    tot_delay = 0.0
    tot_impact = 0.0
    tot_opp = 0.0
    tot_explicit = 0.0
    tot_is = 0.0
    linked_is_drag = 0.0
    linked_implicit_drag = 0.0
    linked_explicit_drag = 0.0

    for tr in sorted(batch.trades, key=lambda t: (t.trade_date, t.execution_ts_utc, t.trade_id)):
        rate = fx.get_rate(tr.local_currency, tr.trade_date, exceptions=exceptions, entity_id=tr.trade_id) or 1.0
        sq = tr.signed_quantity
        uq = tr.unfilled_signed_quantity

        notional_dec = tr.order_quantity * tr.decision_price_local * rate
        notional_exec = tr.quantity * tr.execution_price_local * rate
        fill_rate = tr.quantity / tr.order_quantity if tr.order_quantity > 0 else 1.0

        # Perold (1988) Implementation Shortfall sign convention: positive = cost/drag
        delay_cost = sq * (tr.arrival_price_local - tr.decision_price_local) * rate
        impact_cost = sq * (tr.execution_price_local - tr.arrival_price_local) * rate
        opp_cost = (
            uq * (tr.close_price_local - tr.decision_price_local) * rate
            if config.tca_include_opportunity_cost
            else 0.0
        )
        explicit_cost = (
            tr.total_explicit_fee_local * rate
            if config.tca_include_explicit_fees
            else 0.0
        )
        total_cost = delay_cost + impact_cost + opp_cost + explicit_cost

        is_bps_notional = (total_cost / notional_exec * 10000.0) if notional_exec > 1e-12 else 0.0

        dret = ret_by_date.get(tr.trade_date)
        cap_base = dret.capital_base if dret is not None else 0.0
        if cap_base > 1e-12:
            is_ret_drag = total_cost / cap_base
            imp_ret_drag = (delay_cost + impact_cost + opp_cost) / cap_base
            exp_ret_drag = explicit_cost / cap_base
            is_bps_nav = is_ret_drag * 10000.0
        else:
            is_ret_drag = imp_ret_drag = exp_ret_drag = is_bps_nav = 0.0

        w = w_net.get(tr.trade_date, 1.0)
        linked_is_drag += w * is_ret_drag
        linked_implicit_drag += w * imp_ret_drag
        linked_explicit_drag += w * exp_ret_drag

        vwap_slip: Optional[float] = None
        vwap_bps: Optional[float] = None
        if tr.benchmark_vwap_local is not None:
            vwap_slip = sq * (tr.execution_price_local - tr.benchmark_vwap_local) * rate
            vwap_bps = (vwap_slip / notional_exec * 10000.0) if notional_exec > 1e-12 else 0.0

        tot_exec_notional += notional_exec
        tot_delay += delay_cost
        tot_impact += impact_cost
        tot_opp += opp_cost
        tot_explicit += explicit_cost
        tot_is += total_cost

        trade_records.append(
            TradeTCARecord(
                trade_id=tr.trade_id,
                order_id=tr.order_id,
                trade_date=tr.trade_date,
                strategy_id=tr.strategy_id,
                instrument_id=tr.instrument_id,
                sector=tr.sector,
                venue=tr.venue,
                side=tr.side.upper(),
                local_currency=tr.local_currency.upper(),
                fx_rate=rate,
                order_quantity=tr.order_quantity,
                executed_quantity=tr.quantity,
                fill_rate=fill_rate,
                notional_decision_base=notional_dec,
                notional_executed_base=notional_exec,
                delay_cost_base=delay_cost,
                market_impact_cost_base=impact_cost,
                opportunity_cost_base=opp_cost,
                explicit_cost_base=explicit_cost,
                total_is_cost_base=total_cost,
                total_is_bps_of_notional=is_bps_notional,
                total_is_bps_of_nav=is_bps_nav,
                vwap_slippage_base=vwap_slip,
                vwap_slippage_bps=vwap_bps,
            )
        )

    def aggregate(group_type: str, key_fn) -> List[AggregatedTCABucket]:
        buckets: Dict[str, List[TradeTCARecord]] = {}
        for r in trade_records:
            buckets.setdefault(key_fn(r), []).append(r)
        out: List[AggregatedTCABucket] = []
        for k in sorted(buckets.keys()):
            rows = buckets[k]
            notional = sum(x.notional_executed_base for x in rows)
            delay = sum(x.delay_cost_base for x in rows)
            impact = sum(x.market_impact_cost_base for x in rows)
            opp = sum(x.opportunity_cost_base for x in rows)
            exp = sum(x.explicit_cost_base for x in rows)
            tot = sum(x.total_is_cost_base for x in rows)
            bps = (tot / notional * 10000.0) if notional > 1e-12 else 0.0
            avg_fill = sum(x.fill_rate for x in rows) / len(rows)
            out.append(
                AggregatedTCABucket(
                    group_type=group_type,
                    group_key=k,
                    trade_count=len(rows),
                    total_executed_notional_base=notional,
                    delay_cost_base=delay,
                    market_impact_cost_base=impact,
                    opportunity_cost_base=opp,
                    explicit_cost_base=exp,
                    total_is_cost_base=tot,
                    is_bps_of_notional=bps,
                    avg_fill_rate=avg_fill,
                    trade_ids=[x.trade_id for x in rows],
                    order_ids=sorted({x.order_id for x in rows}),
                )
            )
        return out

    tot_bps = (tot_is / tot_exec_notional * 10000.0) if tot_exec_notional > 1e-12 else 0.0

    return ExecutionCostReport(
        benchmark_type=config.tca_default_benchmark,
        trade_count=len(trade_records),
        total_executed_notional_base=tot_exec_notional,
        total_delay_cost_base=tot_delay,
        total_market_impact_cost_base=tot_impact,
        total_opportunity_cost_base=tot_opp,
        total_explicit_cost_base=tot_explicit,
        total_implementation_shortfall_base=tot_is,
        total_is_bps_of_notional=tot_bps,
        total_is_return_drag=linked_is_drag,
        implicit_cost_return_drag=linked_implicit_drag,
        explicit_trade_cost_return_drag=linked_explicit_drag,
        by_strategy=aggregate("STRATEGY", lambda r: r.strategy_id),
        by_instrument=aggregate("INSTRUMENT", lambda r: r.instrument_id),
        by_venue=aggregate("VENUE", lambda r: r.venue),
        trades=trade_records,
    )
