"""Daily and Carino-linked strategy, instrument, sector, and currency contribution engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from pae.config import AttributionConfig
from pae.domain.models import (
    AttributionInputBatch,
    DataQualityException,
    FeeAdjustment,
    PositionSnapshot,
    TradeExecution,
)
from pae.engine.returns import FxConverter, ReturnSummary


@dataclass(frozen=True)
class DailyPositionContribution:
    """Single-day position contribution with exact FX, rebalance, fee, and provenance links."""

    trade_date: str
    position_id: str
    strategy_id: str
    instrument_id: str
    sector: str
    asset_class: str
    local_currency: str
    trade_ids: List[str]
    fee_adjustment_ids: List[str]
    start_quantity: float
    end_quantity: float
    fx_start: float
    fx_end: float
    start_mv_base: float
    end_mv_base: float
    start_weight: float
    end_weight: float
    local_return: Optional[float]
    fx_return: float
    base_return: Optional[float]
    local_price_pnl_base: float
    fx_translation_pnl_base: float
    fx_interaction_pnl_base: float
    trading_pnl_base: float
    income_pnl_base: float
    gross_pnl_base: float
    fees_base: float
    net_pnl_base: float
    local_price_contribution: float
    fx_translation_contribution: float
    fx_interaction_contribution: float
    fx_total_contribution: float
    trading_contribution: float
    income_contribution: float
    gross_contribution: float
    fee_drag_contribution: float
    net_contribution: float
    valuation_rounding_delta_base: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DailyContributionSummary:
    """Daily sum of contributions vs. portfolio return showing exact residual."""

    trade_date: str
    capital_base: float
    portfolio_gross_return: float
    portfolio_net_return: float
    sum_position_gross_contribution: float
    sum_position_net_contribution: float
    sum_position_fee_drag: float
    cash_contribution: float
    fund_level_fee_drag: float
    explained_gross_return: float
    explained_net_return: float
    gross_residual: float
    net_residual: float
    total_valuation_rounding_delta_base: float
    position_count: int
    trade_count: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AggregatedContribution:
    """Multi-period Carino-linked contribution grouped by strategy, instrument, sector, or currency."""

    group_type: str
    group_key: str
    avg_weight: float
    linked_local_price_contribution: float
    linked_fx_contribution: float
    linked_trading_contribution: float
    linked_income_contribution: float
    linked_gross_contribution: float
    linked_fee_drag_contribution: float
    linked_net_contribution: float
    total_gross_pnl_base: float
    total_fees_base: float
    total_net_pnl_base: float
    position_ids: List[str] = field(default_factory=list)
    trade_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ContributionReport:
    """Complete daily and multi-period contribution breakdown with full provenance."""

    linking_method: str
    daily_positions: List[DailyPositionContribution]
    daily_summaries: List[DailyContributionSummary]
    by_strategy: List[AggregatedContribution]
    by_instrument: List[AggregatedContribution]
    by_sector: List[AggregatedContribution]
    by_currency: List[AggregatedContribution]
    linked_cash_contribution: float
    linked_fund_level_fee_drag: float
    linked_gross_residual: float
    linked_net_residual: float
    sum_linked_gross_contributions: float
    sum_linked_net_contributions: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def compute_contributions(
    batch: AttributionInputBatch,
    config: AttributionConfig,
    fx: FxConverter,
    returns: ReturnSummary,
    exceptions: List[DataQualityException],
) -> ContributionReport:
    """Compute daily position contributions with exact FX/rebalance decomposition and Carino linking."""
    navs = sorted(batch.nav_snapshots, key=lambda n: n.trade_date)
    nav_by_date = {n.trade_date: n for n in navs}
    dates = [n.trade_date for n in navs]
    ret_by_date = {r.trade_date: r for r in returns.daily_returns}

    # Group positions, trades, and fee adjustments by date and (strategy, instrument)
    pos_by_date: Dict[str, List[PositionSnapshot]] = {}
    for pos in batch.positions:
        pos_by_date.setdefault(pos.trade_date, []).append(pos)

    trades_by_key: Dict[Tuple[str, str, str], List[TradeExecution]] = {}
    trades_by_date: Dict[str, List[TradeExecution]] = {}
    for tr in batch.trades:
        trades_by_key.setdefault((tr.trade_date, tr.strategy_id, tr.instrument_id), []).append(tr)
        trades_by_date.setdefault(tr.trade_date, []).append(tr)

    pos_fees_by_key: Dict[Tuple[str, str, str], List[FeeAdjustment]] = {}
    strategy_fees_by_key: Dict[Tuple[str, str], List[FeeAdjustment]] = {}
    fund_fees_by_date: Dict[str, List[FeeAdjustment]] = {}
    for fee in batch.fee_adjustments:
        if fee.strategy_id and fee.instrument_id:
            pos_fees_by_key.setdefault((fee.trade_date, fee.strategy_id, fee.instrument_id), []).append(fee)
        elif fee.strategy_id:
            strategy_fees_by_key.setdefault((fee.trade_date, fee.strategy_id), []).append(fee)
        else:
            fund_fees_by_date.setdefault(fee.trade_date, []).append(fee)

    daily_positions: List[DailyPositionContribution] = []
    daily_summaries: List[DailyContributionSummary] = []

    for idx, d in enumerate(dates):
        prior_d = dates[idx - 1] if idx > 0 else None
        nav = nav_by_date[d]
        dret = ret_by_date[d]
        capital_base = dret.capital_base
        end_nav_base = dret.end_nav_base
        day_positions = sorted(pos_by_date.get(d, []), key=lambda p: (p.strategy_id, p.instrument_id))

        # Check for orphan trades (trades with no matching position snapshot on that date)
        matched_trade_ids = set()

        sum_pos_gross_c = 0.0
        sum_pos_net_c = 0.0
        sum_pos_fee_c = 0.0
        total_rounding_delta = 0.0

        # Allocate strategy-only fee adjustments proportionally across positions in that strategy
        strat_mv_totals: Dict[str, float] = {}
        pos_fx_cache: Dict[str, Tuple[float, float]] = {}
        for pos in day_positions:
            fx_0 = fx.get_prior_rate(pos.local_currency, d, prior_d, exceptions=exceptions, entity_id=pos.position_id)
            fx_1 = fx.get_rate(pos.local_currency, d, exceptions=exceptions, entity_id=pos.position_id, as_of_utc=pos.as_of_utc)
            if fx_0 is None:
                fx_0 = fx_1 if fx_1 is not None else 1.0
            if fx_1 is None:
                fx_1 = fx_0
            pos_fx_cache[pos.position_id] = (fx_0, fx_1)
            mv0 = abs(pos.start_quantity * pos.start_price_local * fx_0)
            mv1 = abs(pos.end_quantity * pos.end_price_local * fx_1)
            strat_mv_totals[pos.strategy_id] = strat_mv_totals.get(pos.strategy_id, 0.0) + max(mv0, mv1, 1.0)

        # Track strategy fees that could not be attached to any position (fall back to fund fees)
        unallocated_strat_fees_base = 0.0
        for (fd, sid), flist in strategy_fees_by_key.items():
            if fd == d and sid not in strat_mv_totals:
                for fee in flist:
                    rate = fx.get_rate(fee.currency, d, exceptions=exceptions, entity_id=fee.fee_id) or 1.0
                    unallocated_strat_fees_base += fee.amount_local * rate

        for pos in day_positions:
            fx_0, fx_1 = pos_fx_cache[pos.position_id]
            pos_trades = trades_by_key.get((d, pos.strategy_id, pos.instrument_id), [])
            for tr in pos_trades:
                matched_trade_ids.add(tr.trade_id)

            # Quantity bridge check for rebalance integrity
            net_traded_qty = sum(tr.signed_quantity for tr in pos_trades)
            expected_end_qty = pos.start_quantity + net_traded_qty
            if abs(expected_end_qty - pos.end_quantity) > 1e-6:
                exceptions.append(
                    DataQualityException(
                        code="QUANTITY_MISMATCH",
                        severity="ERROR",
                        trade_date=d,
                        entity_id=pos.position_id,
                        message=(
                            f"Position {pos.position_id} start_qty ({pos.start_quantity}) + traded ({net_traded_qty}) "
                            f"= {expected_end_qty} != end_qty ({pos.end_quantity})."
                        ),
                        details={
                            "start_quantity": pos.start_quantity,
                            "traded_quantity": net_traded_qty,
                            "end_quantity": pos.end_quantity,
                        },
                    )
                )

            mv_start_calc = pos.start_quantity * pos.start_price_local * fx_0
            mv_end_calc = pos.end_quantity * pos.end_price_local * fx_1

            # Check custodian-reported vs computed base MV (detects rounding residuals)
            rounding_delta = 0.0
            if pos.start_market_value_base is not None:
                rounding_delta -= (pos.start_market_value_base - mv_start_calc)
            if pos.end_market_value_base is not None:
                rounding_delta += (pos.end_market_value_base - mv_end_calc)
            total_rounding_delta += rounding_delta

            # Exact additive hold P&L decomposition (Karnosky-Singer / geometric identity):
            # q_0*P_1*fx_1 - q_0*P_0*fx_0 = q_0*(P_1-P_0)*fx_0 + q_0*P_0*(fx_1-fx_0) + q_0*(P_1-P_0)*(fx_1-fx_0)
            local_price_pnl = pos.start_quantity * (pos.end_price_local - pos.start_price_local) * fx_0
            fx_translation_pnl = pos.start_quantity * pos.start_price_local * (fx_1 - fx_0)
            fx_interaction_pnl = pos.start_quantity * (pos.end_price_local - pos.start_price_local) * (fx_1 - fx_0)

            # Intraday rebalance / trading P&L (mark-to-close on executed trades)
            trading_pnl = sum(
                tr.signed_quantity * (pos.end_price_local - tr.execution_price_local) * fx_1
                for tr in pos_trades
            )
            income_pnl = pos.income_local * fx_1
            gross_pnl = local_price_pnl + fx_translation_pnl + fx_interaction_pnl + trading_pnl + income_pnl

            # Fees attributable to this position
            fee_adj_ids: List[str] = []
            pos_fee_base = pos.explicit_fees_local * fx_1
            pos_fee_base += sum(tr.total_explicit_fee_local * fx_1 for tr in pos_trades)
            for fee in pos_fees_by_key.get((d, pos.strategy_id, pos.instrument_id), []):
                rate = fx.get_rate(fee.currency, d, exceptions=exceptions, entity_id=fee.fee_id) or 1.0
                pos_fee_base += fee.amount_local * rate
                fee_adj_ids.append(fee.fee_id)

            # Strategy-level fees pro-rated across positions in that strategy
            strat_fees = strategy_fees_by_key.get((d, pos.strategy_id), [])
            if strat_fees:
                denom_strat = strat_mv_totals.get(pos.strategy_id, 1.0)
                share = max(abs(mv_start_calc), abs(mv_end_calc), 1.0) / denom_strat
                for fee in strat_fees:
                    rate = fx.get_rate(fee.currency, d, exceptions=exceptions, entity_id=fee.fee_id) or 1.0
                    pos_fee_base += fee.amount_local * rate * share
                    fee_adj_ids.append(fee.fee_id)

            net_pnl = gross_pnl - pos_fee_base

            # Return & weight calculations (guarded against zero capital base)
            if dret.is_zero_nav_day or capital_base <= 1e-12:
                start_w = 0.0
                local_c = fx_trans_c = fx_int_c = fx_tot_c = trade_c = inc_c = gross_c = fee_c = net_c = 0.0
            else:
                start_w = mv_start_calc / capital_base
                local_c = local_price_pnl / capital_base
                fx_trans_c = fx_translation_pnl / capital_base
                fx_int_c = fx_interaction_pnl / capital_base
                fx_tot_c = fx_trans_c + fx_int_c
                trade_c = trading_pnl / capital_base
                inc_c = income_pnl / capital_base
                gross_c = gross_pnl / capital_base
                fee_c = pos_fee_base / capital_base
                net_c = net_pnl / capital_base

            end_w = (mv_end_calc / end_nav_base) if end_nav_base > 1e-12 else 0.0

            local_ret: Optional[float] = None
            base_ret: Optional[float] = None
            fx_ret = (fx_1 / fx_0) - 1.0 if fx_0 > 0 else 0.0
            if pos.start_price_local > 0:
                local_ret = (pos.end_price_local - pos.start_price_local) / pos.start_price_local
                base_ret = (1.0 + local_ret) * (1.0 + fx_ret) - 1.0

            sum_pos_gross_c += gross_c
            sum_pos_net_c += net_c
            sum_pos_fee_c += fee_c

            daily_positions.append(
                DailyPositionContribution(
                    trade_date=d,
                    position_id=pos.position_id,
                    strategy_id=pos.strategy_id,
                    instrument_id=pos.instrument_id,
                    sector=pos.sector,
                    asset_class=pos.asset_class,
                    local_currency=pos.local_currency.upper(),
                    trade_ids=[tr.trade_id for tr in pos_trades],
                    fee_adjustment_ids=fee_adj_ids,
                    start_quantity=pos.start_quantity,
                    end_quantity=pos.end_quantity,
                    fx_start=fx_0,
                    fx_end=fx_1,
                    start_mv_base=mv_start_calc,
                    end_mv_base=mv_end_calc,
                    start_weight=start_w,
                    end_weight=end_w,
                    local_return=local_ret,
                    fx_return=fx_ret,
                    base_return=base_ret,
                    local_price_pnl_base=local_price_pnl,
                    fx_translation_pnl_base=fx_translation_pnl,
                    fx_interaction_pnl_base=fx_interaction_pnl,
                    trading_pnl_base=trading_pnl,
                    income_pnl_base=income_pnl,
                    gross_pnl_base=gross_pnl,
                    fees_base=pos_fee_base,
                    net_pnl_base=net_pnl,
                    local_price_contribution=local_c,
                    fx_translation_contribution=fx_trans_c,
                    fx_interaction_contribution=fx_int_c,
                    fx_total_contribution=fx_tot_c,
                    trading_contribution=trade_c,
                    income_contribution=inc_c,
                    gross_contribution=gross_c,
                    fee_drag_contribution=fee_c,
                    net_contribution=net_c,
                    valuation_rounding_delta_base=rounding_delta,
                )
            )

        # Warn on orphan trades not linked to any position
        for tr in trades_by_date.get(d, []):
            if tr.trade_id not in matched_trade_ids:
                exceptions.append(
                    DataQualityException(
                        code="ORPHAN_TRADE",
                        severity="WARNING",
                        trade_date=d,
                        entity_id=tr.trade_id,
                        message=(
                            f"Trade {tr.trade_id} ({tr.strategy_id}/{tr.instrument_id}) has no matching "
                            f"PositionSnapshot on {d}."
                        ),
                    )
                )

        # Cash interest & fund-level fees
        fund_fees_today = unallocated_strat_fees_base
        for fee in fund_fees_by_date.get(d, []):
            rate = fx.get_rate(fee.currency, d, exceptions=exceptions, entity_id=fee.fee_id) or 1.0
            fund_fees_today += fee.amount_local * rate

        if dret.is_zero_nav_day or capital_base <= 1e-12:
            cash_c = 0.0
            fund_fee_c = 0.0
        else:
            cash_c = nav.cash_interest_base / capital_base
            fund_fee_c = fund_fees_today / capital_base

        explained_gross = sum_pos_gross_c + cash_c
        explained_net = sum_pos_net_c + cash_c - fund_fee_c
        gross_res = dret.gross_return - explained_gross
        net_res = dret.net_return - explained_net

        daily_summaries.append(
            DailyContributionSummary(
                trade_date=d,
                capital_base=capital_base,
                portfolio_gross_return=dret.gross_return,
                portfolio_net_return=dret.net_return,
                sum_position_gross_contribution=sum_pos_gross_c,
                sum_position_net_contribution=sum_pos_net_c,
                sum_position_fee_drag=sum_pos_fee_c,
                cash_contribution=cash_c,
                fund_level_fee_drag=fund_fee_c,
                explained_gross_return=explained_gross,
                explained_net_return=explained_net,
                gross_residual=gross_res,
                net_residual=net_res,
                total_valuation_rounding_delta_base=total_rounding_delta,
                position_count=len(day_positions),
                trade_count=len(trades_by_date.get(d, [])),
            )
        )

    # Multi-period Carino linking
    w_net = returns.carino_weights_net
    w_gross = returns.carino_weights_gross
    n_valid = max(returns.valid_return_days, 1)

    def aggregate(group_type: str, key_fn) -> List[AggregatedContribution]:
        buckets: Dict[str, Dict[str, object]] = {}
        for pc in daily_positions:
            k = key_fn(pc)
            wn = w_net.get(pc.trade_date, 0.0)
            wg = w_gross.get(pc.trade_date, 0.0)
            b = buckets.setdefault(
                k,
                {
                    "weight_sum": 0.0,
                    "local": 0.0,
                    "fx": 0.0,
                    "trading": 0.0,
                    "income": 0.0,
                    "gross": 0.0,
                    "fee": 0.0,
                    "net": 0.0,
                    "gross_pnl": 0.0,
                    "fees_pnl": 0.0,
                    "net_pnl": 0.0,
                    "pos_ids": [],
                    "trade_ids": [],
                },
            )
            b["weight_sum"] = float(b["weight_sum"]) + pc.start_weight
            b["local"] = float(b["local"]) + wg * pc.local_price_contribution
            b["fx"] = float(b["fx"]) + wg * pc.fx_total_contribution
            b["trading"] = float(b["trading"]) + wg * pc.trading_contribution
            b["income"] = float(b["income"]) + wg * pc.income_contribution
            b["gross"] = float(b["gross"]) + wg * pc.gross_contribution
            b["fee"] = float(b["fee"]) + wn * pc.fee_drag_contribution
            b["net"] = float(b["net"]) + wn * pc.net_contribution
            b["gross_pnl"] = float(b["gross_pnl"]) + pc.gross_pnl_base
            b["fees_pnl"] = float(b["fees_pnl"]) + pc.fees_base
            b["net_pnl"] = float(b["net_pnl"]) + pc.net_pnl_base
            b["pos_ids"].append(pc.position_id)  # type: ignore[union-attr]
            b["trade_ids"].extend(pc.trade_ids)  # type: ignore[union-attr]

        out: List[AggregatedContribution] = []
        for k in sorted(buckets.keys()):
            b = buckets[k]
            out.append(
                AggregatedContribution(
                    group_type=group_type,
                    group_key=k,
                    avg_weight=float(b["weight_sum"]) / n_valid,
                    linked_local_price_contribution=float(b["local"]),
                    linked_fx_contribution=float(b["fx"]),
                    linked_trading_contribution=float(b["trading"]),
                    linked_income_contribution=float(b["income"]),
                    linked_gross_contribution=float(b["gross"]),
                    linked_fee_drag_contribution=float(b["fee"]),
                    linked_net_contribution=float(b["net"]),
                    total_gross_pnl_base=float(b["gross_pnl"]),
                    total_fees_base=float(b["fees_pnl"]),
                    total_net_pnl_base=float(b["net_pnl"]),
                    position_ids=sorted(set(b["pos_ids"])),  # type: ignore[arg-type]
                    trade_ids=sorted(set(b["trade_ids"])),  # type: ignore[arg-type]
                )
            )
        return out

    by_strategy = aggregate("STRATEGY", lambda p: p.strategy_id)
    by_instrument = aggregate("INSTRUMENT", lambda p: p.instrument_id)
    by_sector = aggregate("SECTOR", lambda p: p.sector)
    by_currency = aggregate("CURRENCY", lambda p: p.local_currency)

    linked_cash_gross = sum(w_gross.get(s.trade_date, 0.0) * s.cash_contribution for s in daily_summaries)
    linked_cash_net = sum(w_net.get(s.trade_date, 0.0) * s.cash_contribution for s in daily_summaries)
    linked_fund_fees_net = sum(w_net.get(s.trade_date, 0.0) * s.fund_level_fee_drag for s in daily_summaries)
    linked_gross_res = sum(w_gross.get(s.trade_date, 0.0) * s.gross_residual for s in daily_summaries)
    linked_net_res = sum(w_net.get(s.trade_date, 0.0) * s.net_residual for s in daily_summaries)

    sum_linked_gross = sum(x.linked_gross_contribution for x in by_strategy) + linked_cash_gross
    sum_linked_net = sum(x.linked_net_contribution for x in by_strategy) + linked_cash_net - linked_fund_fees_net

    return ContributionReport(
        linking_method=config.multi_period_linking,
        daily_positions=daily_positions,
        daily_summaries=daily_summaries,
        by_strategy=by_strategy,
        by_instrument=by_instrument,
        by_sector=by_sector,
        by_currency=by_currency,
        linked_cash_contribution=linked_cash_net,
        linked_fund_level_fee_drag=linked_fund_fees_net,
        linked_gross_residual=linked_gross_res,
        linked_net_residual=linked_net_res,
        sum_linked_gross_contributions=sum_linked_gross,
        sum_linked_net_contributions=sum_linked_net,
    )
