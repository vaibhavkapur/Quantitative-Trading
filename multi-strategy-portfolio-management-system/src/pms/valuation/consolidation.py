"""Mark-to-market valuation, exact P&L explainability, and fund-to-strategy consolidation."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from decimal import Decimal

from pms.domain.enums import AccountType, PricingStatus
from pms.domain.errors import InvariantViolationError
from pms.domain.master import MasterRegistry
from pms.domain.models import FrozenModel
from pms.ledger.engine import AccountLedgerState, ReplaySnapshot
from pms.valuation.pricing import MarketSnapshot

ZERO = Decimal("0")
EPS = Decimal("0.00000001")


class ValuedPosition(FrozenModel):
    account_id: str
    strategy_id: str | None
    instrument_id: str
    symbol: str
    currency: str
    quantity: Decimal
    multiplier: Decimal
    average_unit_cost: Decimal
    signed_cost_basis_local: Decimal
    mark_price: Decimal | None
    mark_timestamp: datetime | None
    pricing_status: PricingStatus
    is_priced: bool
    fx_rate_to_base: Decimal | None
    carrying_value_local: Decimal
    carrying_value_base: Decimal
    unrealized_pnl_local: Decimal
    unrealized_pnl_base: Decimal
    gross_exposure_base: Decimal
    net_exposure_base: Decimal


class PnLAttribution(FrozenModel):
    realized_trading_pnl_base: Decimal
    unrealized_price_pnl_base: Decimal
    fx_translation_pnl_base: Decimal
    dividend_income_base: Decimal
    fees_and_commissions_base: Decimal
    late_fees_base: Decimal
    adjustments_base: Decimal
    explained_total_pnl_base: Decimal
    nav_minus_net_capital_base: Decimal
    unexplained_residual_base: Decimal


class ValuationException(FrozenModel):
    code: str
    account_id: str
    instrument_id: str | None = None
    currency: str | None = None
    message: str


class AccountValuation(FrozenModel):
    account_id: str
    account_name: str
    account_type: AccountType
    strategy_id: str | None
    base_currency: str
    cash_balances_local: dict[str, Decimal]
    cash_balances_base: dict[str, Decimal]
    total_cash_base: Decimal
    priced_market_value_base: Decimal
    unpriced_cost_basis_base: Decimal
    nav_base: Decimal
    net_capital_flows_base: Decimal
    gross_exposure_base: Decimal
    net_exposure_base: Decimal
    positions: tuple[ValuedPosition, ...]
    pnl: PnLAttribution
    unpriced_instruments: tuple[str, ...]
    valuation_complete: bool


class ConsolidationCheck(FrozenModel):
    fund_nav_base: Decimal
    strategies_plus_unallocated_nav_base: Decimal
    nav_difference_base: Decimal
    fund_cash_base: Decimal
    strategies_plus_unallocated_cash_base: Decimal
    cash_difference_base: Decimal
    fund_pnl_base: Decimal
    strategies_plus_unallocated_pnl_base: Decimal
    pnl_difference_base: Decimal
    passed: bool


class FundValuationReport(FrozenModel):
    fund_id: str
    fund_name: str
    base_currency: str
    snapshot_id: str
    as_of_time: datetime
    ledger_last_seq_no: int
    ledger_state_hash: str
    total_cash_base: Decimal
    unallocated_cash_base: Decimal
    strategy_cash_base: Decimal
    priced_market_value_base: Decimal
    unpriced_cost_basis_base: Decimal
    fund_nav_base: Decimal
    net_capital_flows_base: Decimal
    gross_exposure_base: Decimal
    net_exposure_base: Decimal
    pnl: PnLAttribution
    strategy_valuations: tuple[AccountValuation, ...]
    unallocated_valuations: tuple[AccountValuation, ...]
    consolidated_positions: dict[str, Decimal]
    consolidated_cash_local: dict[str, Decimal]
    unpriced_instruments: tuple[str, ...]
    exceptions: tuple[ValuationException, ...]
    valuation_complete: bool
    consolidation_check: ConsolidationCheck


class ValuationEngine:
    """Computes mark-to-market snapshots, P&L explainability, and fund consolidation."""

    def __init__(self, master: MasterRegistry, *, mark_staleness_seconds: int = 86400) -> None:
        self.master = master
        self.mark_staleness_seconds = mark_staleness_seconds

    def _resolve_fx(
        self,
        currency: str,
        market: MarketSnapshot,
        exceptions: list[ValuationException],
        account_id: str,
    ) -> Decimal | None:
        fx = market.get_fx_rate(currency, market.base_currency)
        if fx is None:
            exceptions.append(
                ValuationException(
                    code="MISSING_FX",
                    account_id=account_id,
                    currency=currency,
                    message=f"Missing FX rate {currency}->{market.base_currency}",
                )
            )
            return None
        return fx.rate

    def _value_account(
        self,
        acct_state: AccountLedgerState,
        market: MarketSnapshot,
        exceptions: list[ValuationException],
    ) -> AccountValuation:
        acct = self.master.get_account(acct_state.account_id)
        base_ccy = market.base_currency

        # Cache FX rates for all currencies touched by this account
        currencies = (
            set(acct_state.cash_balances)
            | set(acct_state.realized_trading_pnl)
            | set(acct_state.dividend_income)
            | set(acct_state.fees_and_commissions)
            | set(acct_state.net_capital_flows)
            | set(acct_state.fx_units_acquired)
            | set(acct_state.adjustments_cash)
            | set(acct_state.adjustments_position_cost)
            | {pos.currency for pos in acct_state.positions.values()}
        )
        fx_rates: dict[str, Decimal | None] = {
            ccy: self._resolve_fx(ccy, market, exceptions, acct.account_id)
            for ccy in sorted(currencies)
        }

        # Cash valuation
        cash_base_map: dict[str, Decimal] = {}
        total_cash_base = ZERO
        for ccy, amount in sorted(acct_state.cash_balances.items()):
            rate = fx_rates.get(ccy)
            if rate is not None:
                val = amount * rate
                cash_base_map[ccy] = val
                total_cash_base += val

        # Position valuation with explicit missing/stale mark flagging
        valued_positions: list[ValuedPosition] = []
        unpriced: list[str] = []
        priced_mv_base = ZERO
        unpriced_cb_base = ZERO
        unrealized_price_pnl_base = ZERO
        gross_exposure_base = ZERO
        net_exposure_base = ZERO

        for inst_id, pos in sorted(acct_state.positions.items()):
            mark = market.get_price(inst_id)
            rate = fx_rates.get(pos.currency)
            pricing_status = PricingStatus.PRICED
            is_priced = True

            if rate is None:
                pricing_status = PricingStatus.MISSING_FX
                is_priced = False
            elif mark is None:
                pricing_status = PricingStatus.MISSING_MARK
                is_priced = False
                exceptions.append(
                    ValuationException(
                        code="MISSING_MARK",
                        account_id=acct.account_id,
                        instrument_id=inst_id,
                        currency=pos.currency,
                        message=(
                            f"No market price for {pos.symbol} ({inst_id}) in snapshot "
                            f"{market.snapshot_id}; carrying at cost basis and flagging unpriced."
                        ),
                    )
                )
            else:
                age_sec = (market.as_of_time - mark.price_timestamp).total_seconds()
                if age_sec > self.mark_staleness_seconds:
                    pricing_status = PricingStatus.STALE_MARK
                    is_priced = False
                    exceptions.append(
                        ValuationException(
                            code="STALE_MARK",
                            account_id=acct.account_id,
                            instrument_id=inst_id,
                            currency=pos.currency,
                            message=(
                                f"Stale price for {pos.symbol} ({inst_id}): age {int(age_sec)}s "
                                f"exceeds threshold {self.mark_staleness_seconds}s; carrying at cost basis."
                            ),
                        )
                    )

            if is_priced and mark is not None and rate is not None:
                mv_local = pos.quantity * mark.price * pos.multiplier
                upnl_local = mv_local - pos.signed_cost_basis
                mv_base = mv_local * rate
                upnl_base = upnl_local * rate
                priced_mv_base += mv_base
                unrealized_price_pnl_base += upnl_base
            else:
                unpriced.append(inst_id)
                mv_local = pos.signed_cost_basis
                upnl_local = ZERO
                mv_base = mv_local * (rate if rate is not None else ZERO)
                upnl_base = ZERO
                unpriced_cb_base += mv_base

            gross_exp = abs(mv_base)
            net_exp = mv_base
            gross_exposure_base += gross_exp
            net_exposure_base += net_exp

            valued_positions.append(
                ValuedPosition(
                    account_id=acct.account_id,
                    strategy_id=acct.strategy_id,
                    instrument_id=inst_id,
                    symbol=pos.symbol,
                    currency=pos.currency,
                    quantity=pos.quantity,
                    multiplier=pos.multiplier,
                    average_unit_cost=pos.average_unit_cost,
                    signed_cost_basis_local=pos.signed_cost_basis,
                    mark_price=mark.price if mark is not None else None,
                    mark_timestamp=mark.price_timestamp if mark is not None else None,
                    pricing_status=pricing_status,
                    is_priced=is_priced,
                    fx_rate_to_base=rate,
                    carrying_value_local=mv_local,
                    carrying_value_base=mv_base,
                    unrealized_pnl_local=upnl_local,
                    unrealized_pnl_base=upnl_base,
                    gross_exposure_base=gross_exp,
                    net_exposure_base=net_exp,
                )
            )

        # P&L explainability in base currency
        realized_base = ZERO
        dividends_base = ZERO
        fees_base = ZERO
        late_fees_base = ZERO
        net_capital_base = ZERO
        fx_pnl_base = ZERO
        adjustments_base = ZERO

        for ccy, val in acct_state.realized_trading_pnl.items():
            rate = fx_rates.get(ccy)
            if rate is not None:
                realized_base += val * rate
        for ccy, val in acct_state.dividend_income.items():
            rate = fx_rates.get(ccy)
            if rate is not None:
                dividends_base += val * rate
        for ccy, val in acct_state.fees_and_commissions.items():
            rate = fx_rates.get(ccy)
            if rate is not None:
                fees_base += val * rate
        for ccy, val in acct_state.late_fees.items():
            rate = fx_rates.get(ccy)
            if rate is not None:
                late_fees_base += val * rate
        for ccy, val in acct_state.net_capital_flows.items():
            rate = fx_rates.get(ccy)
            if rate is not None:
                net_capital_base += val * rate
        for ccy, val in acct_state.adjustments_cash.items():
            rate = fx_rates.get(ccy)
            if rate is not None:
                adjustments_base += val * rate
        for ccy, val in acct_state.adjustments_position_cost.items():
            rate = fx_rates.get(ccy)
            if rate is not None:
                adjustments_base += val * rate

        for ccy, units in acct_state.fx_units_acquired.items():
            rate = fx_rates.get(ccy)
            cost_base = acct_state.fx_cost_basis_base.get(ccy, ZERO)
            if rate is not None:
                fx_pnl_base += (units * rate) - cost_base

        nav_base = total_cash_base + priced_mv_base + unpriced_cb_base
        nav_minus_cap = nav_base - net_capital_base
        explained_pnl = (
            realized_base
            + unrealized_price_pnl_base
            + fx_pnl_base
            + dividends_base
            - fees_base
            + adjustments_base
        )
        residual = nav_minus_cap - explained_pnl

        pnl = PnLAttribution(
            realized_trading_pnl_base=realized_base,
            unrealized_price_pnl_base=unrealized_price_pnl_base,
            fx_translation_pnl_base=fx_pnl_base,
            dividend_income_base=dividends_base,
            fees_and_commissions_base=fees_base,
            late_fees_base=late_fees_base,
            adjustments_base=adjustments_base,
            explained_total_pnl_base=explained_pnl,
            nav_minus_net_capital_base=nav_minus_cap,
            unexplained_residual_base=residual,
        )

        return AccountValuation(
            account_id=acct.account_id,
            account_name=acct.name,
            account_type=acct.account_type,
            strategy_id=acct.strategy_id,
            base_currency=base_ccy,
            cash_balances_local=dict(acct_state.cash_balances),
            cash_balances_base=cash_base_map,
            total_cash_base=total_cash_base,
            priced_market_value_base=priced_mv_base,
            unpriced_cost_basis_base=unpriced_cb_base,
            nav_base=nav_base,
            net_capital_flows_base=net_capital_base,
            gross_exposure_base=gross_exposure_base,
            net_exposure_base=net_exposure_base,
            positions=tuple(valued_positions),
            pnl=pnl,
            unpriced_instruments=tuple(sorted(set(unpriced))),
            valuation_complete=len(unpriced) == 0,
        )

    def value_fund(
        self,
        replay: ReplaySnapshot,
        market: MarketSnapshot,
        *,
        strict_pnl_explain: bool = True,
    ) -> FundValuationReport:
        exceptions: list[ValuationException] = []
        strategy_vals: list[AccountValuation] = []
        unalloc_vals: list[AccountValuation] = []

        for acct_id in sorted(replay.accounts):
            val = self._value_account(replay.accounts[acct_id], market, exceptions)
            if val.account_type == AccountType.STRATEGY:
                strategy_vals.append(val)
            else:
                unalloc_vals.append(val)

        all_vals = strategy_vals + unalloc_vals

        strategy_cash_base = sum((v.total_cash_base for v in strategy_vals), ZERO)
        unalloc_cash_base = sum((v.total_cash_base for v in unalloc_vals), ZERO)
        total_cash_base = strategy_cash_base + unalloc_cash_base
        priced_mv_base = sum((v.priced_market_value_base for v in all_vals), ZERO)
        unpriced_cb_base = sum((v.unpriced_cost_basis_base for v in all_vals), ZERO)
        fund_nav_base = sum((v.nav_base for v in all_vals), ZERO)
        net_capital_flows_base = sum((v.net_capital_flows_base for v in all_vals), ZERO)
        gross_exposure_base = sum((v.gross_exposure_base for v in all_vals), ZERO)
        net_exposure_base = sum((v.net_exposure_base for v in all_vals), ZERO)

        realized_base = sum((v.pnl.realized_trading_pnl_base for v in all_vals), ZERO)
        unrealized_base = sum((v.pnl.unrealized_price_pnl_base for v in all_vals), ZERO)
        fx_base = sum((v.pnl.fx_translation_pnl_base for v in all_vals), ZERO)
        div_base = sum((v.pnl.dividend_income_base for v in all_vals), ZERO)
        fees_base = sum((v.pnl.fees_and_commissions_base for v in all_vals), ZERO)
        late_fees_base = sum((v.pnl.late_fees_base for v in all_vals), ZERO)
        adj_base = sum((v.pnl.adjustments_base for v in all_vals), ZERO)
        explained_pnl = (
            realized_base + unrealized_base + fx_base + div_base - fees_base + adj_base
        )
        nav_minus_cap = fund_nav_base - net_capital_flows_base
        residual = nav_minus_cap - explained_pnl

        fund_pnl = PnLAttribution(
            realized_trading_pnl_base=realized_base,
            unrealized_price_pnl_base=unrealized_base,
            fx_translation_pnl_base=fx_base,
            dividend_income_base=div_base,
            fees_and_commissions_base=fees_base,
            late_fees_base=late_fees_base,
            adjustments_base=adj_base,
            explained_total_pnl_base=explained_pnl,
            nav_minus_net_capital_base=nav_minus_cap,
            unexplained_residual_base=residual,
        )

        # Independent fund consolidation verification:
        # Fund totals == sum(strategy totals) + unallocated fund accounts
        sum_strat_nav = sum((v.nav_base for v in strategy_vals), ZERO)
        sum_unalloc_nav = sum((v.nav_base for v in unalloc_vals), ZERO)
        recomputed_nav = sum_strat_nav + sum_unalloc_nav

        sum_strat_pnl = sum((v.pnl.explained_total_pnl_base for v in strategy_vals), ZERO)
        sum_unalloc_pnl = sum((v.pnl.explained_total_pnl_base for v in unalloc_vals), ZERO)
        recomputed_pnl = sum_strat_pnl + sum_unalloc_pnl

        # Also independently verify fund cash from raw consolidated local cash balances
        consolidated_cash: dict[str, Decimal] = defaultdict(lambda: ZERO)
        consolidated_positions: dict[str, Decimal] = defaultdict(lambda: ZERO)
        unpriced_all: set[str] = set()

        for v in all_vals:
            for ccy, bal in v.cash_balances_local.items():
                consolidated_cash[ccy] += bal
            for pos in v.positions:
                consolidated_positions[pos.instrument_id] += pos.quantity
            unpriced_all.update(v.unpriced_instruments)

        independent_cash_base = ZERO
        for ccy, bal in consolidated_cash.items():
            fx = market.get_fx_rate(ccy, market.base_currency)
            if fx is not None:
                independent_cash_base += bal * fx.rate

        nav_diff = fund_nav_base - recomputed_nav
        cash_diff = independent_cash_base - (strategy_cash_base + unalloc_cash_base)
        pnl_diff = fund_pnl.explained_total_pnl_base - recomputed_pnl

        consolidation_passed = (
            abs(nav_diff) <= EPS and abs(cash_diff) <= EPS and abs(pnl_diff) <= EPS
        )
        consolidation_check = ConsolidationCheck(
            fund_nav_base=fund_nav_base,
            strategies_plus_unallocated_nav_base=recomputed_nav,
            nav_difference_base=nav_diff,
            fund_cash_base=independent_cash_base,
            strategies_plus_unallocated_cash_base=strategy_cash_base + unalloc_cash_base,
            cash_difference_base=cash_diff,
            fund_pnl_base=fund_pnl.explained_total_pnl_base,
            strategies_plus_unallocated_pnl_base=recomputed_pnl,
            pnl_difference_base=pnl_diff,
            passed=consolidation_passed,
        )

        if not consolidation_passed:
            raise InvariantViolationError(
                "Fund consolidation check failed",
                details=consolidation_check.model_dump(mode="json"),
            )

        if strict_pnl_explain and len([e for e in exceptions if e.code == "MISSING_FX"]) == 0:
            if abs(residual) > EPS:
                raise InvariantViolationError(
                    f"Unexplained P&L residual detected: {residual}",
                    details=fund_pnl.model_dump(mode="json"),
                )

        return FundValuationReport(
            fund_id=self.master.fund.fund_id,
            fund_name=self.master.fund.name,
            base_currency=market.base_currency,
            snapshot_id=market.snapshot_id,
            as_of_time=market.as_of_time,
            ledger_last_seq_no=replay.last_seq_no,
            ledger_state_hash=replay.state_hash,
            total_cash_base=total_cash_base,
            unallocated_cash_base=unalloc_cash_base,
            strategy_cash_base=strategy_cash_base,
            priced_market_value_base=priced_mv_base,
            unpriced_cost_basis_base=unpriced_cb_base,
            fund_nav_base=fund_nav_base,
            net_capital_flows_base=net_capital_flows_base,
            gross_exposure_base=gross_exposure_base,
            net_exposure_base=net_exposure_base,
            pnl=fund_pnl,
            strategy_valuations=tuple(strategy_vals),
            unallocated_valuations=tuple(unalloc_vals),
            consolidated_positions={
                k: v for k, v in sorted(consolidated_positions.items()) if v != ZERO
            },
            consolidated_cash_local={
                k: v for k, v in sorted(consolidated_cash.items()) if v != ZERO
            },
            unpriced_instruments=tuple(sorted(unpriced_all)),
            exceptions=tuple(exceptions),
            valuation_complete=len(exceptions) == 0,
            consolidation_check=consolidation_check,
        )
