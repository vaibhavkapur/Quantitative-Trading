"""Deterministic event-sourced ledger replay engine producing balanced state."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import Field

from pms.domain.enums import AdjustmentType, CashDirection, CorporateActionType
from pms.domain.master import MasterRegistry
from pms.domain.models import (
    AdjustmentEvent,
    CashTransferEvent,
    CorporateActionEvent,
    FeeEvent,
    FrozenModel,
    FxConversionEvent,
    LedgerEvent,
    StoredEvent,
    StrategyAllocationEvent,
    TradeFillEvent,
)
from pms.ledger.journal import JournalEntry, JournalLine, verify_trial_balance
from pms.ledger.lots import LotBook, OpenLot, TradeLotOutcome
from pms.ledger.store import EventStore

ZERO = Decimal("0")


class PositionState(FrozenModel):
    account_id: str
    instrument_id: str
    symbol: str
    currency: str
    multiplier: Decimal
    quantity: Decimal
    average_unit_cost: Decimal
    signed_cost_basis: Decimal
    lots: tuple[OpenLot, ...] = Field(default_factory=tuple)


class AccountLedgerState(FrozenModel):
    account_id: str
    cash_balances: dict[str, Decimal] = Field(default_factory=dict)
    positions: dict[str, PositionState] = Field(default_factory=dict)
    realized_trading_pnl: dict[str, Decimal] = Field(default_factory=dict)
    dividend_income: dict[str, Decimal] = Field(default_factory=dict)
    fees_and_commissions: dict[str, Decimal] = Field(default_factory=dict)
    late_fees: dict[str, Decimal] = Field(default_factory=dict)
    net_capital_flows: dict[str, Decimal] = Field(default_factory=dict)
    fx_units_acquired: dict[str, Decimal] = Field(default_factory=dict)
    fx_cost_basis_base: dict[str, Decimal] = Field(default_factory=dict)
    adjustments_cash: dict[str, Decimal] = Field(default_factory=dict)
    adjustments_position_cost: dict[str, Decimal] = Field(default_factory=dict)


class ReplaySnapshot(FrozenModel):
    fund_id: str
    base_currency: str
    as_of_time: datetime | None
    event_count: int
    last_seq_no: int
    last_event_hash: str
    state_hash: str
    accounts: dict[str, AccountLedgerState]
    trial_balance: dict[str, dict[str, Decimal]]
    journal_entry_count: int
    order_fill_tracker: dict[str, Decimal] = Field(default_factory=dict)


class _MutableAccountBook:
    def __init__(self, account_id: str) -> None:
        self.account_id = account_id
        self.cash: dict[str, Decimal] = defaultdict(lambda: ZERO)
        self.lot_books: dict[str, LotBook] = {}
        self.realized_pnl: dict[str, Decimal] = defaultdict(lambda: ZERO)
        self.dividends: dict[str, Decimal] = defaultdict(lambda: ZERO)
        self.fees: dict[str, Decimal] = defaultdict(lambda: ZERO)
        self.late_fees: dict[str, Decimal] = defaultdict(lambda: ZERO)
        self.net_capital_flows: dict[str, Decimal] = defaultdict(lambda: ZERO)
        self.fx_units_acquired: dict[str, Decimal] = defaultdict(lambda: ZERO)
        self.fx_cost_basis_base: dict[str, Decimal] = defaultdict(lambda: ZERO)
        self.adjustments_cash: dict[str, Decimal] = defaultdict(lambda: ZERO)
        self.adjustments_position_cost: dict[str, Decimal] = defaultdict(lambda: ZERO)


def _clean_decimal_map(mapping: dict[str, Decimal]) -> dict[str, Decimal]:
    return {k: mapping[k] for k in sorted(mapping) if mapping[k] != ZERO}


class LedgerEngine:
    """Validates, appends, and deterministically replays immutable ledger events."""

    def __init__(self, master: MasterRegistry, store: EventStore) -> None:
        self.master = master
        self.store = store

    def record_event(self, event: LedgerEvent) -> StoredEvent:
        self.master.validate_event(event)
        return self.store.append(event)

    def replay(
        self,
        *,
        as_of: datetime | None = None,
        include_journals: bool = False,
    ) -> tuple[ReplaySnapshot, list[JournalEntry]]:
        events = self.store.list_events(as_of=as_of)
        books: dict[str, _MutableAccountBook] = {
            acct.account_id: _MutableAccountBook(acct.account_id)
            for acct in self.master.list_accounts()
        }
        journals: list[JournalEntry] = []
        order_fill_tracker: dict[str, Decimal] = defaultdict(lambda: ZERO)

        for stored in events:
            entry = self._apply_stored_event(stored, books, order_fill_tracker)
            entry.assert_balanced()
            journals.append(entry)

        trial_balance = verify_trial_balance(journals)

        account_states: dict[str, AccountLedgerState] = {}
        for account_id in sorted(books):
            book = books[account_id]
            positions: dict[str, PositionState] = {}
            for inst_id in sorted(book.lot_books):
                lb = book.lot_books[inst_id]
                if lb.net_quantity != ZERO:
                    inst = self.master.get_instrument(inst_id)
                    positions[inst_id] = PositionState(
                        account_id=account_id,
                        instrument_id=inst_id,
                        symbol=inst.symbol,
                        currency=inst.currency,
                        multiplier=inst.multiplier,
                        quantity=lb.net_quantity,
                        average_unit_cost=lb.average_unit_cost,
                        signed_cost_basis=lb.signed_cost_basis,
                        lots=tuple(lb.lots),
                    )
            account_states[account_id] = AccountLedgerState(
                account_id=account_id,
                cash_balances=_clean_decimal_map(book.cash),
                positions=positions,
                realized_trading_pnl=_clean_decimal_map(book.realized_pnl),
                dividend_income=_clean_decimal_map(book.dividends),
                fees_and_commissions=_clean_decimal_map(book.fees),
                late_fees=_clean_decimal_map(book.late_fees),
                net_capital_flows=_clean_decimal_map(book.net_capital_flows),
                fx_units_acquired=_clean_decimal_map(book.fx_units_acquired),
                fx_cost_basis_base=_clean_decimal_map(book.fx_cost_basis_base),
                adjustments_cash=_clean_decimal_map(book.adjustments_cash),
                adjustments_position_cost=_clean_decimal_map(book.adjustments_position_cost),
            )

        last_seq = events[-1].seq_no if events else 0
        last_hash = events[-1].event_hash if events else ("0" * 64)

        canonical_state: dict[str, Any] = {
            "fund_id": self.master.fund.fund_id,
            "last_seq_no": last_seq,
            "last_event_hash": last_hash,
            "accounts": {
                aid: state.model_dump(mode="json") for aid, state in account_states.items()
            },
            "trial_balance": {
                ccy: {"debits": str(v["debits"]), "credits": str(v["credits"])}
                for ccy, v in sorted(trial_balance.items())
            },
        }
        state_hash = hashlib.sha256(
            json.dumps(canonical_state, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

        snapshot = ReplaySnapshot(
            fund_id=self.master.fund.fund_id,
            base_currency=self.master.fund.base_currency,
            as_of_time=as_of,
            event_count=len(events),
            last_seq_no=last_seq,
            last_event_hash=last_hash,
            state_hash=state_hash,
            accounts=account_states,
            trial_balance=trial_balance,
            journal_entry_count=len(journals),
            order_fill_tracker=dict(sorted(order_fill_tracker.items())),
        )
        return snapshot, (journals if include_journals else [])

    def _get_or_create_lot_book(
        self, book: _MutableAccountBook, instrument_id: str
    ) -> LotBook:
        if instrument_id not in book.lot_books:
            acct = self.master.get_account(book.account_id)
            inst = self.master.get_instrument(instrument_id)
            book.lot_books[instrument_id] = LotBook(
                account_id=acct.account_id,
                instrument_id=inst.instrument_id,
                currency=inst.currency,
                multiplier=inst.multiplier,
                lot_method=acct.lot_method,
            )
        return book.lot_books[instrument_id]

    def _apply_stored_event(
        self,
        stored: StoredEvent,
        books: dict[str, _MutableAccountBook],
        order_fill_tracker: dict[str, Decimal],
    ) -> JournalEntry:
        event = stored.event
        lines: list[JournalLine] = []

        if isinstance(event, CashTransferEvent):
            book = books[event.account_id]
            if event.direction == CashDirection.DEPOSIT:
                book.cash[event.currency] += event.amount
                book.net_capital_flows[event.currency] += event.amount
                lines.extend(
                    [
                        JournalLine(
                            account_id=event.account_id,
                            gl_account="ASSET:CASH",
                            currency=event.currency,
                            debit=event.amount,
                            memo=event.description or "Cash deposit",
                        ),
                        JournalLine(
                            account_id=event.account_id,
                            gl_account="EQUITY:CAPITAL_CONTRIBUTED",
                            currency=event.currency,
                            credit=event.amount,
                            memo=event.description or "Cash deposit",
                        ),
                    ]
                )
            else:
                book.cash[event.currency] -= event.amount
                book.net_capital_flows[event.currency] -= event.amount
                lines.extend(
                    [
                        JournalLine(
                            account_id=event.account_id,
                            gl_account="EQUITY:CAPITAL_CONTRIBUTED",
                            currency=event.currency,
                            debit=event.amount,
                            memo=event.description or "Cash withdrawal",
                        ),
                        JournalLine(
                            account_id=event.account_id,
                            gl_account="ASSET:CASH",
                            currency=event.currency,
                            credit=event.amount,
                            memo=event.description or "Cash withdrawal",
                        ),
                    ]
                )

        elif isinstance(event, StrategyAllocationEvent):
            from_book = books[event.from_account_id]
            to_book = books[event.to_account_id]
            from_book.cash[event.currency] -= event.amount
            from_book.net_capital_flows[event.currency] -= event.amount
            to_book.cash[event.currency] += event.amount
            to_book.net_capital_flows[event.currency] += event.amount
            lines.extend(
                [
                    JournalLine(
                        account_id=event.to_account_id,
                        gl_account="ASSET:CASH",
                        currency=event.currency,
                        debit=event.amount,
                        memo=event.allocation_reason or "Strategy allocation in",
                    ),
                    JournalLine(
                        account_id=event.from_account_id,
                        gl_account="ASSET:CASH",
                        currency=event.currency,
                        credit=event.amount,
                        memo=event.allocation_reason or "Strategy allocation out",
                    ),
                ]
            )

        elif isinstance(event, TradeFillEvent):
            book = books[event.account_id]
            lb = self._get_or_create_lot_book(book, event.instrument_id)
            outcome = lb.apply_trade(
                seq_no=stored.seq_no,
                event_id=event.event_id,
                event_time=event.event_time,
                side=event.side,
                quantity=event.quantity,
                price=event.price,
            )
            order_fill_tracker[event.order_id] += event.quantity
            lines.extend(
                self._trade_outcome_journal_lines(
                    account_id=event.account_id,
                    instrument_id=event.instrument_id,
                    currency=event.currency,
                    outcome=outcome,
                    execution_id=event.execution_id,
                )
            )
            # Update cash balances
            cash_delta = (
                outcome.closed_long_proceeds
                + outcome.opened_short_proceeds
                - outcome.opened_long_cost
                - outcome.closed_short_cover_cost
                - event.fee
            )
            book.cash[event.currency] += cash_delta
            book.realized_pnl[event.currency] += outcome.total_realized_pnl

            if event.fee > ZERO:
                book.fees[event.currency] += event.fee
                lines.extend(
                    [
                        JournalLine(
                            account_id=event.account_id,
                            gl_account="EXPENSE:TRADING_FEE",
                            currency=event.currency,
                            debit=event.fee,
                            instrument_id=event.instrument_id,
                            memo=f"Execution fee {event.execution_id}",
                        ),
                        JournalLine(
                            account_id=event.account_id,
                            gl_account="ASSET:CASH",
                            currency=event.currency,
                            credit=event.fee,
                            instrument_id=event.instrument_id,
                            memo=f"Execution fee {event.execution_id}",
                        ),
                    ]
                )

        elif isinstance(event, FxConversionEvent):
            book = books[event.account_id]
            base_ccy = self.master.fund.base_currency
            book.cash[event.sell_currency] -= event.sell_amount
            book.cash[event.buy_currency] += event.buy_amount

            if event.sell_currency == base_ccy and event.buy_currency != base_ccy:
                book.fx_units_acquired[event.buy_currency] += event.buy_amount
                book.fx_cost_basis_base[event.buy_currency] += event.sell_amount
            elif event.buy_currency == base_ccy and event.sell_currency != base_ccy:
                book.fx_units_acquired[event.sell_currency] -= event.sell_amount
                book.fx_cost_basis_base[event.sell_currency] -= event.buy_amount

            lines.extend(
                [
                    JournalLine(
                        account_id=event.account_id,
                        gl_account="ASSET:FX_CLEARING",
                        currency=event.sell_currency,
                        debit=event.sell_amount,
                        memo=f"FX sell {event.sell_currency}->{event.buy_currency} @ {event.fx_rate}",
                    ),
                    JournalLine(
                        account_id=event.account_id,
                        gl_account="ASSET:CASH",
                        currency=event.sell_currency,
                        credit=event.sell_amount,
                        memo=f"FX sell {event.sell_currency}",
                    ),
                    JournalLine(
                        account_id=event.account_id,
                        gl_account="ASSET:CASH",
                        currency=event.buy_currency,
                        debit=event.buy_amount,
                        memo=f"FX buy {event.buy_currency}",
                    ),
                    JournalLine(
                        account_id=event.account_id,
                        gl_account="ASSET:FX_CLEARING",
                        currency=event.buy_currency,
                        credit=event.buy_amount,
                        memo=f"FX buy {event.buy_currency} @ {event.fx_rate}",
                    ),
                ]
            )
            if event.fee_amount > ZERO:
                book.cash[event.fee_currency] -= event.fee_amount
                book.fees[event.fee_currency] += event.fee_amount
                lines.extend(
                    [
                        JournalLine(
                            account_id=event.account_id,
                            gl_account="EXPENSE:TRADING_FEE",
                            currency=event.fee_currency,
                            debit=event.fee_amount,
                            memo="FX conversion fee",
                        ),
                        JournalLine(
                            account_id=event.account_id,
                            gl_account="ASSET:CASH",
                            currency=event.fee_currency,
                            credit=event.fee_amount,
                            memo="FX conversion fee",
                        ),
                    ]
                )

        elif isinstance(event, FeeEvent):
            book = books[event.account_id]
            book.cash[event.currency] -= event.amount
            book.fees[event.currency] += event.amount
            if event.is_late_posting:
                book.late_fees[event.currency] += event.amount
            lines.extend(
                [
                    JournalLine(
                        account_id=event.account_id,
                        gl_account=f"EXPENSE:{event.fee_category}",
                        currency=event.currency,
                        debit=event.amount,
                        instrument_id=event.instrument_id,
                        memo=event.description or event.fee_category,
                    ),
                    JournalLine(
                        account_id=event.account_id,
                        gl_account="ASSET:CASH",
                        currency=event.currency,
                        credit=event.amount,
                        instrument_id=event.instrument_id,
                        memo=event.description or event.fee_category,
                    ),
                ]
            )

        elif isinstance(event, CorporateActionEvent):
            book = books[event.account_id]
            lb = self._get_or_create_lot_book(book, event.instrument_id)
            if event.action_type == CorporateActionType.SPLIT:
                lb.apply_split(event.ratio_numerator, event.ratio_denominator)
                # Split preserves cost basis exactly; zero monetary journal movement.
                lines.append(
                    JournalLine(
                        account_id=event.account_id,
                        gl_account="MEMO:CORPORATE_ACTION_SPLIT",
                        currency=event.currency,
                        debit=ZERO,
                        credit=ZERO,
                        instrument_id=event.instrument_id,
                        memo=f"Split {event.ratio_numerator}:{event.ratio_denominator}",
                    )
                )
            elif event.action_type == CorporateActionType.CASH_DIVIDEND:
                held_qty = lb.net_quantity
                gross_div = held_qty * event.cash_per_share
                if gross_div > ZERO:
                    tax = gross_div * event.withholding_tax_rate
                    net_div = gross_div - tax
                    book.cash[event.currency] += net_div
                    book.dividends[event.currency] += gross_div
                    lines.extend(
                        [
                            JournalLine(
                                account_id=event.account_id,
                                gl_account="ASSET:CASH",
                                currency=event.currency,
                                debit=net_div,
                                instrument_id=event.instrument_id,
                                memo=event.description or "Cash dividend",
                            ),
                            JournalLine(
                                account_id=event.account_id,
                                gl_account="INCOME:DIVIDEND",
                                currency=event.currency,
                                credit=gross_div,
                                instrument_id=event.instrument_id,
                                memo=event.description or "Cash dividend",
                            ),
                        ]
                    )
                    if tax > ZERO:
                        book.fees[event.currency] += tax
                        lines.append(
                            JournalLine(
                                account_id=event.account_id,
                                gl_account="EXPENSE:WITHHOLDING_TAX",
                                currency=event.currency,
                                debit=tax,
                                instrument_id=event.instrument_id,
                                memo="Dividend withholding tax",
                            )
                        )
                elif gross_div < ZERO:
                    # Short position pays dividend in lieu
                    payment = abs(gross_div)
                    book.cash[event.currency] -= payment
                    book.dividends[event.currency] -= payment
                    lines.extend(
                        [
                            JournalLine(
                                account_id=event.account_id,
                                gl_account="INCOME:DIVIDEND",
                                currency=event.currency,
                                debit=payment,
                                instrument_id=event.instrument_id,
                                memo="Short dividend in lieu",
                            ),
                            JournalLine(
                                account_id=event.account_id,
                                gl_account="ASSET:CASH",
                                currency=event.currency,
                                credit=payment,
                                instrument_id=event.instrument_id,
                                memo="Short dividend in lieu",
                            ),
                        ]
                    )

        elif isinstance(event, AdjustmentEvent):
            book = books[event.account_id]
            if event.adjustment_type == AdjustmentType.CASH:
                book.cash[event.currency] += event.cash_delta
                book.adjustments_cash[event.currency] += event.cash_delta
                if event.cash_delta > ZERO:
                    lines.extend(
                        [
                            JournalLine(
                                account_id=event.account_id,
                                gl_account="ASSET:CASH",
                                currency=event.currency,
                                debit=event.cash_delta,
                                memo=f"Recon adjustment {event.reconciliation_break_id}",
                            ),
                            JournalLine(
                                account_id=event.account_id,
                                gl_account="EQUITY:RECONCILIATION_ADJUSTMENT",
                                currency=event.currency,
                                credit=event.cash_delta,
                                memo=f"Approved by {event.approved_by}: {event.reason}",
                            ),
                        ]
                    )
                else:
                    amt = abs(event.cash_delta)
                    lines.extend(
                        [
                            JournalLine(
                                account_id=event.account_id,
                                gl_account="EQUITY:RECONCILIATION_ADJUSTMENT",
                                currency=event.currency,
                                debit=amt,
                                memo=f"Approved by {event.approved_by}: {event.reason}",
                            ),
                            JournalLine(
                                account_id=event.account_id,
                                gl_account="ASSET:CASH",
                                currency=event.currency,
                                credit=amt,
                                memo=f"Recon adjustment {event.reconciliation_break_id}",
                            ),
                        ]
                    )
            elif event.adjustment_type == AdjustmentType.POSITION:
                assert event.instrument_id is not None
                lb = self._get_or_create_lot_book(book, event.instrument_id)
                prior_signed_basis = lb.signed_cost_basis
                outcome = lb.apply_position_adjustment(
                    seq_no=stored.seq_no,
                    event_id=event.event_id,
                    event_time=event.event_time,
                    quantity_delta=event.quantity_delta,
                    unit_cost=event.unit_cost,
                )
                new_signed_basis = lb.signed_cost_basis
                basis_delta = new_signed_basis - prior_signed_basis
                book.adjustments_position_cost[event.currency] += basis_delta
                if basis_delta > ZERO:
                    lines.extend(
                        [
                            JournalLine(
                                account_id=event.account_id,
                                gl_account="ASSET:INVESTMENT_COST",
                                currency=event.currency,
                                debit=basis_delta,
                                instrument_id=event.instrument_id,
                                memo=f"Position recon adjustment {event.reconciliation_break_id}",
                            ),
                            JournalLine(
                                account_id=event.account_id,
                                gl_account="EQUITY:RECONCILIATION_ADJUSTMENT",
                                currency=event.currency,
                                credit=basis_delta,
                                instrument_id=event.instrument_id,
                                memo=f"Approved by {event.approved_by}: {event.reason}",
                            ),
                        ]
                    )
                elif basis_delta < ZERO:
                    amt = abs(basis_delta)
                    lines.extend(
                        [
                            JournalLine(
                                account_id=event.account_id,
                                gl_account="EQUITY:RECONCILIATION_ADJUSTMENT",
                                currency=event.currency,
                                debit=amt,
                                instrument_id=event.instrument_id,
                                memo=f"Approved by {event.approved_by}: {event.reason}",
                            ),
                            JournalLine(
                                account_id=event.account_id,
                                gl_account="ASSET:INVESTMENT_COST",
                                currency=event.currency,
                                credit=amt,
                                instrument_id=event.instrument_id,
                                memo=f"Position recon adjustment {event.reconciliation_break_id}",
                            ),
                        ]
                    )
                else:
                    lines.append(
                        JournalLine(
                            account_id=event.account_id,
                            gl_account="MEMO:POSITION_ADJUSTMENT_ZERO_COST",
                            currency=event.currency,
                            debit=ZERO,
                            credit=ZERO,
                            instrument_id=event.instrument_id,
                            memo=f"Zero-cost position adjustment {event.reconciliation_break_id}",
                        )
                    )
                # Note: realized_pnl from a position adjustment is captured via basis_delta against EQUITY:RECONCILIATION_ADJUSTMENT
                _ = outcome

        return JournalEntry(
            event_id=stored.event_id,
            seq_no=stored.seq_no,
            event_time=stored.event_time,
            lines=tuple(lines),
        )

    @staticmethod
    def _trade_outcome_journal_lines(
        *,
        account_id: str,
        instrument_id: str,
        currency: str,
        outcome: TradeLotOutcome,
        execution_id: str,
    ) -> list[JournalLine]:
        lines: list[JournalLine] = []
        if outcome.opened_long_qty > ZERO:
            lines.extend(
                [
                    JournalLine(
                        account_id=account_id,
                        gl_account="ASSET:INVESTMENT_COST",
                        currency=currency,
                        debit=outcome.opened_long_cost,
                        instrument_id=instrument_id,
                        memo=f"Open long {execution_id}",
                    ),
                    JournalLine(
                        account_id=account_id,
                        gl_account="ASSET:CASH",
                        currency=currency,
                        credit=outcome.opened_long_cost,
                        instrument_id=instrument_id,
                        memo=f"Open long {execution_id}",
                    ),
                ]
            )
        if outcome.closed_long_qty > ZERO:
            lines.extend(
                [
                    JournalLine(
                        account_id=account_id,
                        gl_account="ASSET:CASH",
                        currency=currency,
                        debit=outcome.closed_long_proceeds,
                        instrument_id=instrument_id,
                        memo=f"Close long proceeds {execution_id}",
                    ),
                    JournalLine(
                        account_id=account_id,
                        gl_account="ASSET:INVESTMENT_COST",
                        currency=currency,
                        credit=outcome.closed_long_cost,
                        instrument_id=instrument_id,
                        memo=f"Close long cost {execution_id}",
                    ),
                ]
            )
            if outcome.closed_long_pnl > ZERO:
                lines.append(
                    JournalLine(
                        account_id=account_id,
                        gl_account="INCOME:REALIZED_TRADING_PNL",
                        currency=currency,
                        credit=outcome.closed_long_pnl,
                        instrument_id=instrument_id,
                        memo=f"Realized long gain {execution_id}",
                    )
                )
            elif outcome.closed_long_pnl < ZERO:
                lines.append(
                    JournalLine(
                        account_id=account_id,
                        gl_account="INCOME:REALIZED_TRADING_PNL",
                        currency=currency,
                        debit=abs(outcome.closed_long_pnl),
                        instrument_id=instrument_id,
                        memo=f"Realized long loss {execution_id}",
                    )
                )

        if outcome.opened_short_qty > ZERO:
            lines.extend(
                [
                    JournalLine(
                        account_id=account_id,
                        gl_account="ASSET:CASH",
                        currency=currency,
                        debit=outcome.opened_short_proceeds,
                        instrument_id=instrument_id,
                        memo=f"Open short proceeds {execution_id}",
                    ),
                    JournalLine(
                        account_id=account_id,
                        gl_account="LIABILITY:SHORT_PROCEEDS",
                        currency=currency,
                        credit=outcome.opened_short_proceeds,
                        instrument_id=instrument_id,
                        memo=f"Open short liability {execution_id}",
                    ),
                ]
            )
        if outcome.closed_short_qty > ZERO:
            lines.extend(
                [
                    JournalLine(
                        account_id=account_id,
                        gl_account="LIABILITY:SHORT_PROCEEDS",
                        currency=currency,
                        debit=outcome.closed_short_proceeds,
                        instrument_id=instrument_id,
                        memo=f"Close short liability {execution_id}",
                    ),
                    JournalLine(
                        account_id=account_id,
                        gl_account="ASSET:CASH",
                        currency=currency,
                        credit=outcome.closed_short_cover_cost,
                        instrument_id=instrument_id,
                        memo=f"Close short cover {execution_id}",
                    ),
                ]
            )
            if outcome.closed_short_pnl > ZERO:
                lines.append(
                    JournalLine(
                        account_id=account_id,
                        gl_account="INCOME:REALIZED_TRADING_PNL",
                        currency=currency,
                        credit=outcome.closed_short_pnl,
                        instrument_id=instrument_id,
                        memo=f"Realized short gain {execution_id}",
                    )
                )
            elif outcome.closed_short_pnl < ZERO:
                lines.append(
                    JournalLine(
                        account_id=account_id,
                        gl_account="INCOME:REALIZED_TRADING_PNL",
                        currency=currency,
                        debit=abs(outcome.closed_short_pnl),
                        instrument_id=instrument_id,
                        memo=f"Realized short loss {execution_id}",
                    )
                )
        return lines
