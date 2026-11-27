"""Exact Decimal lot accounting (FIFO, LIFO, AVG_COST) with flip and split support."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from pms.domain.enums import LotMethod, Side
from pms.domain.models import FrozenModel

ZERO = Decimal("0")


class OpenLot(FrozenModel):
    lot_id: str
    account_id: str
    instrument_id: str
    currency: str
    opened_at: datetime
    source_event_id: str
    quantity: Decimal  # >0 long, <0 short
    unit_cost: Decimal  # positive price per unit
    multiplier: Decimal
    cost_basis_abs: Decimal  # positive remaining cost basis (long) or short proceeds (short)

    @property
    def is_long(self) -> bool:
        return self.quantity > ZERO

    @property
    def signed_cost_basis(self) -> Decimal:
        """Positive for long cost basis, negative for short proceeds liability."""
        return self.cost_basis_abs if self.quantity > ZERO else -self.cost_basis_abs


@dataclass
class TradeLotOutcome:
    closed_long_qty: Decimal = ZERO
    closed_long_cost: Decimal = ZERO
    closed_long_proceeds: Decimal = ZERO
    closed_long_pnl: Decimal = ZERO

    opened_long_qty: Decimal = ZERO
    opened_long_cost: Decimal = ZERO

    closed_short_qty: Decimal = ZERO
    closed_short_proceeds: Decimal = ZERO
    closed_short_cover_cost: Decimal = ZERO
    closed_short_pnl: Decimal = ZERO

    opened_short_qty: Decimal = ZERO
    opened_short_proceeds: Decimal = ZERO

    flipped: bool = False

    @property
    def total_realized_pnl(self) -> Decimal:
        return self.closed_long_pnl + self.closed_short_pnl


class LotBook:
    """Mutable replay-time lot book for a single (account_id, instrument_id)."""

    def __init__(
        self,
        *,
        account_id: str,
        instrument_id: str,
        currency: str,
        multiplier: Decimal,
        lot_method: LotMethod = LotMethod.FIFO,
    ) -> None:
        self.account_id = account_id
        self.instrument_id = instrument_id
        self.currency = currency
        self.multiplier = multiplier
        self.lot_method = lot_method
        self._lots: list[OpenLot] = []

    @property
    def lots(self) -> list[OpenLot]:
        return list(self._lots)

    @property
    def net_quantity(self) -> Decimal:
        return sum((lot.quantity for lot in self._lots), ZERO)

    @property
    def signed_cost_basis(self) -> Decimal:
        return sum((lot.signed_cost_basis for lot in self._lots), ZERO)

    @property
    def average_unit_cost(self) -> Decimal:
        qty = self.net_quantity
        if qty == ZERO:
            return ZERO
        total_abs_cost = sum((lot.cost_basis_abs for lot in self._lots), ZERO)
        return total_abs_cost / (abs(qty) * self.multiplier)

    def _select_lot_index(self) -> int:
        if self.lot_method == LotMethod.LIFO:
            return len(self._lots) - 1
        return 0

    def _consolidate_avg_cost(self, opened_at: datetime, event_id: str) -> None:
        if self.lot_method != LotMethod.AVG_COST or len(self._lots) <= 1:
            return
        total_qty = self.net_quantity
        if total_qty == ZERO:
            self._lots.clear()
            return
        total_abs_cost = sum((lot.cost_basis_abs for lot in self._lots), ZERO)
        avg_unit = total_abs_cost / (abs(total_qty) * self.multiplier)
        self._lots = [
            OpenLot(
                lot_id=f"LOT-AVG-{self.account_id}-{self.instrument_id}",
                account_id=self.account_id,
                instrument_id=self.instrument_id,
                currency=self.currency,
                opened_at=opened_at,
                source_event_id=event_id,
                quantity=total_qty,
                unit_cost=avg_unit,
                multiplier=self.multiplier,
                cost_basis_abs=total_abs_cost,
            )
        ]

    def apply_trade(
        self,
        *,
        seq_no: int,
        event_id: str,
        event_time: datetime,
        side: Side,
        quantity: Decimal,
        price: Decimal,
    ) -> TradeLotOutcome:
        outcome = TradeLotOutcome()
        remaining = quantity
        prior_net = self.net_quantity

        if side == Side.BUY:
            # First close any existing SHORT lots (quantity < 0)
            while remaining > ZERO and self._lots and self._lots[0].quantity < ZERO:
                idx = self._select_lot_index()
                lot = self._lots[idx]
                lot_abs_qty = abs(lot.quantity)
                close_qty = min(remaining, lot_abs_qty)

                if close_qty == lot_abs_qty:
                    proceeds_closed = lot.cost_basis_abs
                    self._lots.pop(idx)
                else:
                    proceeds_closed = (lot.cost_basis_abs * close_qty) / lot_abs_qty
                    new_abs_qty = lot_abs_qty - close_qty
                    new_cost_abs = lot.cost_basis_abs - proceeds_closed
                    self._lots[idx] = lot.model_copy(
                        update={
                            "quantity": -new_abs_qty,
                            "cost_basis_abs": new_cost_abs,
                        }
                    )

                cover_cost = close_qty * price * self.multiplier
                outcome.closed_short_qty += close_qty
                outcome.closed_short_proceeds += proceeds_closed
                outcome.closed_short_cover_cost += cover_cost
                outcome.closed_short_pnl += proceeds_closed - cover_cost
                remaining -= close_qty

            # Remaining quantity opens a LONG lot
            if remaining > ZERO:
                open_cost = remaining * price * self.multiplier
                suffix = "FLIP" if prior_net < ZERO else "OPEN"
                self._lots.append(
                    OpenLot(
                        lot_id=f"LOT-{seq_no}-{suffix}",
                        account_id=self.account_id,
                        instrument_id=self.instrument_id,
                        currency=self.currency,
                        opened_at=event_time,
                        source_event_id=event_id,
                        quantity=remaining,
                        unit_cost=price,
                        multiplier=self.multiplier,
                        cost_basis_abs=open_cost,
                    )
                )
                outcome.opened_long_qty = remaining
                outcome.opened_long_cost = open_cost
                if prior_net < ZERO:
                    outcome.flipped = True
        else:
            # SELL: first close any existing LONG lots (quantity > 0)
            while remaining > ZERO and self._lots and self._lots[0].quantity > ZERO:
                idx = self._select_lot_index()
                lot = self._lots[idx]
                lot_abs_qty = lot.quantity
                close_qty = min(remaining, lot_abs_qty)

                if close_qty == lot_abs_qty:
                    cost_closed = lot.cost_basis_abs
                    self._lots.pop(idx)
                else:
                    cost_closed = (lot.cost_basis_abs * close_qty) / lot_abs_qty
                    new_abs_qty = lot_abs_qty - close_qty
                    new_cost_abs = lot.cost_basis_abs - cost_closed
                    self._lots[idx] = lot.model_copy(
                        update={
                            "quantity": new_abs_qty,
                            "cost_basis_abs": new_cost_abs,
                        }
                    )

                proceeds = close_qty * price * self.multiplier
                outcome.closed_long_qty += close_qty
                outcome.closed_long_cost += cost_closed
                outcome.closed_long_proceeds += proceeds
                outcome.closed_long_pnl += proceeds - cost_closed
                remaining -= close_qty

            # Remaining quantity opens a SHORT lot
            if remaining > ZERO:
                open_proceeds = remaining * price * self.multiplier
                suffix = "FLIP" if prior_net > ZERO else "OPEN"
                self._lots.append(
                    OpenLot(
                        lot_id=f"LOT-{seq_no}-{suffix}",
                        account_id=self.account_id,
                        instrument_id=self.instrument_id,
                        currency=self.currency,
                        opened_at=event_time,
                        source_event_id=event_id,
                        quantity=-remaining,
                        unit_cost=price,
                        multiplier=self.multiplier,
                        cost_basis_abs=open_proceeds,
                    )
                )
                outcome.opened_short_qty = remaining
                outcome.opened_short_proceeds = open_proceeds
                if prior_net > ZERO:
                    outcome.flipped = True

        self._consolidate_avg_cost(event_time, event_id)
        return outcome

    def apply_split(self, ratio_numerator: Decimal, ratio_denominator: Decimal) -> Decimal:
        """Apply a stock split preserving exact total cost basis on every open lot."""
        ratio = ratio_numerator / ratio_denominator
        updated: list[OpenLot] = []
        for lot in self._lots:
            new_qty = lot.quantity * ratio
            new_unit_cost = lot.cost_basis_abs / (abs(new_qty) * self.multiplier)
            updated.append(
                lot.model_copy(
                    update={
                        "quantity": new_qty,
                        "unit_cost": new_unit_cost,
                    }
                )
            )
        self._lots = updated
        return self.net_quantity

    def apply_position_adjustment(
        self,
        *,
        seq_no: int,
        event_id: str,
        event_time: datetime,
        quantity_delta: Decimal,
        unit_cost: Decimal,
    ) -> TradeLotOutcome:
        """Apply an approved reconciliation position adjustment as a synthetic buy/sell without cash flow."""
        side = Side.BUY if quantity_delta > ZERO else Side.SELL
        return self.apply_trade(
            seq_no=seq_no,
            event_id=event_id,
            event_time=event_time,
            side=side,
            quantity=abs(quantity_delta),
            price=unit_cost,
        )
