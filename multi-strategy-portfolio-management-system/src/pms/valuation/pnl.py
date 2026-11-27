"""P&L bridge between two point-in-time valuation snapshots."""

from __future__ import annotations

from decimal import Decimal

from pms.domain.models import FrozenModel
from pms.valuation.consolidation import FundValuationReport

ZERO = Decimal("0")


class PnLBridge(FrozenModel):
    from_snapshot_id: str
    to_snapshot_id: str
    delta_nav_base: Decimal
    delta_net_capital_flows_base: Decimal
    delta_realized_trading_pnl_base: Decimal
    delta_unrealized_price_pnl_base: Decimal
    delta_fx_translation_pnl_base: Decimal
    delta_dividend_income_base: Decimal
    delta_fees_and_commissions_base: Decimal
    delta_late_fees_base: Decimal
    delta_adjustments_base: Decimal
    explained_delta_pnl_base: Decimal
    actual_delta_pnl_base: Decimal
    unexplained_residual_base: Decimal


def compute_pnl_bridge(
    before: FundValuationReport, after: FundValuationReport
) -> PnLBridge:
    """Explain exact NAV and P&L change between two snapshots (e.g., before/after late fee or mark move)."""
    delta_nav = after.fund_nav_base - before.fund_nav_base
    delta_cap = after.net_capital_flows_base - before.net_capital_flows_base
    delta_realized = after.pnl.realized_trading_pnl_base - before.pnl.realized_trading_pnl_base
    delta_unrealized = after.pnl.unrealized_price_pnl_base - before.pnl.unrealized_price_pnl_base
    delta_fx = after.pnl.fx_translation_pnl_base - before.pnl.fx_translation_pnl_base
    delta_div = after.pnl.dividend_income_base - before.pnl.dividend_income_base
    delta_fees = after.pnl.fees_and_commissions_base - before.pnl.fees_and_commissions_base
    delta_late_fees = after.pnl.late_fees_base - before.pnl.late_fees_base
    delta_adj = after.pnl.adjustments_base - before.pnl.adjustments_base

    explained = (
        delta_realized + delta_unrealized + delta_fx + delta_div - delta_fees + delta_adj
    )
    actual = delta_nav - delta_cap
    residual = actual - explained

    return PnLBridge(
        from_snapshot_id=before.snapshot_id,
        to_snapshot_id=after.snapshot_id,
        delta_nav_base=delta_nav,
        delta_net_capital_flows_base=delta_cap,
        delta_realized_trading_pnl_base=delta_realized,
        delta_unrealized_price_pnl_base=delta_unrealized,
        delta_fx_translation_pnl_base=delta_fx,
        delta_dividend_income_base=delta_div,
        delta_fees_and_commissions_base=delta_fees,
        delta_late_fees_base=delta_late_fees,
        delta_adjustments_base=delta_adj,
        explained_delta_pnl_base=explained,
        actual_delta_pnl_base=actual,
        unexplained_residual_base=residual,
    )
