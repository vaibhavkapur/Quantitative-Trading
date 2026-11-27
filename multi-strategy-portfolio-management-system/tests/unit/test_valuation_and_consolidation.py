"""Unit tests for MTM valuation, P&L explainability, P&L bridge, and consolidation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from pms.domain.enums import PricingStatus
from pms.service import PortfolioService
from pms.valuation.pnl import compute_pnl_bridge
from pms.valuation.pricing import MarketSnapshot, PriceMark


def test_consolidation_and_pnl_explainability_on_baseline(
    service_with_fixtures: PortfolioService,
) -> None:
    report = service_with_fixtures.get_valuation()

    assert report.consolidation_check.passed is True
    assert report.consolidation_check.nav_difference_base == Decimal("0")
    assert report.consolidation_check.cash_difference_base == Decimal("0")
    assert report.consolidation_check.pnl_difference_base == Decimal("0")

    # Every single cent of P&L must be explained
    assert report.pnl.unexplained_residual_base == Decimal("0")
    assert report.pnl.explained_total_pnl_base == report.pnl.nav_minus_net_capital_base

    # Unpriced instrument EQ-UNPRICED-US must be explicitly flagged
    assert "EQ-UNPRICED-US" in report.unpriced_instruments
    assert report.valuation_complete is False
    assert any(ex.code == "MISSING_MARK" and ex.instrument_id == "EQ-UNPRICED-US" for ex in report.exceptions)


def test_stale_mark_is_flagged_as_unpriced(
    service_with_fixtures: PortfolioService,
    baseline_market: MarketSnapshot,
) -> None:
    stale_time = baseline_market.as_of_time - timedelta(days=3)
    modified_prices = tuple(
        PriceMark(
            instrument_id=p.instrument_id,
            price=p.price,
            currency=p.currency,
            price_timestamp=stale_time if p.instrument_id == "EQ-AAPL-US" else p.price_timestamp,
            availability_timestamp=p.availability_timestamp,
            source=p.source,
        )
        for p in baseline_market.prices
    )
    stale_snapshot = baseline_market.model_copy(
        update={"snapshot_id": "SNAP-STALE-TEST", "prices": modified_prices}
    )
    service_with_fixtures.register_market_snapshot(stale_snapshot)
    report = service_with_fixtures.get_valuation(snapshot_id="SNAP-STALE-TEST")

    assert "EQ-AAPL-US" in report.unpriced_instruments
    aapl_positions = [
        pos
        for acct in report.strategy_valuations
        for pos in acct.positions
        if pos.instrument_id == "EQ-AAPL-US"
    ]
    assert len(aapl_positions) == 1
    assert aapl_positions[0].pricing_status == PricingStatus.STALE_MARK
    assert aapl_positions[0].is_priced is False


def test_pnl_bridge_explains_mark_move_exactly(
    service_with_fixtures: PortfolioService,
    baseline_market: MarketSnapshot,
) -> None:
    before = service_with_fixtures.get_valuation()

    # Bump AAPL mark from 98.00 to 100.00 (+2.00 * 2000 shares = +4000.00 USD)
    bumped_prices = tuple(
        p.model_copy(update={"price": Decimal("100.00")})
        if p.instrument_id == "EQ-AAPL-US"
        else p
        for p in baseline_market.prices
    )
    after_market = baseline_market.model_copy(
        update={
            "snapshot_id": "SNAP-COD-20260115-T2",
            "as_of_time": datetime(2026, 1, 15, 22, 0, 0, tzinfo=UTC),
            "prices": bumped_prices,
        }
    )
    service_with_fixtures.register_market_snapshot(after_market)
    after = service_with_fixtures.get_valuation(snapshot_id="SNAP-COD-20260115-T2")

    bridge = compute_pnl_bridge(before, after)
    assert bridge.delta_nav_base == Decimal("4000.00")
    assert bridge.delta_unrealized_price_pnl_base == Decimal("4000.00")
    assert bridge.unexplained_residual_base == Decimal("0")
