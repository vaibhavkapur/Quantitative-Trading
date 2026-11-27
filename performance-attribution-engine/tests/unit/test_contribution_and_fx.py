"""Unit tests for daily/Carino-linked contributions, FX decomposition, and provenance."""

from __future__ import annotations

import unittest

from pae.config import AttributionConfig
from pae.domain.models import (
    AttributionInputBatch,
    DailyNavSnapshot,
    FxRateObservation,
    PositionSnapshot,
    TradeExecution,
)
from pae.engine.pipeline import AttributionPipeline


class TestContributionAndFx(unittest.TestCase):
    def test_exact_fx_decomposition_local_plus_translation_plus_interaction(self) -> None:
        # 1000 shares of EU stock @ EUR 50, FX 1.10 -> MV0 = $55,000; Cash = $45,000; NAV0 = $100,000
        # Price goes 50 -> 55 (+10%); FX goes 1.10 -> 1.21 (+10%)
        # Exact base return = 1.10 * 1.10 - 1 = +21% -> P&L = $55,000 * 0.21 = $11,550
        # Local price P&L = 1000 * 5 * 1.10 = $5,500 (+5.50% contrib)
        # FX translation  = 1000 * 50 * 0.11 = $5,500 (+5.50% contrib)
        # FX interaction  = 1000 * 5 * 0.11 = $550 (+0.55% contrib)
        batch = AttributionInputBatch(
            batch_id="B_FX",
            portfolio_id="P_FX",
            reporting_currency="USD",
            as_of_utc="2026-01-06T00:00:00Z",
            nav_snapshots=[DailyNavSnapshot("N1", "2026-01-05", "P_FX", "USD", 100000.0, 111550.0, 45000.0, 45000.0)],
            positions=[
                PositionSnapshot(
                    "POS-1", "2026-01-05", "P_FX", "S_EU", "SAP_SIM", "TECH", "EQUITY", "EUR",
                    1000.0, 1000.0, 50.0, 55.0,
                )
            ],
            fx_rates=[
                FxRateObservation("EUR/USD", "EUR", "USD", "2026-01-02", 1.10, "2026-01-02T21:00:00Z", "2026-01-02T21:05:00Z"),
                FxRateObservation("EUR/USD", "EUR", "USD", "2026-01-05", 1.21, "2026-01-05T21:00:00Z", "2026-01-05T21:05:00Z"),
            ],
        )
        rpt = AttributionPipeline(AttributionConfig()).run(batch)
        pc = rpt.contributions.daily_positions[0]
        self.assertAlmostEqual(pc.local_price_contribution, 0.055, places=12)
        self.assertAlmostEqual(pc.fx_translation_contribution, 0.055, places=12)
        self.assertAlmostEqual(pc.fx_interaction_contribution, 0.0055, places=12)
        self.assertAlmostEqual(pc.gross_contribution, 0.1155, places=12)
        self.assertAlmostEqual(rpt.reconciliation.linked_net_residual, 0.0, places=12)
        self.assertEqual(rpt.reconciliation.status, "RECONCILED")

    def test_quantity_mismatch_and_orphan_trade_flagged(self) -> None:
        batch = AttributionInputBatch(
            batch_id="B_MISMATCH",
            portfolio_id="P1",
            reporting_currency="USD",
            as_of_utc="2026-01-06T00:00:00Z",
            nav_snapshots=[DailyNavSnapshot("N1", "2026-01-05", "P1", "USD", 10000.0, 10000.0)],
            positions=[
                # start_qty=100, end_qty=150 with no matching trade -> QUANTITY_MISMATCH
                PositionSnapshot("POS-1", "2026-01-05", "P1", "S1", "AAPL_SIM", "TECH", "EQUITY", "USD", 100.0, 150.0, 100.0, 100.0)
            ],
            trades=[
                # Trade on MSFT_SIM with no matching PositionSnapshot -> ORPHAN_TRADE
                TradeExecution(
                    "TRD-ORPHAN", "ORD-1", "2026-01-05", "2026-01-05T15:00:00Z",
                    "P1", "S1", "MSFT_SIM", "TECH", "BUY", 10.0, 10.0, "USD",
                    200.0, 200.0, 200.0, 200.0,
                )
            ],
        )
        rpt = AttributionPipeline(AttributionConfig()).run(batch)
        codes = {e.code for e in rpt.reconciliation.exceptions}
        self.assertIn("QUANTITY_MISMATCH", codes)
        self.assertIn("ORPHAN_TRADE", codes)
        self.assertEqual(rpt.reconciliation.status, "UNRECONCILED")


if __name__ == "__main__":
    unittest.main()
