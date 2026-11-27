"""Unit tests for Brinson attribution, risk-factor regression, TCA, and upstream adapters."""

from __future__ import annotations

import unittest

from pae.adapters.contracts import (
    HFTFillAdapter,
    OptionalPolarsAdapter,
    PortfolioOptimizerTargetAdapter,
    UpstreamTCAAdapter,
)
from pae.config import AttributionConfig
from pae.domain.models import (
    AttributionInputBatch,
    DailyNavSnapshot,
    FactorObservation,
    PositionSnapshot,
)
from pae.engine.pipeline import AttributionPipeline


class TestFactorAndAdapters(unittest.TestCase):
    def test_factor_regression_recovers_exact_betas_and_includes_non_causation_disclaimer(self) -> None:
        # Construct 8 days where r_gross = rf + 0.0005 + 1.25 * MKT - 0.40 * HML exactly
        mkt = [0.010, -0.008, 0.015, 0.004, -0.012, 0.009, 0.006, -0.005]
        hml = [0.005, 0.007, -0.004, 0.006, -0.002, -0.005, 0.008, 0.003]
        rf = 0.0001
        nav = 100000.0
        navs = []
        positions = []
        factors = []
        px = 100.0
        qty = 1000.0

        for i in range(8):
            d = f"2026-01-{5 + i:02d}"
            r = rf + 0.0005 + 1.25 * mkt[i] - 0.40 * hml[i]
            end_px = px * (1.0 + r)
            end_nav = nav * (1.0 + r)
            navs.append(DailyNavSnapshot(f"N{i}", d, "PF", "USD", nav, end_nav))
            positions.append(PositionSnapshot(f"P{i}", d, "PF", "S1", "SYNTH", "TECH", "EQUITY", "USD", qty, qty, px, end_px))
            factors.append(FactorObservation(d, {"MKT": mkt[i], "HML": hml[i]}, risk_free_rate=rf))
            nav = end_nav
            px = end_px

        batch = AttributionInputBatch(
            "B_FAC", "PF", "USD", "2026-01-20T00:00:00Z",
            nav_snapshots=navs, positions=positions, factors=factors,
        )
        cfg = AttributionConfig(factor_ridge_regularization=0.0)
        rpt = AttributionPipeline(cfg).run(batch)

        self.assertEqual(rpt.factors.status, "AVAILABLE")
        self.assertIn("does not prove causation", rpt.factors.disclaimer.lower())
        betas = {f.factor_name: f.beta for f in rpt.factors.factors}
        self.assertAlmostEqual(betas["MKT"], 1.25, places=6)
        self.assertAlmostEqual(betas["HML"], -0.40, places=6)
        self.assertAlmostEqual(rpt.factors.daily_alpha, 0.0005, places=6)
        self.assertAlmostEqual(rpt.factors.r_squared, 1.0, places=6)
        self.assertAlmostEqual(rpt.factors.residual_check, 0.0, places=10)

    def test_upstream_adapters_convert_cleanly(self) -> None:
        tca_adapter = UpstreamTCAAdapter()
        trades = tca_adapter.adapt_records(
            [
                {
                    "fill_id": "F-1",
                    "parent_order_id": "O-1",
                    "trade_date": "2026-01-05",
                    "fill_ts_utc": "2026-01-05T15:00:00Z",
                    "strategy_id": "S1",
                    "symbol": "AAPL",
                    "sector": "TECH",
                    "side": "BUY",
                    "order_qty": 100,
                    "filled_qty": 80,
                    "decision_px": 100.0,
                    "arrival_px": 100.2,
                    "exec_px": 100.5,
                    "close_px": 101.0,
                    "commission": 5.0,
                }
            ],
            portfolio_id="P1",
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].unfilled_signed_quantity, 20.0)

        hft_adapter = HFTFillAdapter()
        hft_trades = hft_adapter.adapt_fills(
            [
                {
                    "exec_id": "E-1",
                    "order_id": "HO-1",
                    "ts_utc": "2026-01-05T15:00:00Z",
                    "instrument": "MSFT",
                    "side": "S",
                    "qty": 50,
                    "price": 300.0,
                    "arrival_mid_price": 300.1,
                    "fee": 1.2,
                }
            ],
            portfolio_id="P1",
            strategy_id="HFT_1",
            sector_lookup={"MSFT": "TECH"},
        )
        self.assertEqual(hft_trades[0].side, "SELL")
        self.assertEqual(hft_trades[0].sector, "TECH")

        opt_adapter = PortfolioOptimizerTargetAdapter()
        drift = opt_adapter.compute_rebalance_drift({"A": 0.6, "B": 0.4}, {"A": 0.58, "B": 0.42})
        self.assertAlmostEqual(drift["l1_weight_drift"], 0.04, places=12)

        rows = [{"a": 1}, {"a": 2}]
        converted = OptionalPolarsAdapter.to_polars_or_rows(rows)
        self.assertIsNotNone(converted)


if __name__ == "__main__":
    unittest.main()
