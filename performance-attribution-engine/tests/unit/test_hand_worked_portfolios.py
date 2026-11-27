"""Verification against analytically hand-worked portfolios (plan.md M4 requirement)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from pae.config import AttributionConfig
from pae.domain.models import (
    AttributionInputBatch,
    BenchmarkObservation,
    DailyNavSnapshot,
    PositionSnapshot,
    TradeExecution,
)
from pae.engine.pipeline import AttributionPipeline

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "data" / "fixtures"


class TestHandWorkedPortfolios(unittest.TestCase):
    def test_hand_worked_2day_multiccy_rebalance_cashflow_portfolio(self) -> None:
        """Verify 2-day USD+EUR portfolio with SOD cash flow, trade, commission, and fee to 1e-10."""
        raw = json.loads((FIXTURES_DIR / "hand_worked_portfolio.json").read_text(encoding="utf-8"))
        batch = AttributionInputBatch.from_dict(raw)
        rpt = AttributionPipeline(AttributionConfig()).run(batch)

        # Day 1 exact analytical values: Gross=+11.20%, Fee=-0.20%, Net=+11.00%
        d1 = rpt.returns.daily_returns[0]
        self.assertAlmostEqual(d1.gross_return, 0.1120, places=10)
        self.assertAlmostEqual(d1.fee_drag_return, 0.0020, places=10)
        self.assertAlmostEqual(d1.net_return, 0.1100, places=10)

        # Day 2 exact analytical values (SOD CF +$89k -> capital base $200k):
        # Gross=+2.00%, Fee=-0.05%, Net=+1.95%
        d2 = rpt.returns.daily_returns[1]
        self.assertAlmostEqual(d2.capital_base, 200000.0, places=10)
        self.assertAlmostEqual(d2.gross_return, 0.0200, places=10)
        self.assertAlmostEqual(d2.fee_drag_return, 0.0005, places=10)
        self.assertAlmostEqual(d2.net_return, 0.0195, places=10)

        # Cumulative returns: Gross=(1.112*1.02)-1 = 0.13424, Net=(1.11*1.0195)-1 = 0.131645
        self.assertAlmostEqual(rpt.returns.cumulative_gross_return, 0.13424, places=10)
        self.assertAlmostEqual(rpt.returns.cumulative_net_return, 0.131645, places=10)

        # Hand-worked Day 1 EU_FIN FX decomposition
        eu_d1 = [p for p in rpt.contributions.daily_positions if p.position_id == "HW-POS-1-EU"][0]
        self.assertAlmostEqual(eu_d1.local_price_pnl_base, 2000.0, places=10)
        self.assertAlmostEqual(eu_d1.fx_translation_pnl_base, 4000.0, places=10)
        self.assertAlmostEqual(eu_d1.fx_interaction_pnl_base, 200.0, places=10)
        self.assertAlmostEqual(eu_d1.gross_contribution, 0.062, places=10)

        # Hand-worked Day 2 US_TECH Perold Implementation Shortfall:
        # BUY order_qty=600, filled=500, decision=110, arrival=111, exec=112, close=115, comm=100, vwap=111.5
        # Delay = 500*(111-110) = $500
        # Impact = 500*(112-111) = $500
        # Opportunity = 100*(115-110) = $500
        # Explicit = $100
        # Total IS = $1,600; VWAP slippage = 500*(112-111.5) = $250
        tca = rpt.execution_costs.trades[0]
        self.assertAlmostEqual(tca.delay_cost_base, 500.0, places=10)
        self.assertAlmostEqual(tca.market_impact_cost_base, 500.0, places=10)
        self.assertAlmostEqual(tca.opportunity_cost_base, 500.0, places=10)
        self.assertAlmostEqual(tca.explicit_cost_base, 100.0, places=10)
        self.assertAlmostEqual(tca.total_is_cost_base, 1600.0, places=10)
        self.assertAlmostEqual(tca.vwap_slippage_base, 250.0, places=10)

        # Reconciliation exact to machine precision
        self.assertEqual(rpt.reconciliation.status, "RECONCILED")
        self.assertAlmostEqual(rpt.reconciliation.linked_net_residual, 0.0, places=10)

    def test_hand_worked_3_sector_brinson_fachler_and_bhb(self) -> None:
        """Exact paper calculation for 3-sector Brinson-Fachler and Brinson-Hood-Beebower.

        Portfolio ($100,000, zero cash):
          TECH:       w_p=0.50, r_p=0.10 -> contrib = 0.050
          FINANCIALS: w_p=0.30, r_p=0.02 -> contrib = 0.006
          ENERGY:     w_p=0.20, r_p=-0.05 -> contrib = -0.010
          R_p = +0.046 (+4.60%)

        Benchmark:
          TECH:       w_b=0.40, r_b=0.08 -> contrib = 0.032
          FINANCIALS: w_b=0.40, r_b=0.04 -> contrib = 0.016
          ENERGY:     w_b=0.20, r_b=-0.02 -> contrib = -0.004
          R_b = +0.044 (+4.40%)
          Active Return = +0.002 (+20 bps)

        Brinson-Fachler exact effects:
          Allocation:
            TECH: (0.5-0.4)*(0.08 - 0.044) = +0.0036
            FIN:  (0.3-0.4)*(0.04 - 0.044) = +0.0004
            ENG:  (0.2-0.2)*(-0.02 - 0.044) = 0.0000
            Total Allocation = +0.0040 (+40 bps)
          Selection:
            TECH: 0.4*(0.10 - 0.08) = +0.0080
            FIN:  0.4*(0.02 - 0.04) = -0.0080
            ENG:  0.2*(-0.05 - (-0.02)) = -0.0060
            Total Selection = -0.0060 (-60 bps)
          Interaction:
            TECH: (0.5-0.4)*(0.10 - 0.08) = +0.0020
            FIN:  (0.3-0.4)*(0.02 - 0.04) = +0.0020
            ENG:  0.0
            Total Interaction = +0.0040 (+40 bps)
          Sum = +0.0040 - 0.0060 + 0.0040 = +0.0020 == Active Return!
        """
        batch = AttributionInputBatch(
            batch_id="HW-BRINSON",
            portfolio_id="P_BR",
            reporting_currency="USD",
            as_of_utc="2026-01-06T00:00:00Z",
            benchmark_id="BM_3SEC",
            nav_snapshots=[DailyNavSnapshot("N1", "2026-01-05", "P_BR", "USD", 100000.0, 104600.0)],
            positions=[
                PositionSnapshot("P_T", "2026-01-05", "P_BR", "S1", "T1", "TECH", "EQUITY", "USD", 500.0, 500.0, 100.0, 110.0),
                PositionSnapshot("P_F", "2026-01-05", "P_BR", "S1", "F1", "FINANCIALS", "EQUITY", "USD", 300.0, 300.0, 100.0, 102.0),
                PositionSnapshot("P_E", "2026-01-05", "P_BR", "S1", "E1", "ENERGY", "EQUITY", "USD", 200.0, 200.0, 100.0, 95.0),
            ],
            benchmark=[
                BenchmarkObservation(
                    "BM_3SEC",
                    "2026-01-05",
                    "USD",
                    0.044,
                    sector_weights={"TECH": 0.40, "FINANCIALS": 0.40, "ENERGY": 0.20},
                    sector_returns={"TECH": 0.08, "FINANCIALS": 0.04, "ENERGY": -0.02},
                )
            ],
        )

        # Brinson-Fachler
        rpt_bf = AttributionPipeline(AttributionConfig(brinson_model="BRINSON_FACHLER")).run(batch)
        self.assertAlmostEqual(rpt_bf.brinson.total_allocation_effect, 0.0040, places=10)
        self.assertAlmostEqual(rpt_bf.brinson.total_selection_effect, -0.0060, places=10)
        self.assertAlmostEqual(rpt_bf.brinson.total_interaction_effect, 0.0040, places=10)
        self.assertAlmostEqual(rpt_bf.brinson.total_residual, 0.0, places=10)

        # Brinson-Hood-Beebower: Allocation = (0.1*0.08) + (-0.1*0.04) = +0.0040 too because sum(wp-wb)=0!
        rpt_bhb = AttributionPipeline(AttributionConfig(brinson_model="BRINSON_HOOD_BEEBOWER")).run(batch)
        self.assertAlmostEqual(rpt_bhb.brinson.total_allocation_effect, 0.0040, places=10)
        # Per-sector allocation differs between BF and BHB on TECH: BF=0.0036 vs BHB=0.0080
        tech_bf = [s for s in rpt_bf.brinson.by_sector if s.sector == "TECH"][0]
        tech_bhb = [s for s in rpt_bhb.brinson.by_sector if s.sector == "TECH"][0]
        self.assertAlmostEqual(tech_bf.allocation_effect, 0.0036, places=10)
        self.assertAlmostEqual(tech_bhb.allocation_effect, 0.0080, places=10)

    def test_hand_worked_sell_side_tca_price_improvement(self) -> None:
        """Verify SELL trade where execution beats arrival price yields negative impact (price improvement)."""
        # SELL 1000 shares: decision=100, arrival=99 (delay cost = -1000*(99-100)=+$1000),
        # execution=99.5 (price improvement vs arrival: -1000*(99.5-99)=-$500), comm=$50 -> Total IS = $550
        batch = AttributionInputBatch(
            batch_id="HW-SELL-TCA",
            portfolio_id="P_TCA",
            reporting_currency="USD",
            as_of_utc="2026-01-06T00:00:00Z",
            nav_snapshots=[DailyNavSnapshot("N1", "2026-01-05", "P_TCA", "USD", 100000.0, 99450.0)],
            positions=[
                PositionSnapshot("P1", "2026-01-05", "P_TCA", "S1", "X", "TECH", "EQUITY", "USD", 1000.0, 0.0, 100.0, 99.0)
            ],
            trades=[
                TradeExecution(
                    "T1", "O1", "2026-01-05", "2026-01-05T15:00:00Z", "P_TCA", "S1", "X", "TECH",
                    "SELL", 1000.0, 1000.0, "USD", 100.0, 99.0, 99.5, 99.0, commission_local=50.0,
                )
            ],
        )
        rpt = AttributionPipeline(AttributionConfig()).run(batch)
        t = rpt.execution_costs.trades[0]
        self.assertAlmostEqual(t.delay_cost_base, 1000.0, places=10)
        self.assertAlmostEqual(t.market_impact_cost_base, -500.0, places=10)
        self.assertAlmostEqual(t.explicit_cost_base, 50.0, places=10)
        self.assertAlmostEqual(t.total_is_cost_base, 550.0, places=10)
        self.assertEqual(rpt.reconciliation.status, "RECONCILED")


if __name__ == "__main__":
    unittest.main()
