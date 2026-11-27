"""Mandatory scenario tests covering all 7 required failure & edge cases from plan.md Section 5.

Required scenarios:
  1. External cash flow
  2. Zero nav
  3. Fee corrections
  4. Currency conversion
  5. Missing benchmark
  6. Rebalance
  7. Rounding residual
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from pae.api.service import AttributionService
from pae.config import AttributionConfig
from pae.domain.models import (
    AttributionInputBatch,
    BenchmarkObservation,
    CashFlow,
    DailyNavSnapshot,
    FeeAdjustment,
    FxRateObservation,
    PositionSnapshot,
    TradeExecution,
)
from pae.engine.pipeline import AttributionPipeline
from pae.errors import ZeroNavError
from pae.security.rbac import Actor, Role
from pae.storage.sqlite_store import SQLiteAttributionStore

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "data" / "fixtures"


class TestRequiredScenarios(unittest.TestCase):
    def test_scenario_1_external_cash_flow(self) -> None:
        """External cash flow: huge SOD subscription & EOD redemption must not distort TWR."""
        # Day 1: Start NAV $100k, SOD inflow +$900k (capital base $1,000k).
        # Hold $100k equity (+5% -> +$5,000) + $900k cash ($0 interest).
        # True TWR = +$5,000 / $1,000,000 = +0.50%. EOD redemption -$500k -> End NAV = $505,000.
        batch = AttributionInputBatch(
            batch_id="SCEN-1-CF",
            portfolio_id="P_CF",
            reporting_currency="USD",
            as_of_utc="2026-01-06T00:00:00Z",
            nav_snapshots=[
                DailyNavSnapshot("N1", "2026-01-05", "P_CF", "USD", 100000.0, 505000.0, 0.0, 400000.0, 0.0),
            ],
            positions=[
                PositionSnapshot("P1", "2026-01-05", "P_CF", "S1", "EQ_1", "TECH", "EQUITY", "USD", 1000.0, 1000.0, 100.0, 105.0),
            ],
            cash_flows=[
                CashFlow("CF-SOD", "2026-01-05", "P_CF", 900000.0, "USD", "START_OF_DAY", "Large LP subscription"),
                CashFlow("CF-EOD", "2026-01-05", "P_CF", -500000.0, "USD", "END_OF_DAY", "Partial EOD redemption"),
            ],
        )
        rpt = AttributionPipeline(AttributionConfig()).run(batch)
        self.assertEqual(rpt.reconciliation.status, "RECONCILED")
        self.assertAlmostEqual(rpt.returns.cumulative_net_return, 0.005, places=12)
        self.assertAlmostEqual(rpt.returns.total_external_cash_flows_base, 400000.0, places=6)
        self.assertAlmostEqual(rpt.reconciliation.linked_net_residual, 0.0, places=12)

    def test_scenario_2_zero_nav(self) -> None:
        """Zero nav: full SOD redemption causing capital_base == 0 emits CRITICAL ZERO_NAV without NaN/Inf."""
        batch = AttributionInputBatch(
            batch_id="SCEN-2-ZERO-NAV",
            portfolio_id="P_ZERO",
            reporting_currency="USD",
            as_of_utc="2026-01-07T00:00:00Z",
            nav_snapshots=[
                DailyNavSnapshot("N1", "2026-01-05", "P_ZERO", "USD", 100000.0, 102000.0),
                # Full redemption at start of Day 2 -> capital_base = 0
                DailyNavSnapshot("N2", "2026-01-06", "P_ZERO", "USD", 102000.0, 0.0),
            ],
            positions=[
                PositionSnapshot("P1", "2026-01-05", "P_ZERO", "S1", "EQ_1", "TECH", "EQUITY", "USD", 1000.0, 1000.0, 100.0, 102.0),
                PositionSnapshot("P2", "2026-01-06", "P_ZERO", "S1", "EQ_1", "TECH", "EQUITY", "USD", 0.0, 0.0, 102.0, 102.0),
            ],
            cash_flows=[
                CashFlow("CF-FULL-RED", "2026-01-06", "P_ZERO", -102000.0, "USD", "START_OF_DAY", "Full redemption"),
            ],
        )
        # Graceful degraded mode (default)
        rpt = AttributionPipeline(AttributionConfig(fail_on_zero_nav=False)).run(batch)
        self.assertEqual(rpt.reconciliation.status, "DEGRADED_ZERO_NAV")
        self.assertEqual(rpt.returns.zero_nav_days, 1)
        self.assertTrue(any(e.code == "ZERO_NAV" and e.severity == "CRITICAL" for e in rpt.reconciliation.exceptions))
        self.assertTrue(math.isfinite(rpt.returns.cumulative_net_return))
        self.assertAlmostEqual(rpt.returns.cumulative_net_return, 0.02, places=10)

        # Strict mode raises structured ZeroNavError
        with self.assertRaises(ZeroNavError):
            AttributionPipeline(AttributionConfig(fail_on_zero_nav=True)).run(batch)

    def test_scenario_3_fee_corrections(self) -> None:
        """Fee corrections: retroactive fee adjustment creates immutable revised v2 superseding v1."""
        v1_raw = json.loads((FIXTURES_DIR / "multi_strategy_multi_ccy_baseline.json").read_text(encoding="utf-8"))
        v2_raw = json.loads((FIXTURES_DIR / "multi_strategy_fee_correction_v2.json").read_text(encoding="utf-8"))
        store = SQLiteAttributionStore(":memory:")
        svc = AttributionService(config=AttributionConfig(), store=store)
        analyst = Actor("analyst-scen", Role.ANALYST)

        v1 = svc.run_attribution(AttributionInputBatch.from_dict(v1_raw), actor=analyst)
        v2 = svc.run_attribution(
            AttributionInputBatch.from_dict(v2_raw),
            actor=analyst,
            report_id=v1["report_id"],
            correction_reason="Retroactive prime-broker rebate",
        )
        store.close()

        self.assertEqual(v2["version_number"], 2)
        self.assertEqual(v2["supersedes_version_id"], v1["version_id"])
        self.assertEqual(v2["reconciliation_status"], "RECONCILED")
        # Gross return should be virtually unchanged on day 5 (only tiny compounding base shift on later days)
        # Net return increases because -$450 rebate reduces fees
        self.assertGreater(
            v2["report"]["returns"]["cumulative_net_return"],
            v1["report"]["returns"]["cumulative_net_return"],
        )
        self.assertTrue(
            any(e["code"] == "FEE_CORRECTION_APPLIED" for e in v2["report"]["reconciliation"]["exceptions"])
        )

    def test_scenario_4_currency_conversion(self) -> None:
        """Currency conversion: multi-ccy FX translation/interaction + stale/missing FX fault detection."""
        # Case A: Clean EUR/GBP conversion with inverse quote USD/JPY
        batch = AttributionInputBatch(
            batch_id="SCEN-4-FX",
            portfolio_id="P_FX",
            reporting_currency="USD",
            as_of_utc="2026-01-06T00:00:00Z",
            nav_snapshots=[
                # 1000 JPY stock @ JPY 1000, USD/JPY moves 100 -> 125 (JPY weakens -20%: 0.01 -> 0.008)
                # Stock rises JPY 1000 -> JPY 1250 (+25% local): base MV stays 1000*1250/125 = $10,000 == $10,000!
                # Exact local = +$2,500, FX trans = -$2,000, FX interact = -$500 -> Net base P&L = $0!
                DailyNavSnapshot("N1", "2026-01-05", "P_FX", "USD", 10000.0, 10000.0),
            ],
            positions=[
                PositionSnapshot("P_JP", "2026-01-05", "P_FX", "S_JP", "JP_AUTO", "INDUSTRIALS", "EQUITY", "JPY", 1000.0, 1000.0, 1000.0, 1250.0),
            ],
            fx_rates=[
                FxRateObservation("USD/JPY", "USD", "JPY", "2026-01-02", 100.0, "2026-01-02T21:00:00Z", "2026-01-02T21:05:00Z"),
                FxRateObservation("USD/JPY", "USD", "JPY", "2026-01-05", 125.0, "2026-01-05T21:00:00Z", "2026-01-05T21:05:00Z"),
            ],
        )
        rpt = AttributionPipeline(AttributionConfig()).run(batch)
        pc = rpt.contributions.daily_positions[0]
        self.assertAlmostEqual(pc.local_price_pnl_base, 2500.0, places=8)
        self.assertAlmostEqual(pc.fx_translation_pnl_base, -2000.0, places=8)
        self.assertAlmostEqual(pc.fx_interaction_pnl_base, -500.0, places=8)
        self.assertAlmostEqual(pc.gross_pnl_base, 0.0, places=8)
        self.assertEqual(rpt.reconciliation.status, "RECONCILED")

        # Case B: Missing FX rate emits explicit MISSING_FX_RATE ERROR
        batch_missing_fx = AttributionInputBatch(
            batch_id="SCEN-4-NOFX",
            portfolio_id="P_FX",
            reporting_currency="USD",
            as_of_utc="2026-01-06T00:00:00Z",
            nav_snapshots=[DailyNavSnapshot("N1", "2026-01-05", "P_FX", "USD", 10000.0, 10000.0)],
            positions=[PositionSnapshot("P_CHF", "2026-01-05", "P_FX", "S_CH", "CH_1", "HEALTHCARE", "EQUITY", "CHF", 100.0, 100.0, 100.0, 100.0)],
            fx_rates=[],
        )
        rpt_bad = AttributionPipeline(AttributionConfig()).run(batch_missing_fx)
        self.assertTrue(any(e.code == "MISSING_FX_RATE" for e in rpt_bad.reconciliation.exceptions))
        self.assertEqual(rpt_bad.reconciliation.status, "UNRECONCILED")

    def test_scenario_5_missing_benchmark(self) -> None:
        """Missing benchmark: missing date withholds cumulative active return and emits structured warning."""
        batch = AttributionInputBatch(
            batch_id="SCEN-5-BM",
            portfolio_id="P_BM",
            reporting_currency="USD",
            as_of_utc="2026-01-07T00:00:00Z",
            benchmark_id="BM_SP500",
            nav_snapshots=[
                DailyNavSnapshot("N1", "2026-01-05", "P_BM", "USD", 100000.0, 101000.0),
                DailyNavSnapshot("N2", "2026-01-06", "P_BM", "USD", 101000.0, 102010.0),
            ],
            positions=[
                PositionSnapshot("P1", "2026-01-05", "P_BM", "S1", "EQ1", "TECH", "EQUITY", "USD", 1000.0, 1000.0, 100.0, 101.0),
                PositionSnapshot("P2", "2026-01-06", "P_BM", "S1", "EQ1", "TECH", "EQUITY", "USD", 1000.0, 1000.0, 101.0, 102.01),
            ],
            # Only Day 1 present; Day 2 benchmark is missing
            benchmark=[
                BenchmarkObservation("BM_SP500", "2026-01-05", "USD", 0.008, {"TECH": 1.0}, {"TECH": 0.008})
            ],
        )
        rpt = AttributionPipeline(AttributionConfig()).run(batch)
        codes = {e.code for e in rpt.reconciliation.exceptions}
        self.assertIn("MISSING_BENCHMARK", codes)
        self.assertIn("INCOMPLETE_BENCHMARK_SERIES", codes)
        self.assertIsNone(rpt.returns.cumulative_benchmark_return)
        self.assertIsNone(rpt.returns.cumulative_active_net_return)
        self.assertEqual(rpt.brinson.status, "PARTIAL")
        # Portfolio attribution itself still reconciled cleanly with warnings!
        self.assertEqual(rpt.reconciliation.status, "RECONCILED_WITH_WARNINGS")
        self.assertAlmostEqual(rpt.returns.cumulative_net_return, 0.0201, places=10)

    def test_scenario_6_rebalance(self) -> None:
        """Rebalance: intraday full exit + new entry + partial trim reconcile cleanly with trade provenance."""
        # Start NAV $100,000: HOLD_A 500 @ $100 ($50k), TRIM_B 500 @ $100 ($50k), NEW_C 0 @ $50 ($0).
        # Day 1 trades:
        #   - Full exit HOLD_A: SELL 500 @ exec $102 (close $104, comm $25) -> hold P&L=500*4=$2000, trade P&L=-500*(104-102)=-$1000 -> gross $1000, net $975
        #   - Partial trim TRIM_B: SELL 200 @ exec $99 (close $98, comm $10) -> hold P&L=500*(-2)=-$1000, trade P&L=-200*(98-99)=+$200 -> gross -$800, net -$810
        #   - New entry NEW_C: BUY 600 @ exec $51 (close $53, comm $15) -> hold P&L=$0, trade P&L=+600*(53-51)=+$1200 -> gross $1200, net $1185
        # Total Gross P&L = +$1,400 (+1.40%), Total Fees = $50 (-0.05%), Net P&L = +$1,350 (+1.35%), End NAV = $101,350.
        batch = AttributionInputBatch(
            batch_id="SCEN-6-REBAL",
            portfolio_id="P_REB",
            reporting_currency="USD",
            as_of_utc="2026-01-06T00:00:00Z",
            nav_snapshots=[DailyNavSnapshot("N1", "2026-01-05", "P_REB", "USD", 100000.0, 101350.0)],
            positions=[
                PositionSnapshot("POS-A", "2026-01-05", "P_REB", "S1", "HOLD_A", "TECH", "EQUITY", "USD", 500.0, 0.0, 100.0, 104.0),
                PositionSnapshot("POS-B", "2026-01-05", "P_REB", "S1", "TRIM_B", "FINANCIALS", "EQUITY", "USD", 500.0, 300.0, 100.0, 98.0),
                PositionSnapshot("POS-C", "2026-01-05", "P_REB", "S2", "NEW_C", "HEALTHCARE", "EQUITY", "USD", 0.0, 600.0, 50.0, 53.0),
            ],
            trades=[
                TradeExecution("TRD-EXIT-A", "O1", "2026-01-05", "2026-01-05T14:30:00Z", "P_REB", "S1", "HOLD_A", "TECH", "SELL", 500.0, 500.0, "USD", 102.5, 102.2, 102.0, 104.0, commission_local=25.0),
                TradeExecution("TRD-TRIM-B", "O2", "2026-01-05", "2026-01-05T15:00:00Z", "P_REB", "S1", "TRIM_B", "FINANCIALS", "SELL", 200.0, 200.0, "USD", 99.2, 99.1, 99.0, 98.0, commission_local=10.0),
                TradeExecution("TRD-NEW-C", "O3", "2026-01-05", "2026-01-05T15:15:00Z", "P_REB", "S2", "NEW_C", "HEALTHCARE", "BUY", 600.0, 600.0, "USD", 50.5, 50.8, 51.0, 53.0, commission_local=15.0),
            ],
        )
        rpt = AttributionPipeline(AttributionConfig()).run(batch)
        self.assertEqual(rpt.reconciliation.status, "RECONCILED")
        self.assertAlmostEqual(rpt.returns.cumulative_gross_return, 0.0140, places=10)
        self.assertAlmostEqual(rpt.returns.cumulative_net_return, 0.0135, places=10)
        self.assertAlmostEqual(rpt.reconciliation.linked_net_residual, 0.0, places=10)
        # Every instrument contribution links directly to both position_ids and trade_ids
        by_inst = {x.group_key: x for x in rpt.contributions.by_instrument}
        self.assertEqual(by_inst["HOLD_A"].trade_ids, ["TRD-EXIT-A"])
        self.assertEqual(by_inst["TRIM_B"].trade_ids, ["TRD-TRIM-B"])
        self.assertEqual(by_inst["NEW_C"].trade_ids, ["TRD-NEW-C"])
        self.assertEqual(rpt.execution_costs.trade_count, 3)

    def test_scenario_7_rounding_residual(self) -> None:
        """Rounding residual: sub-tolerance residual reconciles & displays explicitly; breach triggers ERROR."""
        # Case A: Custodian rounds End NAV by +$0.03 on $1,000,000 (+0.0003 bps < 0.01 bps tolerance)
        batch_within = AttributionInputBatch(
            batch_id="SCEN-7-WITHIN",
            portfolio_id="P_RND",
            reporting_currency="USD",
            as_of_utc="2026-01-06T00:00:00Z",
            nav_snapshots=[DailyNavSnapshot("N1", "2026-01-05", "P_RND", "USD", 1000000.0, 1010000.03)],
            positions=[
                PositionSnapshot(
                    "P1", "2026-01-05", "P_RND", "S1", "EQ1", "TECH", "EQUITY", "USD",
                    10000.0, 10000.0, 100.0, 101.0,
                    end_market_value_base=1010000.03,
                )
            ],
        )
        rpt_ok = AttributionPipeline(AttributionConfig(return_tolerance=1e-6, return_tolerance_bps=0.01)).run(batch_within)
        self.assertEqual(rpt_ok.reconciliation.status, "RECONCILED")
        self.assertAlmostEqual(rpt_ok.reconciliation.linked_net_residual, 3e-8, places=12)
        self.assertAlmostEqual(rpt_ok.reconciliation.total_valuation_rounding_delta_base, 0.03, places=6)
        self.assertTrue(any(e.code == "ROUNDING_RESIDUAL_WITHIN_TOLERANCE" for e in rpt_ok.reconciliation.exceptions))

        # Case B: Unexplained discrepancy of +$500 on $1,000,000 (+5.0 bps >> 0.01 bps tolerance) -> UNRECONCILED ERROR
        batch_breach = AttributionInputBatch(
            batch_id="SCEN-7-BREACH",
            portfolio_id="P_RND",
            reporting_currency="USD",
            as_of_utc="2026-01-06T00:00:00Z",
            nav_snapshots=[DailyNavSnapshot("N1", "2026-01-05", "P_RND", "USD", 1000000.0, 1010500.0)],
            positions=[
                PositionSnapshot("P1", "2026-01-05", "P_RND", "S1", "EQ1", "TECH", "EQUITY", "USD", 10000.0, 10000.0, 100.0, 101.0)
            ],
        )
        rpt_bad = AttributionPipeline(AttributionConfig(return_tolerance=1e-6, return_tolerance_bps=0.01)).run(batch_breach)
        self.assertEqual(rpt_bad.reconciliation.status, "UNRECONCILED")
        self.assertAlmostEqual(rpt_bad.reconciliation.linked_net_residual, 0.0005, places=10)
        self.assertTrue(
            any(e.code == "RESIDUAL_EXCEEDS_TOLERANCE" and e.severity == "ERROR" for e in rpt_bad.reconciliation.exceptions)
        )


if __name__ == "__main__":
    unittest.main()
