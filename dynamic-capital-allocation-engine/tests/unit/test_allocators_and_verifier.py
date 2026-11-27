"""Unit tests for baseline allocators, constrained allocator, and independent verifier."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dcae.allocators import (
    ConstrainedAllocator,
    EqualWeightAllocator,
    InverseVolatilityAllocator,
    RiskBudgetAllocator,
)
from dcae.domain import AllocationStatus, ConstraintSpec, EngineConfig
from dcae.estimators import PointInTimeEstimator
from dcae.fixtures import build_baseline_snapshot
from dcae.verification import ConstraintVerifier


class TestAllocatorsAndVerifier(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = EngineConfig()
        self.snap = build_baseline_snapshot(seed=20260409, n_days=60)
        self.est = PointInTimeEstimator(self.cfg).estimate(self.snap)

    def test_equal_weight_reconciles_and_respects_cash_min(self) -> None:
        dec = EqualWeightAllocator(self.cfg).allocate(self.snap, estimation=self.est)
        self.assertEqual(dec.status, AllocationStatus.OPTIMAL)
        self.assertAlmostEqual(
            sum(dec.target_weights.values()) + dec.cash_weight, 1.0, places=10
        )
        self.assertAlmostEqual(dec.cash_weight, self.cfg.constraints.cash_reserve_min, places=10)
        expected_w = (1.0 - self.cfg.constraints.cash_reserve_min) / len(self.snap.strategies)
        for w in dec.target_weights.values():
            self.assertAlmostEqual(w, expected_w, places=10)

    def test_inverse_volatility_overweights_lower_vol_strategy(self) -> None:
        dec = InverseVolatilityAllocator(self.cfg).allocate(self.snap, estimation=self.est)
        self.assertEqual(dec.status, AllocationStatus.OPTIMAL)
        self.assertAlmostEqual(
            sum(dec.target_weights.values()) + dec.cash_weight, 1.0, places=10
        )
        # Rates RV (6% vol) should get higher weight than Vol Carry (16% vol)
        self.assertGreater(
            dec.target_weights["strat_fi_rv"],
            dec.target_weights["strat_vol_carry"],
        )

    def test_risk_budget_matches_target_risk_contributions(self) -> None:
        dec = RiskBudgetAllocator(self.cfg).allocate(self.snap, estimation=self.est)
        self.assertEqual(dec.status, AllocationStatus.OPTIMAL)
        self.assertLess(dec.diagnostics["max_risk_contribution_error"], 1e-6)
        self.assertAlmostEqual(
            sum(dec.target_weights.values()) + dec.cash_weight, 1.0, places=10
        )

    def test_constrained_allocator_passes_all_independent_verifier_checks(self) -> None:
        dec = ConstrainedAllocator(self.cfg).allocate(self.snap, estimation=self.est)
        self.assertIn(
            dec.status, (AllocationStatus.OPTIMAL, AllocationStatus.CONVERGED_APPROX)
        )
        report = ConstraintVerifier(self.cfg).verify(
            snapshot=self.snap, decision=dec, estimation=self.est
        )
        self.assertTrue(report.feasible, msg=f"Violations: {report.hard_violations}")
        self.assertLessEqual(report.reconciliation_error, 1e-9)
        self.assertGreater(len(report.checks), 15)
        # Decision exposes inputs and full objective decomposition
        self.assertIn("expected_returns_annual", dec.inputs_summary)
        self.assertIn("net_objective", dec.objective.to_dict())

    def test_high_transaction_costs_create_no_trade_band(self) -> None:
        # Increase transaction costs to 500 bps across all strategies -> optimizer should barely trade
        high_cost_strats = tuple(
            replace(
                s,
                exposure=replace(s.exposure, transaction_cost_bps=500.0),
            )
            for s in self.snap.strategies
        )
        hc_snap = replace(self.snap, strategies=high_cost_strats)
        dec_low = ConstrainedAllocator(self.cfg).allocate(self.snap)
        dec_high = ConstrainedAllocator(self.cfg).allocate(hc_snap)
        self.assertLess(dec_high.objective.one_way_turnover, dec_low.objective.one_way_turnover)

    def test_independent_verifier_detects_tampered_allocations(self) -> None:
        dec = ConstrainedAllocator(self.cfg).allocate(self.snap, estimation=self.est)
        verifier = ConstraintVerifier(self.cfg)

        # Tamper 1: Break allocation sum reconciliation
        bad_sum = replace(dec, cash_weight=dec.cash_weight + 0.05)
        rep1 = verifier.verify(snapshot=self.snap, decision=bad_sum, estimation=self.est)
        self.assertFalse(rep1.feasible)
        self.assertTrue(
            any(v.name == "allocation_sum_reconciliation" for v in rep1.hard_violations)
        )

        # Tamper 2: Breach strategy max_weight & turnover
        bad_weights = dict(dec.target_weights)
        bad_weights["strat_eq_statarb_us"] = 0.80
        bad_weights["strat_eq_momentum"] = 0.0
        bad_weights["strat_macro_cta"] = 0.0
        bad_weights["strat_vol_carry"] = 0.0
        bad_weights["strat_fi_rv"] = 0.10
        bad_cap = replace(dec, target_weights=bad_weights, cash_weight=0.10)
        rep2 = verifier.verify(snapshot=self.snap, decision=bad_cap, estimation=self.est)
        self.assertFalse(rep2.feasible)
        vnames = {v.name for v in rep2.hard_violations}
        self.assertIn("strategy_max_weight", vnames)
        self.assertIn("one_way_turnover_limit", vnames)


if __name__ == "__main__":
    unittest.main()
