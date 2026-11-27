"""Scenario tests covering all 7 required failure & stress paths in plan.md Section 5."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dcae.allocators import ConstrainedAllocator, InverseVolatilityAllocator
from dcae.audit import SQLiteAuditStore
from dcae.domain import AllocationStatus, EngineConfig
from dcae.errors import (
    InsufficientHistoryError,
    NonPSDCovarianceError,
    StaleDataError,
    ZeroVolatilityError,
)
from dcae.estimators import PointInTimeEstimator
from dcae.fixtures import build_scenario_fixtures
from dcae.proposals import ProposalService
from dcae.verification import ConstraintVerifier


class TestRequiredFailureScenarios(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = EngineConfig()
        self.fixtures = build_scenario_fixtures()

    def test_scenario_1_non_psd_covariance(self) -> None:
        snap = self.fixtures["non_psd_covariance"]

        # Policy A: reject
        reject_cfg = replace(self.cfg, non_psd_policy="reject")
        with self.assertRaises(NonPSDCovarianceError) as ctx:
            PointInTimeEstimator(reject_cfg).estimate(snap)
        self.assertLess(ctx.exception.context["min_eigenvalue"], 0.0)

        # Policy B: repair
        est = PointInTimeEstimator(self.cfg).estimate(snap)
        self.assertTrue(est.psd_repaired)
        self.assertLess(est.original_min_eigenvalue, 0.0)
        self.assertGreater(est.repaired_min_eigenvalue, 0.0)
        self.assertGreater(est.psd_repair_frobenius_delta, 0.0)

        dec = ConstrainedAllocator(self.cfg).allocate(snap, estimation=est)
        rep = ConstraintVerifier(self.cfg).verify(snapshot=snap, decision=dec, estimation=est)
        self.assertTrue(rep.feasible)

    def test_scenario_2_zero_volatility(self) -> None:
        snap = self.fixtures["zero_volatility"]

        # Reject mode raises ZeroVolatilityError
        with self.assertRaises(ZeroVolatilityError):
            PointInTimeEstimator(
                replace(self.cfg, zero_volatility_policy="reject")
            ).estimate(snap)

        # Exclude mode safely prevents divide-by-zero in inverse-vol & excludes from new capital
        est = PointInTimeEstimator(self.cfg).estimate(snap)
        self.assertIn("strat_vol_carry", est.excluded_reasons)

        iv_dec = InverseVolatilityAllocator(self.cfg).allocate(snap, estimation=est)
        self.assertEqual(iv_dec.target_weights["strat_vol_carry"], 0.0)
        self.assertAlmostEqual(
            sum(iv_dec.target_weights.values()) + iv_dec.cash_weight, 1.0, places=9
        )

        cmv_dec = ConstrainedAllocator(self.cfg).allocate(snap, estimation=est)
        rep = ConstraintVerifier(self.cfg).verify(snapshot=snap, decision=cmv_dec, estimation=est)
        self.assertTrue(rep.feasible)
        self.assertLessEqual(
            cmv_dec.target_weights["strat_vol_carry"],
            cmv_dec.current_weights["strat_vol_carry"] + 1e-6,
        )

    def test_scenario_3_insufficient_history(self) -> None:
        snap = self.fixtures["insufficient_history"]

        # Reject mode raises InsufficientHistoryError
        with self.assertRaises(InsufficientHistoryError):
            PointInTimeEstimator(
                replace(self.cfg, insufficient_history_policy="reject")
            ).estimate(snap)

        # Exclude mode excludes short-history strategy from fresh capital
        est = PointInTimeEstimator(self.cfg).estimate(snap)
        self.assertIn("strat_eq_momentum", est.excluded_reasons)
        self.assertEqual(est.observation_counts["strat_eq_momentum"], 5)

        dec = ConstrainedAllocator(self.cfg).allocate(snap, estimation=est)
        rep = ConstraintVerifier(self.cfg).verify(snapshot=snap, decision=dec, estimation=est)
        self.assertTrue(rep.feasible)
        self.assertLessEqual(
            dec.target_weights["strat_eq_momentum"],
            dec.current_weights["strat_eq_momentum"] + 1e-6,
        )

    def test_scenario_4_infeasible_caps_safe_fallback(self) -> None:
        snap = self.fixtures["infeasible_caps"]

        # Default policy: no_change fallback preserving sum = 1
        dec_nc = ConstrainedAllocator(self.cfg).allocate(snap)
        self.assertEqual(dec_nc.status, AllocationStatus.FALLBACK_NO_CHANGE)
        self.assertEqual(dec_nc.fallback_applied, "no_change")
        self.assertTrue(any("infeasible_caps" in r for r in dec_nc.infeasibility_reasons))
        self.assertEqual(dec_nc.target_weights, dec_nc.current_weights)
        self.assertAlmostEqual(
            sum(dec_nc.target_weights.values()) + dec_nc.cash_weight, 1.0, places=9
        )

        # Cash-only policy: 100% cash fallback
        dec_cash = ConstrainedAllocator(
            replace(self.cfg, fallback_policy="cash_only")
        ).allocate(snap)
        self.assertEqual(dec_cash.status, AllocationStatus.FALLBACK_CASH)
        self.assertEqual(dec_cash.fallback_applied, "cash_fallback")
        self.assertAlmostEqual(dec_cash.cash_weight, 1.0, places=10)
        self.assertTrue(all(v == 0.0 for v in dec_cash.target_weights.values()))

    def test_scenario_5_stale_positions(self) -> None:
        snap = self.fixtures["stale_positions"]
        store = SQLiteAuditStore(":memory:")
        try:
            service = ProposalService(self.cfg, store)
            with self.assertRaises(StaleDataError) as ctx:
                service.create_proposal(snap, proposer_id="pm_1")
            self.assertEqual(ctx.exception.code, "STALE_DATA")
            self.assertGreater(ctx.exception.context["position_age_hours"], 24.0)
        finally:
            store.close()

    def test_scenario_6_turnover_limit(self) -> None:
        snap = self.fixtures["turnover_limit"]
        tight_cfg = replace(
            self.cfg,
            constraints=replace(self.cfg.constraints, max_one_way_turnover=0.05),
        )
        est = PointInTimeEstimator(tight_cfg).estimate(snap)
        dec = ConstrainedAllocator(tight_cfg).allocate(snap, estimation=est)
        rep = ConstraintVerifier(tight_cfg).verify(snapshot=snap, decision=dec, estimation=est)
        self.assertTrue(rep.feasible, msg=f"Violations: {rep.hard_violations}")
        self.assertLessEqual(
            dec.objective.one_way_turnover,
            0.05 + tight_cfg.constraints.constraint_tolerance,
        )

    def test_scenario_7_correlated_drawdowns(self) -> None:
        snap = self.fixtures["correlated_drawdowns"]
        est = PointInTimeEstimator(self.cfg).estimate(snap)
        self.assertTrue(est.correlation_stress_active)
        self.assertTrue(any("Correlated drawdown" in w for w in est.warnings))

        dec = ConstrainedAllocator(self.cfg).allocate(snap, estimation=est)
        rep = ConstraintVerifier(self.cfg).verify(snapshot=snap, decision=dec, estimation=est)
        self.assertTrue(rep.feasible, msg=f"Violations: {rep.hard_violations}")
        self.assertLessEqual(
            dec.objective.stressed_volatility_annual,
            (self.cfg.constraints.max_stressed_annualized_volatility or 1.0) + 1e-6,
        )


if __name__ == "__main__":
    unittest.main()
