"""Unit tests for domain contracts, snapshot publishing, numerical linalg, and projections."""

from __future__ import annotations

import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dcae.domain import (
    EngineConfig,
    FundSnapshot,
    PositionSnapshot,
    ReturnObservation,
    StrategySnapshot,
)
from dcae.errors import PointInTimeViolationError, ValidationError
from dcae.fixtures import build_baseline_snapshot
from dcae.math import (
    ConvexConstraintSet,
    LinearCap,
    apply_correlation_stress,
    check_psd,
    dykstra_project,
    eigen_cache,
    jacobi_eigen_symmetric,
    nearest_psd_spectral,
    project_l1_ball,
    quad_form,
)
from dcae.snapshots import SnapshotPublisher


class TestDomainContracts(unittest.TestCase):
    def test_baseline_snapshot_validates_and_hashes_deterministically(self) -> None:
        snap1 = build_baseline_snapshot(seed=42)
        snap2 = build_baseline_snapshot(seed=42)
        snap1.validate()
        self.assertEqual(snap1.content_hash(), snap2.content_hash())
        roundtrip = FundSnapshot.from_dict(snap1.to_dict())
        self.assertEqual(snap1.content_hash(), roundtrip.content_hash())

    def test_non_utc_or_inverted_availability_timestamp_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ReturnObservation(
                event_at_utc="2025-01-02T21:00:00",  # missing Z / offset
                available_at_utc="2025-01-02T22:00:00Z",
                nav_usd=100.0,
                period_return=0.01,
            ).validate()

        with self.assertRaises(ValidationError):
            ReturnObservation(
                event_at_utc="2025-01-02T22:00:00Z",
                available_at_utc="2025-01-02T21:00:00Z",  # available before event
                nav_usd=100.0,
                period_return=0.01,
            ).validate()

    def test_position_reconciliation_enforced(self) -> None:
        bad = PositionSnapshot(
            as_of_utc="2025-03-01T21:00:00Z",
            updated_at_utc="2025-03-01T20:00:00Z",
            currency="USD",
            total_nav_usd=1_000_000.0,
            cash_usd=200_000.0,
            strategy_notional_usd={"s1": 500_000.0},  # sum = 700k != 1M
        )
        with self.assertRaises(ValidationError):
            bad.validate()

    def test_live_execution_cannot_be_enabled_in_config(self) -> None:
        cfg = EngineConfig(live_execution_enabled=True)
        with self.assertRaises(ValidationError):
            cfg.validate()

    def test_point_in_time_filter_drops_future_unavailable_returns(self) -> None:
        snap = build_baseline_snapshot(seed=101, n_days=30)
        # Inject one observation with available_at_utc AFTER snapshot.as_of_utc
        s0 = snap.strategies[0]
        future_obs = ReturnObservation(
            event_at_utc="2025-12-31T21:00:00Z",
            available_at_utc="2025-12-31T22:00:00Z",
            nav_usd=999.0,
            period_return=0.50,
        )
        mod_s0 = replace(s0, returns=s0.returns + (future_obs,))
        mod_snap = replace(snap, strategies=(mod_s0,) + snap.strategies[1:])
        series, dropped = SnapshotPublisher.point_in_time_filter(mod_snap)
        self.assertEqual(dropped[s0.strategy_id], 1)
        self.assertEqual(len(series[s0.strategy_id]), len(s0.returns))

    def test_future_position_timestamp_raises_pit_violation(self) -> None:
        snap = build_baseline_snapshot(seed=101, n_days=30)
        future_pos = replace(snap.positions, updated_at_utc="2099-01-01T00:00:00Z")
        bad_snap = replace(snap, positions=future_pos)
        pub = SnapshotPublisher(EngineConfig())
        with self.assertRaises(PointInTimeViolationError):
            pub.validate_and_publish(bad_snap)


class TestNumericalMathAndProjections(unittest.TestCase):
    def test_jacobi_eigen_and_psd_repair(self) -> None:
        non_psd = [
            [1.0, 0.9, 0.9],
            [0.9, 1.0, -0.9],
            [0.9, -0.9, 1.0],
        ]
        is_psd, min_eig, eigs, V = check_psd(non_psd)
        self.assertFalse(is_psd)
        self.assertLess(min_eig, -0.5)
        # Check orthonormality V^T V = I
        for i in range(3):
            for j in range(3):
                dot_ij = sum(V[k][i] * V[k][j] for k in range(3))
                self.assertAlmostEqual(dot_ij, 1.0 if i == j else 0.0, places=9)

        repaired, orig_min, delta, clipped, _ = nearest_psd_spectral(non_psd, min_eigenvalue=1e-6)
        self.assertLess(orig_min, 0.0)
        self.assertGreater(delta, 0.0)
        is_psd_after, min_after, _, _ = check_psd(repaired)
        self.assertTrue(is_psd_after)
        self.assertGreaterEqual(min_after, 1e-6 - 1e-10)

    def test_correlation_stress_increases_portfolio_variance(self) -> None:
        cov = [
            [0.010, 0.001, -0.001],
            [0.001, 0.015, 0.000],
            [-0.001, 0.000, 0.012],
        ]
        stressed_cov, stressed_corr = apply_correlation_stress(cov, blend=0.5, stress_floor=0.75)
        w = [0.3, 0.3, 0.3]
        self.assertGreater(quad_form(w, stressed_cov), quad_form(w, cov))
        self.assertGreater(stressed_corr[0][1], 0.35)

    def test_l1_ball_projection_exact(self) -> None:
        v = [0.4, -0.3, 0.2, -0.1]  # L1 norm = 1.0
        proj = project_l1_ball(v, radius=0.4)
        self.assertAlmostEqual(sum(abs(x) for x in proj), 0.4, places=10)
        # Signs preserved
        self.assertGreaterEqual(proj[0], 0.0)
        self.assertLessEqual(proj[1], 0.0)

    def test_dykstra_projects_simultaneously_onto_box_sum_group_turnover_vol(self) -> None:
        cov = [
            [0.04, 0.01, 0.005],
            [0.01, 0.03, 0.005],
            [0.005, 0.005, 0.02],
        ]
        ev, V = eigen_cache(cov)
        cset = ConvexConstraintSet(
            lower=(0.0, 0.0, 0.0),
            upper=(0.45, 0.45, 0.45),
            sum_min=0.50,
            sum_max=0.95,
            w0=(0.30, 0.30, 0.30),
            max_one_way_turnover=0.15,
            linear_caps=(LinearCap("grp_ab", (1.0, 1.0, 0.0), 0.55),),
            cov_eigvals=ev,
            cov_eigvecs=V,
            max_variance=0.11 ** 2,
        )
        proj, viol, _ = dykstra_project([0.8, 0.5, 0.1], cset, max_iters=300, tol=1e-9)
        self.assertLessEqual(viol, 1e-7)
        self.assertLessEqual(proj[0] + proj[1], 0.55 + 1e-6)
        self.assertLessEqual(math.sqrt(quad_form(proj, cov)), 0.11 + 1e-6)
        self.assertLessEqual(
            0.5 * sum(abs(proj[i] - cset.w0[i]) for i in range(3)),
            0.15 + 1e-6,
        )


if __name__ == "__main__":
    unittest.main()
