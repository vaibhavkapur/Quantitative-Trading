"""Unit tests for configuration guardrails, domain validation, and return/benchmark calculations."""

from __future__ import annotations

import unittest

from pae.config import AttributionConfig
from pae.domain.models import (
    AttributionInputBatch,
    BenchmarkObservation,
    CashFlow,
    DailyNavSnapshot,
    FxRateObservation,
)
from pae.domain.validation import validate_input_batch
from pae.engine.returns import FxConverter, compute_carino_weights, compute_returns_and_benchmark
from pae.errors import ConfigurationError, OrderExecutionProhibitedError, ValidationError


class TestConfigAndValidation(unittest.TestCase):
    def test_order_execution_strictly_prohibited(self) -> None:
        with self.assertRaises(OrderExecutionProhibitedError):
            AttributionConfig(order_execution_enabled=True)

    def test_gross_net_separation_cannot_be_disabled(self) -> None:
        with self.assertRaises(ConfigurationError):
            AttributionConfig(gross_vs_net_strict_separation=False)

    def test_live_environment_mode_rejected_at_mvp(self) -> None:
        with self.assertRaises(ConfigurationError):
            AttributionConfig(environment_mode="LIVE_PRODUCTION")

    def test_config_hash_deterministic(self) -> None:
        c1 = AttributionConfig()
        c2 = AttributionConfig()
        self.assertEqual(c1.config_hash(), c2.config_hash())
        c3 = AttributionConfig(brinson_model="BRINSON_HOOD_BEEBOWER")
        self.assertNotEqual(c1.config_hash(), c3.config_hash())

    def test_validation_rejects_duplicate_nav_dates_and_non_utc(self) -> None:
        nav1 = DailyNavSnapshot("N1", "2026-01-05", "P1", "USD", 100.0, 101.0)
        nav2 = DailyNavSnapshot("N2", "2026-01-05", "P1", "USD", 101.0, 102.0)
        batch = AttributionInputBatch("B1", "P1", "USD", "2026-01-06T00:00:00Z", nav_snapshots=[nav1, nav2])
        with self.assertRaises(ValidationError):
            validate_input_batch(batch)

        bad_ts = AttributionInputBatch("B2", "P1", "USD", "2026-01-06 00:00:00", nav_snapshots=[nav1])
        with self.assertRaises(ValidationError):
            validate_input_batch(bad_ts)

    def test_lookahead_availability_emits_warning(self) -> None:
        nav = DailyNavSnapshot(
            "N1",
            "2026-01-05",
            "P1",
            "USD",
            100.0,
            101.0,
            as_of_utc="2026-01-05T21:00:00Z",
            availability_utc="2026-01-06T12:00:00Z",
        )
        batch = AttributionInputBatch("B1", "P1", "USD", "2026-01-05T22:00:00Z", nav_snapshots=[nav])
        excs = validate_input_batch(batch)
        self.assertTrue(any(e.code == "LOOKAHEAD_VIOLATION" for e in excs))


class TestReturnAndCarinoMath(unittest.TestCase):
    def test_carino_weights_link_exact_cumulative_return(self) -> None:
        pairs = [("2026-01-05", 0.04), ("2026-01-06", -0.025), ("2026-01-07", 0.031)]
        w = compute_carino_weights(pairs)
        linked = sum(w[d] * r for d, r in pairs)
        exact = (1.04 * 0.975 * 1.031) - 1.0
        self.assertAlmostEqual(linked, exact, places=12)

    def test_carino_weights_zero_return_limit(self) -> None:
        pairs = [("2026-01-05", 0.0), ("2026-01-06", 0.0)]
        w = compute_carino_weights(pairs)
        self.assertAlmostEqual(w["2026-01-05"], 1.0, places=12)
        self.assertAlmostEqual(w["2026-01-06"], 1.0, places=12)

    def test_benchmark_currency_alignment_with_fx_conversion(self) -> None:
        # Portfolio in USD, Benchmark in EUR with +2% local return and EUR/USD rising 1.00 -> 1.05 (+5%)
        # Aligned USD benchmark return = 1.02 * 1.05 - 1 = 0.071 (+7.1%)
        batch = AttributionInputBatch(
            batch_id="B_BM_FX",
            portfolio_id="P1",
            reporting_currency="USD",
            as_of_utc="2026-01-06T00:00:00Z",
            benchmark_id="BM_EUR",
            nav_snapshots=[DailyNavSnapshot("N1", "2026-01-05", "P1", "USD", 1000.0, 1080.0)],
            fx_rates=[
                FxRateObservation("EUR/USD", "EUR", "USD", "2026-01-02", 1.00, "2026-01-02T21:00:00Z", "2026-01-02T21:05:00Z"),
                FxRateObservation("EUR/USD", "EUR", "USD", "2026-01-05", 1.05, "2026-01-05T21:00:00Z", "2026-01-05T21:05:00Z"),
            ],
            benchmark=[BenchmarkObservation("BM_EUR", "2026-01-05", "EUR", 0.02)],
        )
        cfg = AttributionConfig()
        excs = []
        fx = FxConverter(batch.fx_rates, "USD")
        ret = compute_returns_and_benchmark(batch, cfg, fx, excs)
        self.assertAlmostEqual(ret.cumulative_benchmark_return, 0.071, places=12)
        self.assertAlmostEqual(ret.cumulative_active_net_return, 0.08 - 0.071, places=12)
        self.assertEqual(len(excs), 0)


if __name__ == "__main__":
    unittest.main()
