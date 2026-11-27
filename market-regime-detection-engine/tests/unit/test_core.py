import io
import math
import unittest
from dataclasses import replace
from statistics import mean
from statistics import stdev

from regime.domain import ContractError, parse_bars
from regime.engine import StabilityGate
from regime.evaluation import strategy
from regime.features import TrailingFeatures
from regime.models import baseline
from tests.common import BARS, CFG, SOURCE


class Contracts(unittest.TestCase):
    def test_rejects_invalid_values_and_timestamps(self):
        for changes in (dict(close=0), dict(benchmark_close=float("nan")), dict(spread_bps=-1),
                        dict(available_at="bad"), dict(event_at=BARS[1].event_at)):
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                replace(BARS[0], **changes)
        with self.assertRaisesRegex(ContractError, "line 2"):
            list(parse_bars(io.StringIO(SOURCE.read_text().splitlines()[0] + "\nSYNTH-USD,")))
        for changes in (dict(window=2), dict(confidence_floor=1.01), dict(cost_bps=-1)):
            with self.assertRaises(ContractError):
                replace(CFG, **changes)

    def test_order_and_series_fail_before_advancing_state(self):
        stream = TrailingFeatures(CFG)
        stream.observe(BARS[0])
        for bad in (BARS[0], replace(BARS[1], series="other"), replace(BARS[0], available_at=BARS[1].available_at)):
            with self.assertRaises(ContractError):
                stream.observe(bad)
        self.assertEqual(stream.observe(BARS[1]), (None, "warmup"))


class Features(unittest.TestCase):
    def test_trailing_formula_and_units(self):
        stream = TrailingFeatures(replace(CFG, window=3))
        for bar in BARS[:3]:
            self.assertEqual(stream.observe(bar), (None, "warmup"))
        x, status = stream.observe(BARS[3])
        self.assertIsNone(status)
        a = [math.log(y.close / x.close) for x, y in zip(BARS[:3], BARS[1:4])]
        b = [math.log(y.benchmark_close / x.benchmark_close) for x, y in zip(BARS[:3], BARS[1:4])]
        expected = (stdev(a) * math.sqrt(252), mean(a) * math.sqrt(3) / stdev(a),
                    mean(r.spread_bps for r in BARS[1:4]),
                    sum((i - mean(a)) * (j - mean(b)) for i, j in zip(a, b)) / (2 * stdev(a) * stdev(b)))
        for actual, target in zip(x, expected):
            self.assertAlmostEqual(actual, target)

    def test_flat_and_gapped_inputs(self):
        stream = TrailingFeatures(replace(CFG, window=3))
        for bar in BARS[:4]:
            x, reason = stream.observe(replace(bar, close=100.))
        self.assertEqual((x, reason), (None, "flat_window"))
        self.assertEqual(stream.observe(BARS[20]), (None, "gap"))
        self.assertIsNone(stream.observe(BARS[21])[0])

    def test_baseline_and_immediate_uncertainty(self):
        gate = StabilityGate(replace(CFG, confirm_bars=1))
        for x, label in (((.08, 3, 3, .1), "rising"), ((.08, -3, 3, .1), "falling"),
                         ((.5, -2, 4, .5), "stressed")):
            scores = baseline(x, CFG)
            self.assertAlmostEqual(sum(scores), 1)
            self.assertEqual(gate.step(scores)["label"], label)
        decision = gate.step([.49, .48, .03])
        self.assertEqual((decision["label"], decision["reason"]), ("unknown", "low_confidence"))


class Valuation(unittest.TestCase):
    def test_delayed_signal_costs_and_missing_mark(self):
        bars = [replace(bar, close=px) for bar, px in zip(BARS, [100, 200, 220, None, 242])]
        records = [dict(available_at=b.available_at, rule={"label": "rising" if i < 2 else "falling"})
                   for i, b in enumerate(bars)]
        # A signal arriving exactly at a close cannot fill at that close.
        records[0]["available_at"] = bars[1].event_at
        result = strategy(bars, records, "rule", 10)
        self.assertAlmostEqual(result["gross_return"], .1)
        self.assertAlmostEqual(result["net_return"], .999 * 1.1 * .999 - 1)
        self.assertEqual(result["one_way_turnover"], 2)
        self.assertEqual(result["observed_marks"], 4)
