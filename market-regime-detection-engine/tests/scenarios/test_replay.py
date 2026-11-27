import unittest
from copy import deepcopy
from dataclasses import asdict, replace
from statistics import fmean

from regime.domain import ContractError
from regime.engine import OnlineEngine, StabilityGate
from regime.evaluation import drift
from regime.features import TrailingFeatures
from regime.models import baseline
from tests.common import BARS, CFG, FAST, TEST, TRAIN


class Scenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fitted, cls.fit_rows = OnlineEngine.fit(TRAIN, FAST)

    def engine(self):
        return deepcopy(self.fitted)

    def test_future_smoothing_leak(self):
        short, full = self.engine(), self.engine()
        prefix = [short.observe(b) for b in TEST[:50]]
        changed = [full.observe(b if i < 50 else replace(b, spread_bps=5000)) for i, b in enumerate(TEST)]
        self.assertEqual(prefix, changed[:50])
        self.assertEqual(changed[-1]["hmm"]["label"], "unknown")

    def test_noisy_threshold_oscillation(self):
        gate = StabilityGate(CFG)
        up, down = (baseline((.07, t, 3, .4), CFG) for t in (1., -1.))
        for _ in range(CFG.confirm_bars):
            gate.step(up)
        for score in (down, up) * 10:
            self.assertEqual(gate.step(score)["label"], "rising")
        labels = [gate.step(down)["label"] for _ in range(CFG.confirm_bars)]
        self.assertEqual(labels, ["rising", "rising", "falling"])

    def test_missing_prices_reset_and_recovery(self):
        engine = self.engine()
        records = [engine.observe(b) for b in TEST]
        i = next(i for i, b in enumerate(TEST) if b.close is None)
        self.assertEqual(records[i]["feature_status"], "missing_input")
        for row in records[i:i+CFG.window+1]:
            self.assertIsNone(row["features"])
            self.assertEqual((row["rule"]["label"], row["hmm"]["label"]), ("unknown", "unknown"))
        self.assertEqual(records[i+CFG.window+1]["feature_status"], "ready")
        self.assertTrue(any(row["hmm"]["label"] != "unknown" for row in records[i+CFG.window+1:]))

    def test_structural_shift_abstains_and_flags_drift(self):
        engine = self.engine()
        rows = [engine.observe(replace(b, spread_bps=200)) for b in TEST[:40]]
        self.assertTrue(all(r["hmm"]["reason"] == "outside_training_support" for r in rows[-20:]))
        self.assertTrue(all(r["hmm"]["label"] == "unknown" for r in rows[-20:]))
        self.assertTrue(drift(rows, engine.model)[-1]["max_abs_shift_over_two"])

    def test_scaler_fit_leakage(self):
        engine = self.engine()
        before = asdict(engine.model)
        for j, mu in enumerate(engine.model.center):
            self.assertAlmostEqual(mu, fmean(x[j] for x in self.fit_rows))
        for bar in TEST:
            engine.observe(replace(bar, spread_bps=1e5))
        self.assertEqual(asdict(engine.model), before)
        with self.assertRaisesRegex(ContractError, "after cutoff"):
            OnlineEngine.fit(TRAIN + [TEST[0]], FAST)
        # An event within training whose publication is delayed past cutoff is held out.
        late = replace(TRAIN[-1], available_at=TEST[0].available_at)
        with self.assertRaisesRegex(ContractError, "after cutoff"):
            OnlineEngine.fit(TRAIN[:-1] + [late], FAST)

    def test_cold_start(self):
        stream = TrailingFeatures(CFG)
        self.assertTrue(all(stream.observe(b)[0] is None for b in BARS[:CFG.window]))
        self.assertIsNotNone(stream.observe(BARS[CFG.window])[0])
        with self.assertRaisesRegex(ContractError, "insufficient"):
            OnlineEngine.fit(BARS[:CFG.window], FAST)
        engine = self.engine()
        self.assertEqual(engine.observe(TEST[0])["hmm"]["label"], "unknown")

    def test_online_probabilities_and_state_names(self):
        engine = self.engine()
        model = engine.model
        self.assertTrue(all(abs(sum(row)-1) < 1e-12 for row in model.transition))
        self.assertGreaterEqual(model.means[0][1], model.means[1][1])
        for bar in TEST:
            hmm = engine.observe(bar)["hmm"]
            if hmm["candidate"]:
                self.assertAlmostEqual(sum(hmm["probabilities"].values()), 1)
                self.assertEqual(hmm["confidence"], max(hmm["probabilities"].values()))
