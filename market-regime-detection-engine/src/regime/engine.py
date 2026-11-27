from .domain import ContractError, FEATURES, STATES
from .features import TrailingFeatures
from .models import GaussianHMM, baseline


class StabilityGate:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.label = "unknown"
        self.pending = None
        self.count = 0

    def step(self, probabilities=None, reason=None):
        probs = dict(zip(STATES, probabilities)) if probabilities is not None else {}
        candidate = max(probs, key=probs.get) if probs else None
        confidence = probs.get(candidate, 0.)
        if reason is None and confidence < self.cfg.confidence_floor:
            reason = "low_confidence"
        if reason:
            self.reset()
        elif candidate == self.label:
            self.pending, self.count = None, 0
        else:
            self.count = self.count + 1 if candidate == self.pending else 1
            self.pending = candidate
            if self.count >= self.cfg.confirm_bars:
                self.label = candidate
                self.pending, self.count = None, 0
            else:
                reason = "confirming"
        return dict(label=self.label, candidate=candidate, confidence=confidence,
                    probabilities=probs, reason=reason or "accepted")


class OnlineEngine:
    def __init__(self, cfg, features, model):
        self.cfg = cfg
        self.features = features
        self.model = model
        self.previous = None
        self.rule_gate = StabilityGate(cfg)
        self.hmm_gate = StabilityGate(cfg)

    @classmethod
    def fit(cls, training, cfg):
        features = TrailingFeatures(cfg)
        segments, segment = [], []
        for bar in training:
            if bar.available_at > cfg.train_through:
                raise ContractError("training observation arrived after cutoff")
            x, _ = features.observe(bar)
            if x is not None:
                segment.append(x)
            elif segment:
                segments.append(segment)
                segment = []
        if segment:
            segments.append(segment)
        if sum(map(len, segments)) < max(30, cfg.window * 2):
            raise ContractError("insufficient complete training features")
        model = GaussianHMM.fit(segments, cfg.em_iterations)
        return cls(cfg, features, model), [row for seg in segments for row in seg]

    def observe(self, bar):
        if bar.available_at <= self.cfg.train_through:
            raise ContractError("inference requires a post-cutoff arrival")
        x, reason = self.features.observe(bar)
        distance = None
        if x is None:
            self.previous = None
            rule, hmm = self.rule_gate.step(reason=reason), self.hmm_gate.step(reason=reason)
        else:
            rule = self.rule_gate.step(baseline(x, self.cfg))
            posterior, distance = self.model.filter(x, self.previous)
            reason = "outside_training_support" if distance > self.cfg.ood_rms else None
            self.previous = None if reason else posterior
            hmm = self.hmm_gate.step(posterior, reason)
        return dict(event_at=bar.event_at, available_at=bar.available_at, series=bar.series,
                    features=dict(zip(FEATURES, x)) if x is not None else None,
                    feature_status="ready" if x is not None else reason, distance_rms=distance,
                    rule=rule, hmm=hmm)
