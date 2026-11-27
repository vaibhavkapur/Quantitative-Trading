"""Deterministic baseline and three-state diagonal Gaussian HMM."""
import math
from dataclasses import dataclass
from statistics import fmean, pvariance

K = range(3)
D = range(4)


def normalize(weights):
    total = sum(weights)
    return [v / total for v in weights]


def relative_exp(logs):
    peak = max(logs)
    return [math.exp(v - peak) for v in logs]


def baseline(x, cfg):
    vol, trend, spread, _ = x
    # At either stress threshold the stress logit equals the stronger trend logit.
    stress = max(vol / cfg.stress_vol, spread / cfg.stress_spread_bps)
    return normalize(relative_exp([trend, -trend, abs(trend) + 4 * (stress - 1)]))


def predict(prior, transition):
    return [sum(prior[i] * transition[i][j] for i in K) for j in K]


def weighted_moments(rows, weights):
    total = sum(weights)
    mean = [sum(w * x[j] for x, w in zip(rows, weights)) / total for j in D]
    var = [max(.05, sum(w * (x[j] - mean[j]) ** 2 for x, w in zip(rows, weights)) / total) for j in D]
    return mean, var


@dataclass
class GaussianHMM:
    center: list
    scale: list
    means: list
    variances: list
    transition: list
    initial: list

    def standardize(self, row):
        return [(row[j] - self.center[j]) / self.scale[j] for j in D]

    def emission(self, x):
        distances = [sum((x[j] - self.means[k][j]) ** 2 / self.variances[k][j] for j in D) for k in K]
        logs = [-.5 * (distances[k] + sum(math.log(v) for v in self.variances[k])) for k in K]
        return relative_exp(logs), math.sqrt(min(distances) / len(x))

    def filter(self, row, previous):
        likelihood, distance = self.emission(self.standardize(row))
        prior = self.initial if previous is None else predict(previous, self.transition)
        return normalize([prior[k] * likelihood[k] for k in K]), distance

    @classmethod
    def fit(cls, segments, iterations):
        rows = [row for segment in segments for row in segment]
        center = [fmean(r[j] for r in rows) for j in D]
        scale = [max(1e-6, math.sqrt(pvariance(r[j] for r in rows))) for j in D]
        model = cls(center, scale, [], [], [[.9 if i == j else .05 for j in K] for i in K], [1/3] * 3)
        segments = [[model.standardize(row) for row in segment] for segment in segments]
        xs = [x for segment in segments for x in segment]
        by_risk = sorted(range(len(xs)), key=lambda i: xs[i][0] + xs[i][2])
        cut = 2 * len(xs) // 3
        by_trend = sorted(by_risk[:cut], key=lambda i: xs[i][1])
        groups = [by_trend[cut//2:], by_trend[:cut//2], by_risk[cut:]]
        for group in groups:
            mean, var = weighted_moments([xs[i] for i in group], [1] * len(group))
            model.means.append(mean)
            model.variances.append(var)
        for _ in range(iterations):
            model._em_step(segments, xs)
        # Give latent components reproducible economic names after estimation.
        stressed = max(K, key=lambda k: model.means[k][0] + model.means[k][2])
        calm = sorted((k for k in K if k != stressed), key=lambda k: -model.means[k][1])
        order = calm + [stressed]
        return cls(center, scale, [model.means[k] for k in order], [model.variances[k] for k in order],
                   [[model.transition[i][j] for j in order] for i in order], [model.initial[k] for k in order])

    def _em_step(self, segments, xs):
        # Forward-backward is confined to the fit partition. Gaps split sequences.
        starts = [0.05] * 3
        counts = [[0.05] * 3 for _ in K]
        responsibilities = []
        for segment in segments:
            emissions = [self.emission(x)[0] for x in segment]
            alphas = []
            prior = self.initial
            for likelihood in emissions:
                alpha = normalize([prior[k] * likelihood[k] for k in K])
                alphas.append(alpha)
                prior = predict(alpha, self.transition)
            beta = [1.0] * 3
            gammas = []
            for t in reversed(range(len(segment))):
                gamma = normalize([alphas[t][k] * beta[k] for k in K])
                gammas.append(gamma)
                if t == 0:
                    starts = [starts[k] + gamma[k] for k in K]
                    break
                future = [emissions[t][j] * beta[j] for j in K]
                joint = [[alphas[t-1][i] * self.transition[i][j] * future[j] for j in K] for i in K]
                total = sum(map(sum, joint))
                for i in K:
                    for j in K:
                        counts[i][j] += joint[i][j] / total
                beta = normalize([sum(self.transition[i][j] * future[j] for j in K) for i in K])
            responsibilities.extend(reversed(gammas))
        self.initial = normalize(starts)
        self.transition = [normalize(row) for row in counts]
        for k in K:
            weights = [max(1e-12, gamma[k]) for gamma in responsibilities]
            self.means[k], self.variances[k] = weighted_moments(xs, weights)
