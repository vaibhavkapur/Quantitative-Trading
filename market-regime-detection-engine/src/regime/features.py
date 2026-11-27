import math
from collections import deque
from statistics import fmean

from .domain import ContractError, timestamp


class TrailingFeatures:
    def __init__(self, cfg):
        self.cfg = cfg
        self.window = deque(maxlen=cfg.window)
        self.last = None
        self.anchor = None

    def observe(self, bar):
        if bar.series != self.cfg.series:
            raise ContractError("series mismatch")
        last = self.last
        if last and (bar.event_at[:10] <= last.event_at[:10] or bar.available_at <= last.available_at):
            raise ContractError("expected strictly ordered daily events and arrivals")
        gap = last and (timestamp(bar.event_at).date() - timestamp(last.event_at).date()).days > self.cfg.max_gap_days
        self.last = bar
        if gap or None in (bar.close, bar.benchmark_close, bar.spread_bps):
            self.window.clear()
            self.anchor = None
        if None in (bar.close, bar.benchmark_close, bar.spread_bps):
            return None, "missing_input"
        anchor, self.anchor = self.anchor, bar
        if anchor:
            self.window.append((math.log(bar.close) - math.log(anchor.close),
                                math.log(bar.benchmark_close) - math.log(anchor.benchmark_close), bar.spread_bps))
        if len(self.window) < self.cfg.window:
            return None, "gap" if gap else "warmup"
        asset, bench, spreads = zip(*self.window)
        a, b = fmean(asset), fmean(bench)
        aa = sum((r - a) ** 2 for r in asset)
        bb = sum((r - b) ** 2 for r in bench)
        if aa < 1e-20 or bb < 1e-20:
            return None, "flat_window"
        sd = math.sqrt(aa / (len(asset) - 1))
        corr = sum((x - a) * (y - b) for x, y in zip(asset, bench)) / math.sqrt(aa * bb)
        return (sd * math.sqrt(252), a * math.sqrt(len(asset)) / sd,
                fmean(spreads), max(-1.0, min(1.0, corr))), None
