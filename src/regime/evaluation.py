"""Descriptive comparisons over a single chronological holdout."""
from collections import Counter
from statistics import fmean, median

from .domain import FEATURES


def stability(records, name):
    labels = [r[name]["label"] for r in records]
    runs = []
    for label in labels:
        if runs and runs[-1][0] == label:
            runs[-1][1] += 1
        else:
            runs.append([label, 1])
    transitions = sum(a != b for a, b in zip(labels, labels[1:]))
    return {"label_counts": dict(Counter(labels)), "reason_counts": dict(Counter(r[name]["reason"] for r in records)),
            "coverage": 1 - labels.count("unknown") / len(labels), "changes_including_unknown": transitions,
            "changes_per_100_intervals": 100 * transitions / max(1, len(labels) - 1),
            "direct_known_changes": sum(a != b and "unknown" not in (a, b) for a, b in zip(labels, labels[1:])),
            "median_run_bars_including_unknown": median(n for _, n in runs)}


def strategy(bars, records, name, cost_bps):
    # Latest signal available strictly before this close may trade at this close.
    # A missing mark defers both valuation and trading to the next observed close.
    last_price = None
    cursor, position, target = 0, 0, 0
    net = gross = 1.0
    costs = turnover = 0.
    observations = []
    rate = cost_bps / 10000
    for bar in bars:
        while cursor < len(records) and records[cursor]["available_at"] < bar.event_at:
            target = int(name == "long" or records[cursor][name]["label"] == "rising")
            cursor += 1
        if bar.close is None:
            continue
        before = net
        if last_price is not None:
            factor = bar.close / last_price if position else 1.
            net *= factor
            gross *= factor
        traded = abs(target - position)
        fee = net * traded * rate
        net -= fee
        costs += fee
        turnover += traded
        position, last_price = target, bar.close
        observations.append(net / before - 1)
    # Liquidate at the final observed mark including one-way exit costs.
    if position:
        fee = net * rate
        costs += fee
        net -= fee
        turnover += 1
        observations[-1] = (1 + observations[-1]) * (1 - rate) - 1
    blocks = [fmean(observations[i:i+10]) for i in range(0, len(observations) - 9, 10)]
    return {"gross_return": gross - 1, "net_return": net - 1, "fees_per_initial_dollar": costs,
            "one_way_turnover": turnover, "observed_marks": len(observations),
            "ten_mark_mean_return_range": [min(blocks), max(blocks)] if blocks else None}


def drift(records, model, size=20):
    blocks = []
    for i in range(0, len(records), size):
        batch = records[i:i+size]
        valid = [r["features"] for r in batch if r["features"] is not None]
        shifts = {key: (fmean(x[key] for x in valid) - model.center[j]) / model.scale[j]
                  for j, key in enumerate(FEATURES)} if valid else {}
        blocks.append(dict(start=batch[0]["available_at"], end=batch[-1]["available_at"], bars=len(batch),
                           complete=len(valid), mean_shift_training_sd=shifts,
                           max_abs_shift_over_two=any(abs(z) > 2 for z in shifts.values()),
                           outside_training_support=sum(r["hmm"]["reason"] == "outside_training_support" for r in batch)))
    return blocks


def evaluate(bars, records, engine, training_rows, training_count):
    cfg = engine.cfg
    strategies = {name: strategy(bars, records, name, cfg.cost_bps) for name in ("rule", "hmm", "long")}
    return {"training_bars": training_count, "training_features": len(training_rows),
            "first_arrival": records[0]["available_at"], "last_arrival": records[-1]["available_at"],
            "holdout_bars": len(records), "feature_status": dict(Counter(r["feature_status"] for r in records)),
            "stability": {name: stability(records, name) for name in ("rule", "hmm")},
            "illustrative_strategy": {"cost_bps_per_side": cfg.cost_bps, "comparison": strategies,
                                      "underperformed_long_after_costs": [s for s in ("rule", "hmm")
                                      if strategies[s]["net_return"] < strategies["long"]["net_return"]]},
            "drift_blocks": drift(records, engine.model)}
