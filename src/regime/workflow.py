import hashlib
import io
from dataclasses import asdict
from itertools import takewhile
from pathlib import Path

from .domain import ContractError, ENGINE_VERSION, FEATURES, FEATURE_VERSION, Settings, parse_bars
from .engine import OnlineEngine
from .evaluation import evaluate
from .store import packed, publish


def fingerprint(obj):
    return hashlib.sha256(packed(obj).encode()).hexdigest()


def run(source, settings, db_path):
    cfg = Settings.load(settings)
    payload = Path(source).read_bytes()
    bars = list(parse_bars(io.StringIO(payload.decode("utf-8"))))
    split = sum(1 for _ in takewhile(lambda b: b.available_at <= cfg.train_through, bars))
    holdout = bars[split:]
    if not holdout:
        raise ContractError("empty holdout")
    engine, training_rows = OnlineEngine.fit(bars[:split], cfg)
    model = asdict(engine.model)
    meta = {"engine_version": ENGINE_VERSION, "feature_version": FEATURE_VERSION, "feature_order": FEATURES,
            "settings": asdict(cfg), "settings_sha256": fingerprint(asdict(cfg)),
            "source_sha256": hashlib.sha256(payload).hexdigest(), "model_sha256": fingerprint(model),
            "model": model, "train_last_available_at": bars[split-1].available_at}
    meta["run_id"] = fingerprint(meta)
    records = [engine.observe(bar) for bar in holdout]
    report = evaluate(holdout, records, engine, training_rows, split)
    report.update({k: meta[k] for k in ("run_id", "engine_version", "source_sha256", "model_sha256", "settings_sha256")})
    publish(db_path, meta, report, records)
    return report
