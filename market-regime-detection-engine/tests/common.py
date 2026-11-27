from dataclasses import replace
from pathlib import Path

from regime.domain import Settings, read_bars

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/fixtures/daily.csv"
SETTINGS = ROOT / "config/daily.v1.json"
CFG = Settings.load(SETTINGS)
BARS = list(read_bars(SOURCE))
TRAIN = [b for b in BARS if b.available_at <= CFG.train_through]
TEST = BARS[len(TRAIN):]
FAST = replace(CFG, em_iterations=3)
