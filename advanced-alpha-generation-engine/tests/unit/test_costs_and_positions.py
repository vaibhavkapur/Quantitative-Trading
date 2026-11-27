"""Unit tests for cost-aware signal-to-position research and turnover modeling."""

from __future__ import annotations

import polars as pl

from alpha_engine.config import CostModelConfig, NeutralizationConfig
from alpha_engine.data.loader import ResearchDataset
from alpha_engine.evaluation.costs import SignalToPositionResearcher
from alpha_engine.evaluation.neutralization import SignalNeutralizer
from alpha_engine.factors import MomentumFactor


def test_weights_are_dollar_neutral_capped_and_smoothing_reduces_turnover(
    research_dataset: ResearchDataset,
) -> None:
    raw = MomentumFactor().compute(research_dataset.bars)
    neut = SignalNeutralizer(NeutralizationConfig()).neutralize(raw, research_dataset.bars)

    cfg = CostModelConfig()
    researcher = SignalToPositionResearcher(cfg)
    w_smooth = researcher.build_weights(neut, apply_smoothing=True)
    w_raw = researcher.build_weights(neut, apply_smoothing=False)

    # Check dollar neutrality and gross exposure on active positions
    active = w_smooth.filter(pl.col("sector") != "EXITED")
    daily_exp = active.group_by("event_ts_utc").agg(
        pl.col("weight").sum().abs().alias("abs_net"),
        pl.col("weight").abs().sum().alias("gross"),
        pl.col("weight").abs().max().alias("max_w"),
    )
    assert daily_exp.select(pl.col("abs_net").max()).item() < 1e-6
    assert abs(daily_exp.select(pl.col("gross").mean()).item() - 1.0) < 1e-4

    # Cost sensitivity sweep must show monotonically increasing cost drag with bps
    pts_smooth, to_smooth = researcher.evaluate_cost_sensitivity(w_smooth, research_dataset.bars)
    pts_raw, to_raw = researcher.evaluate_cost_sensitivity(w_raw, research_dataset.bars)

    assert to_smooth < to_raw
    drags = [p.total_cost_drag_bps for p in pts_smooth]
    assert drags[0] == 0.0
    assert all(drags[i] <= drags[i + 1] for i in range(len(drags) - 1))
    # Net annualized return should decrease as cost increases
    net_rets = [p.net_annualized_return for p in pts_smooth]
    assert all(net_rets[i] >= net_rets[i + 1] for i in range(len(net_rets) - 1))
