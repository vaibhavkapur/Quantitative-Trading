"""Unit tests for neutralization, IC/ICIR metrics, BH adjustment, correlation, and stability."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from alpha_engine.config import NeutralizationConfig
from alpha_engine.contracts.models import FoldMetrics
from alpha_engine.data.loader import ResearchDataset
from alpha_engine.errors import FactorDuplicationError
from alpha_engine.evaluation.metrics import (
    assert_no_factor_duplication,
    benjamini_hochberg_adjust,
    compute_factor_correlation_matrix,
    evaluate_stability,
    summarize_ic_series,
)
from alpha_engine.evaluation.neutralization import SignalNeutralizer
from alpha_engine.factors import MomentumFactor


def test_neutralizer_removes_sector_means_and_standardizes(research_dataset: ResearchDataset) -> None:
    raw = MomentumFactor().compute(research_dataset.bars)
    neut = SignalNeutralizer(NeutralizationConfig()).neutralize(raw, research_dataset.bars)

    # Every date + sector group should have near-zero mean neutralized value
    sec_means = (
        neut.group_by(["event_ts_utc", "sector"])
        .agg(pl.col("neutralized_value").mean().abs().alias("abs_mean"))
        .select(pl.col("abs_mean").max())
        .item()
    )
    assert sec_means < 1e-8

    # Cross-sectional std should be ~1.0
    daily_std = (
        neut.group_by("event_ts_utc")
        .agg(pl.col("neutralized_value").std(ddof=0).alias("sd"))
        .select(pl.col("sd").mean())
        .item()
    )
    assert abs(daily_std - 1.0) < 0.05


def test_ic_summary_and_bh_adjustment() -> None:
    rng = np.random.default_rng(123)
    ics = list(rng.normal(0.05, 0.08, size=200))
    s1 = summarize_ic_series(ics, seed=42)
    s2 = summarize_ic_series(ics, seed=42)
    assert s1 == s2
    assert s1["mean_ic"] > 0.03
    assert s1["ci_low"] < s1["mean_ic"] < s1["ci_high"]
    assert s1["pvalue"] < 0.01

    raw_p = [0.001, 0.02, 0.04, 0.50]
    adj_p = benjamini_hochberg_adjust(raw_p)
    assert all(adj >= raw for adj, raw in zip(adj_p, raw_p, strict=True))
    assert adj_p[0] <= adj_p[1] <= adj_p[2] <= adj_p[3] <= 1.0


def test_factor_correlation_and_duplication_guard(research_dataset: ResearchDataset) -> None:
    raw = MomentumFactor().compute(research_dataset.bars)
    neut = SignalNeutralizer(NeutralizationConfig()).neutralize(raw, research_dataset.bars)
    clone = neut.with_columns((pl.col("neutralized_value") * 2.5 + 0.01).alias("neutralized_value"))

    corr = compute_factor_correlation_matrix({"mom_20_5": neut, "mom_clone": clone})
    assert abs(corr["mom_20_5"]["mom_clone"] - 1.0) < 1e-4

    with pytest.raises(FactorDuplicationError):
        assert_no_factor_duplication("mom_clone", corr, max_abs_correlation=0.85)


def test_stability_report_flags_sign_flips_and_high_cv() -> None:
    ts = datetime(2024, 1, 1, 21, 0, tzinfo=UTC)
    unstable_folds = [
        FoldMetrics(
            fold_id=i,
            train_start_utc=ts,
            train_end_utc=ts,
            test_start_utc=ts,
            test_end_utc=ts,
            n_test_days=60,
            oos_spearman_ic=0.04 if i % 2 == 0 else -0.05,
            oos_pearson_ic=0.04 if i % 2 == 0 else -0.05,
            oos_icir=0.3 if i % 2 == 0 else -0.3,
            hit_rate=0.55 if i % 2 == 0 else 0.45,
            avg_daily_turnover=0.15,
            max_abs_sector_exposure=0.01,
            coefficients={"f1": 0.5 if i % 2 == 0 else -0.6},
        )
        for i in range(4)
    ]
    report = evaluate_stability(unstable_folds)
    assert not report.is_stable
    assert report.coefficient_sign_flip_rate == 1.0
    assert len(report.reasons) >= 2
