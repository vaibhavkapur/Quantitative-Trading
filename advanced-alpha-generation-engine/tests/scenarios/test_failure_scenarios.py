"""Scenario tests reproducing all 7 required failure paths from plan.md Section 5.

Required scenarios verified with injected faults:
1. Survivorship bias
2. Corporate actions
3. Look-ahead
4. Missing symbols
5. Factor duplication
6. High turnover
7. Unstable coefficients
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from alpha_engine.catalog import AlphaCatalog
from alpha_engine.config import EngineConfig, RegistryPolicy, UniversePolicy, WalkForwardConfig
from alpha_engine.contracts.models import FoldMetrics, SignalDefinition
from alpha_engine.contracts.validation import PointInTimeValidator
from alpha_engine.data.loader import ResearchDataset
from alpha_engine.errors import (
    CorporateActionIntegrityError,
    FactorDuplicationError,
    LookAheadViolationError,
    MissingSymbolError,
    PromotionGateError,
    SurvivorshipViolationError,
)
from alpha_engine.evaluation.metrics import evaluate_stability
from alpha_engine.evaluation.walk_forward import WalkForwardEvaluator, WalkForwardSplitter
from alpha_engine.factors import BaseFactor, MomentumFactor


class _ClonedMomentumFactor(BaseFactor):
    """Injected near-duplicate factor for Scenario 5 (Factor duplication)."""

    def __init__(self) -> None:
        self._base = MomentumFactor(lookback_days=20, skip_days=5)
        self._def = SignalDefinition(
            signal_id="mom_20_5_duplicate",
            name="Duplicated Momentum Clone",
            version="1.0.0",
            family="momentum",
            description="Scaled copy of mom_20_5 used to verify duplication rejection.",
            parameters={"lookback_days": 20, "skip_days": 5, "scale": 1.75},
            lookback_bars=20,
        )

    @property
    def definition(self) -> SignalDefinition:
        return self._def

    def compute(self, bars_df: pl.DataFrame, fundamentals_df: pl.DataFrame | None = None) -> pl.DataFrame:
        base_df = self._base.compute(bars_df, fundamentals_df)
        return self._finalize(base_df.with_columns((pl.col("raw_value") * 1.75).alias("raw_value")))


class _HighTurnoverNoiseFactor(BaseFactor):
    """Injected ultra-high-turnover alternating noise factor for Scenario 6 (High turnover)."""

    def __init__(self, seed: int = 999) -> None:
        self.seed = seed
        self._def = SignalDefinition(
            signal_id="fault_high_turnover_churn",
            name="Injected High-Turnover Churn Signal",
            version="1.0.0",
            family="reversal",
            description="Deterministic daily sign-flipping noise signal to reproduce high-turnover failure.",
            parameters={"seed": seed},
            lookback_bars=1,
        )

    @property
    def definition(self) -> SignalDefinition:
        return self._def

    def compute(self, bars_df: pl.DataFrame, fundamentals_df: pl.DataFrame | None = None) -> pl.DataFrame:
        rng = np.random.default_rng(self.seed)
        ordered = bars_df.sort(["event_ts_utc", "symbol"])
        vals = rng.normal(0.0, 1.0, size=ordered.height)
        return self._finalize(ordered.with_columns(pl.Series("raw_value", vals)))


def test_scenario_1_survivorship_bias_detected(
    research_dataset: ResearchDataset,
    universe_policy: UniversePolicy,
) -> None:
    """Scenario 1: Dropping delisted symbols from history MUST raise SurvivorshipViolationError."""
    validator = PointInTimeValidator(universe_policy)
    delisted_syms = (
        research_dataset.membership.filter(pl.col("exit_ts_utc").is_not_null())["symbol"].to_list()
    )
    assert len(delisted_syms) > 0

    survivor_only_bars = research_dataset.bars.filter(~pl.col("symbol").is_in(delisted_syms))
    with pytest.raises(SurvivorshipViolationError) as exc_info:
        validator.validate_survivorship(survivor_only_bars, research_dataset.membership)
    assert set(exc_info.value.details["missing_delisted_symbols"]) == set(delisted_syms)

    # Also verify bars after exit_ts_utc are rejected
    first_delisted = delisted_syms[0]
    exit_ts = research_dataset.membership.filter(pl.col("symbol") == first_delisted)["exit_ts_utc"].item()
    bad_bar = research_dataset.bars.filter(pl.col("symbol") == first_delisted).head(1).with_columns(
        pl.lit(exit_ts + timedelta(days=5)).alias("event_ts_utc"),
        pl.lit(exit_ts + timedelta(days=5, minutes=5)).alias("available_at_utc"),
    )
    with_post_exit = pl.concat([research_dataset.bars, bad_bar], how="vertical")
    with pytest.raises(SurvivorshipViolationError, match="after symbol exit_ts_utc"):
        validator.validate_survivorship(with_post_exit, research_dataset.membership)


def test_scenario_2_corporate_actions_unadjusted_split_rejected(
    research_dataset: ResearchDataset,
    universe_policy: UniversePolicy,
) -> None:
    """Scenario 2: Unadjusted split price cliff in adjusted_close MUST raise CorporateActionIntegrityError."""
    validator = PointInTimeValidator(universe_policy)
    # Fault: overwrite adjusted_close with raw unadjusted close so 2-for-1 and 3-for-1 splits create huge jumps
    unadjusted_bars = research_dataset.bars.with_columns(pl.col("close").alias("adjusted_close"))
    with pytest.raises(CorporateActionIntegrityError) as exc_info:
        validator.validate_corporate_actions(unadjusted_bars, research_dataset.corporate_actions)
    assert len(exc_info.value.details["bad_splits"]) >= 1


def test_scenario_3_look_ahead_bias_rejected(
    research_dataset: ResearchDataset,
    universe_policy: UniversePolicy,
) -> None:
    """Scenario 3: Signal or fundamental look-ahead MUST raise LookAheadViolationError."""
    validator = PointInTimeValidator(universe_policy)
    sig = MomentumFactor().compute(research_dataset.bars)

    # Fault A: Signal claims availability 10 minutes BEFORE bar was available
    leaky_sig = sig.with_columns(
        (pl.col("available_at_utc") - timedelta(minutes=10)).alias("available_at_utc")
    )
    with pytest.raises(LookAheadViolationError):
        validator.validate_signal_causality(leaky_sig, research_dataset.bars)

    # Fault B: Signal uses a fundamental filing BEFORE filing available_at_utc
    leaky_fund_sig = sig.with_columns(
        (pl.col("available_at_utc") + timedelta(days=30)).alias("source_available_at_utc")
    )
    with pytest.raises(LookAheadViolationError, match="unlagged source data"):
        validator.validate_signal_causality(leaky_fund_sig, research_dataset.bars)

    # Fault C: Walk-forward splitter preserves strict embargo (no date overlap between train and test)
    dates = research_dataset.bars.select("event_ts_utc").unique().sort("event_ts_utc")["event_ts_utc"].to_list()
    splits = WalkForwardSplitter(WalkForwardConfig()).split(dates)
    for sp in splits:
        assert sp.train_end_utc < sp.test_start_utc


def test_scenario_4_missing_symbols_coverage_violation_rejected(
    research_dataset: ResearchDataset,
    universe_policy: UniversePolicy,
) -> None:
    """Scenario 4: Dropping >10% of active symbols on trading dates MUST raise MissingSymbolError."""
    validator = PointInTimeValidator(universe_policy)
    drop_syms = [f"SYN{i:03d}" for i in range(1, 11)]  # drop 10 of 30 symbols (33%)
    sparse_bars = research_dataset.bars.filter(~pl.col("symbol").is_in(drop_syms))

    with pytest.raises(MissingSymbolError) as exc_info:
        validator.validate_symbol_coverage(sparse_bars, research_dataset.membership)
    assert exc_info.value.details["min_coverage_ratio_observed"] < universe_policy.min_symbol_coverage_ratio


def test_scenario_5_factor_duplication_blocked_from_promotion(
    tmp_path: Path,
    engine_config: EngineConfig,
    universe_policy: UniversePolicy,
    registry_policy: RegistryPolicy,
    research_dataset: ResearchDataset,
) -> None:
    """Scenario 5: Near-duplicate factor cannot be promoted alongside its parent factor."""
    evaluator = WalkForwardEvaluator(engine_config, universe_policy, registry_policy)
    orig = MomentumFactor(lookback_days=20, skip_days=5)
    dup = _ClonedMomentumFactor()

    rep_orig, oos_orig = evaluator.evaluate_factor(orig, research_dataset)
    rep_dup, oos_dup = evaluator.evaluate_factor(dup, research_dataset)

    # Ensure both look strong individually so the duplication gate is the decisive blocker
    pass_orig = rep_orig.model_copy(
        update={
            "passed_gates": True,
            "gate_failures": [],
            "oos_ic_pvalue": 0.0005,
            "oos_icir": 0.8,
        }
    )
    pass_dup = rep_dup.model_copy(
        update={
            "passed_gates": True,
            "gate_failures": [],
            "oos_ic_pvalue": 0.0005,
            "oos_icir": 0.79,
        }
    )

    catalog = AlphaCatalog(tmp_path / "dup_catalog", registry_policy)
    registered = catalog.register_batch(
        [
            (orig.definition, pass_orig, oos_orig),
            (dup.definition, pass_dup, oos_dup),
        ]
    )
    assert registered[0].passed_gates is True
    assert registered[1].passed_gates is False
    assert any("max_peer_correlation" in f for f in registered[1].gate_failures)

    catalog.promote(orig.definition.signal_id, orig.definition.version, approver="cto", approval_ticket="T-DUP-1")
    with pytest.raises((PromotionGateError, FactorDuplicationError)):
        catalog.promote(dup.definition.signal_id, dup.definition.version, approver="cto", approval_ticket="T-DUP-2")


def test_scenario_6_high_turnover_signal_rejected_and_blocked(
    tmp_path: Path,
    engine_config: EngineConfig,
    universe_policy: UniversePolicy,
    registry_policy: RegistryPolicy,
    research_dataset: ResearchDataset,
) -> None:
    """Scenario 6: High-turnover noisy signal fails turnover/cost gates and cannot be promoted."""
    # Disable smoothing to measure raw turnover impact and keep strict turnover threshold
    tight_cfg = engine_config.model_copy(
        update={
            "cost_model": engine_config.cost_model.model_copy(
                update={"turnover_smoothing_alpha": 1.0, "no_trade_buffer": 0.0}
            )
        }
    )
    evaluator = WalkForwardEvaluator(tight_cfg, universe_policy, registry_policy)
    churn = _HighTurnoverNoiseFactor()
    rep, oos_df = evaluator.evaluate_factor(churn, research_dataset)

    assert rep.avg_daily_turnover > registry_policy.max_avg_daily_turnover
    assert not rep.passed_gates
    assert any("avg_daily_turnover" in f for f in rep.gate_failures)

    catalog = AlphaCatalog(tmp_path / "turnover_catalog", registry_policy)
    catalog.register_batch([(churn.definition, rep, oos_df)])
    with pytest.raises(PromotionGateError):
        catalog.promote(churn.definition.signal_id, churn.definition.version, approver="cto", approval_ticket="T-TO-1")


def test_scenario_7_unstable_coefficients_flagged_and_blocked(
    tmp_path: Path,
    engine_config: EngineConfig,
    universe_policy: UniversePolicy,
    registry_policy: RegistryPolicy,
    research_dataset: ResearchDataset,
) -> None:
    """Scenario 7: Fold-to-fold sign-flipping coefficients fail stability gate and block promotion."""
    ts = datetime(2024, 1, 1, 21, 0, tzinfo=UTC)
    oscillating_folds = [
        FoldMetrics(
            fold_id=i,
            train_start_utc=ts,
            train_end_utc=ts,
            test_start_utc=ts,
            test_end_utc=ts,
            n_test_days=60,
            oos_spearman_ic=0.03 if i % 2 == 0 else -0.025,
            oos_pearson_ic=0.03 if i % 2 == 0 else -0.025,
            oos_icir=0.30 if i % 2 == 0 else -0.25,
            hit_rate=0.55 if i % 2 == 0 else 0.45,
            avg_daily_turnover=0.12,
            max_abs_sector_exposure=0.01,
            coefficients={
                "mom_20_5": 0.90 if i % 2 == 0 else -0.85,
                "val_earnings_yield": -0.60 if i % 2 == 0 else 0.70,
            },
        )
        for i in range(4)
    ]
    stability = evaluate_stability(
        oscillating_folds,
        max_sign_flip_rate=registry_policy.max_coefficient_sign_flip_rate,
        max_coefficient_cv=registry_policy.max_coefficient_cv,
        min_ic_sign_consistency=registry_policy.min_ic_sign_consistency,
    )
    assert not stability.is_stable
    assert stability.coefficient_sign_flip_rate == 1.0

    # Verify a candidate carrying this unstable report is blocked from catalog promotion
    evaluator = WalkForwardEvaluator(engine_config, universe_policy, registry_policy)
    mom = MomentumFactor()
    rep, oos_df = evaluator.evaluate_factor(mom, research_dataset)
    unstable_rep = rep.model_copy(
        update={
            "stability": stability,
            "passed_gates": False,
            "gate_failures": [f"stability:{r}" for r in stability.reasons],
        }
    )
    catalog = AlphaCatalog(tmp_path / "unstable_catalog", registry_policy)
    catalog.register_batch([(mom.definition, unstable_rep, oos_df)])
    with pytest.raises(PromotionGateError, match="CANNOT be promoted"):
        catalog.promote(mom.definition.signal_id, mom.definition.version, approver="cto", approval_ticket="T-STAB-1")
