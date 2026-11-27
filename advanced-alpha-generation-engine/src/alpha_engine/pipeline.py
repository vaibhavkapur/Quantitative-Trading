"""End-to-end M1-M4 alpha research, evaluation, catalog registration, and paper-export pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from alpha_engine.adapters.legacy_foundations import (
    QuantumOrderBookAdapter,
    QuantumPairsSpreadAdapter,
    QuantumStatArbAdapter,
)
from alpha_engine.catalog.exporter import PaperSignalExporter
from alpha_engine.catalog.registry import AlphaCatalog
from alpha_engine.config import (
    EngineConfig,
    RegistryPolicy,
    UniversePolicy,
    load_engine_config,
    load_registry_policy,
    load_universe_policy,
)
from alpha_engine.contracts.models import EvaluationReport, PaperSignalExport, SignalDefinition
from alpha_engine.data.loader import PointInTimeDataLoader, ResearchDataset
from alpha_engine.errors import AlphaEngineError, OptionalDependencyError
from alpha_engine.evaluation.walk_forward import WalkForwardEvaluator
from alpha_engine.factors.base import BaseFactor, FeatureMatrixBuilder
from alpha_engine.factors.library import default_factor_suite
from alpha_engine.models.baselines import (
    AlphaModel,
    EqualRankCompositeBaseline,
    GradientBoostedAlphaBaseline,
    RidgeAlphaBaseline,
    XGBoostAlphaBaseline,
)
from alpha_engine.observability import EngineMetricsRecorder


@dataclass
class PipelineResult:
    dataset: ResearchDataset
    reports: list[EvaluationReport]
    oos_panels: dict[str, pl.DataFrame]
    promoted_keys: list[str]
    rejected_keys: list[str]
    paper_exports: list[PaperSignalExport]
    correlation_matrix: dict[str, dict[str, float]]
    metrics: dict[str, Any]


class AlphaResearchPipeline:
    """Runs the complete causal alpha research lifecycle without ever enabling live execution."""

    def __init__(
        self,
        *,
        engine_config: EngineConfig,
        universe_policy: UniversePolicy,
        registry_policy: RegistryPolicy,
        fixtures_dir: Path,
        catalog_dir: Path,
        include_xgboost: bool = True,
    ) -> None:
        self.engine_config = engine_config
        self.universe_policy = universe_policy
        self.registry_policy = registry_policy
        self.fixtures_dir = fixtures_dir
        self.catalog_dir = catalog_dir
        self.include_xgboost = include_xgboost

    @classmethod
    def from_repo_root(cls, repo_root: Path, *, catalog_dir: Path | None = None) -> AlphaResearchPipeline:
        cfg_dir = repo_root / "config"
        return cls(
            engine_config=load_engine_config(cfg_dir / "engine_config.json"),
            universe_policy=load_universe_policy(cfg_dir / "universe_policy.json"),
            registry_policy=load_registry_policy(cfg_dir / "registry_policy.json"),
            fixtures_dir=repo_root / "data" / "fixtures",
            catalog_dir=catalog_dir or (repo_root / "reports" / "catalog"),
        )

    def run(
        self,
        *,
        approver: str = "research-review-board",
        approval_ticket: str = "PAPER-REVIEW-001",
        export_dir: Path | None = None,
        extra_factors: list[BaseFactor] | None = None,
    ) -> PipelineResult:
        metrics = EngineMetricsRecorder()

        # M1: Load and validate point-in-time dataset
        with metrics.time_stage("m1_pit_data_load_and_validate"):
            loader = PointInTimeDataLoader(self.fixtures_dir, self.universe_policy)
            dataset = loader.load()
            metrics.increment("datasets_validated")

        # M2: Build core factors, legacy adapters, and feature matrix for predictive baselines
        with metrics.time_stage("m2_feature_and_signal_generation"):
            core_factors: list[BaseFactor] = default_factor_suite()
            legacy_adapters: list[BaseFactor] = [
                QuantumStatArbAdapter(window_days=20),
                QuantumPairsSpreadAdapter(window_days=30),
                QuantumOrderBookAdapter(smoothing_days=5),
            ]
            all_factors = core_factors + legacy_adapters + (extra_factors or [])
            feature_cols = [f.definition.signal_id for f in core_factors]
            wide_features, _ = FeatureMatrixBuilder(core_factors).build(
                dataset.bars, fundamentals_df=dataset.fundamentals
            )
            metrics.increment("factors_configured", len(all_factors))

        # M3: Walk-forward evaluation of factors & statistical/ML baselines
        evaluator = WalkForwardEvaluator(
            self.engine_config, self.universe_policy, self.registry_policy
        )
        evaluated: list[tuple[SignalDefinition, EvaluationReport, pl.DataFrame]] = []
        oos_panels: dict[str, pl.DataFrame] = {}

        with metrics.time_stage("m3_walk_forward_evaluation"):
            for factor in all_factors:
                rep, oos_df = evaluator.evaluate_factor(factor, dataset)
                evaluated.append((factor.definition, rep, oos_df))
                oos_panels[factor.definition.signal_id] = oos_df
                metrics.increment("signals_evaluated")

            models: list[AlphaModel] = [
                EqualRankCompositeBaseline(
                    feature_signs={
                        "mom_20_5": 1.0,
                        "mom_60_5": 1.0,
                        "rev_3d": 0.0,
                        "inv_vol_20d": 1.0,
                        "val_earnings_yield": 1.0,
                        "qual_roe": 1.0,
                    }
                ),
                RidgeAlphaBaseline(alpha=75.0),
                GradientBoostedAlphaBaseline(seed=self.engine_config.seed),
            ]
            if self.include_xgboost:
                models.append(XGBoostAlphaBaseline(seed=self.engine_config.seed))

            for model in models:
                try:
                    rep, oos_df = evaluator.evaluate_model(
                        model, wide_features, feature_cols, dataset
                    )
                    evaluated.append((model.definition, rep, oos_df))
                    oos_panels[model.definition.signal_id] = oos_df
                    metrics.increment("models_evaluated")
                except OptionalDependencyError as exc:
                    metrics.record_failure(
                        "m3_walk_forward_evaluation", exc.error_code, exc.message, exc.details
                    )

        # M4: Register all trials (passing & failing), promote gated winners, export paper-only signals
        promoted_keys: list[str] = []
        rejected_keys: list[str] = []
        paper_exports: list[PaperSignalExport] = []

        with metrics.time_stage("m4_catalog_and_paper_export"):
            catalog = AlphaCatalog(self.catalog_dir, self.registry_policy)
            final_reports = catalog.register_batch(evaluated)
            metrics.increment("trials_recorded", len(final_reports))

            exporter = PaperSignalExporter(catalog)
            as_of_utc = dataset.bars.select("available_at_utc").max().item()

            # Promote passing signals in descending OOS ICIR order
            ordered_reports = sorted(final_reports, key=lambda r: r.oos_icir, reverse=True)
            for rep in ordered_reports:
                sig_key = f"{rep.signal_id}@{rep.signal_version}"
                if rep.passed_gates:
                    try:
                        catalog.promote(
                            rep.signal_id,
                            rep.signal_version,
                            approver=approver,
                            approval_ticket=approval_ticket,
                        )
                        promoted_keys.append(sig_key)
                        metrics.increment("candidates_promoted")
                        metrics.record_decision(
                            "catalog_promotion",
                            sig_key,
                            "promoted",
                            {"oos_mean_ic": rep.oos_mean_ic, "oos_icir": rep.oos_icir},
                        )
                        # Export paper-only weights for promoted signals
                        oos_df = oos_panels[rep.signal_id]
                        weights_df = evaluator.position_researcher.build_weights(oos_df)
                        out_file = (
                            (export_dir / f"paper_export_{rep.signal_id}.json")
                            if export_dir is not None
                            else None
                        )
                        exp = exporter.export_paper_signals(
                            signal_id=rep.signal_id,
                            version=rep.signal_version,
                            weights_df=weights_df,
                            as_of_utc=as_of_utc,
                            dataset_manifest_digest=dataset.manifest.manifest_digest,
                            output_path=out_file,
                        )
                        paper_exports.append(exp)
                        metrics.increment("paper_exports_written")
                    except AlphaEngineError as exc:
                        rejected_keys.append(sig_key)
                        metrics.increment("candidates_rejected")
                        metrics.record_decision(
                            "catalog_promotion",
                            sig_key,
                            "rejected",
                            exc.to_dict(),
                        )
                else:
                    rejected_keys.append(sig_key)
                    metrics.increment("candidates_rejected")
                    metrics.record_decision(
                        "catalog_promotion",
                        sig_key,
                        "rejected",
                        {"gate_failures": rep.gate_failures},
                    )

        return PipelineResult(
            dataset=dataset,
            reports=final_reports,
            oos_panels=oos_panels,
            promoted_keys=promoted_keys,
            rejected_keys=rejected_keys,
            paper_exports=paper_exports,
            correlation_matrix=catalog.correlation_matrix,
            metrics=metrics.snapshot(),
        )
