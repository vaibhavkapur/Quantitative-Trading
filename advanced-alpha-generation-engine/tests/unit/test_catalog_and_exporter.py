"""Unit tests for versioned alpha catalog, promotion gating, and paper-only exporter."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from alpha_engine.catalog import AlphaCatalog, PaperSignalExporter
from alpha_engine.config import EngineConfig, RegistryPolicy, UniversePolicy
from alpha_engine.data.loader import ResearchDataset
from alpha_engine.errors import (
    ExecutionDisabledError,
    LookAheadViolationError,
    PromotionGateError,
)
from alpha_engine.evaluation.walk_forward import WalkForwardEvaluator
from alpha_engine.factors import MomentumFactor, ShortTermReversalFactor


def test_catalog_records_multiple_trials_and_blocks_failed_promotion(
    tmp_path: Path,
    engine_config: EngineConfig,
    universe_policy: UniversePolicy,
    registry_policy: RegistryPolicy,
    research_dataset: ResearchDataset,
) -> None:
    evaluator = WalkForwardEvaluator(engine_config, universe_policy, registry_policy)
    mom = MomentumFactor()
    rev = ShortTermReversalFactor()

    rep_mom, oos_mom = evaluator.evaluate_factor(mom, research_dataset)
    rep_rev, oos_rev = evaluator.evaluate_factor(rev, research_dataset)

    # Force a deterministic failing report copy to test hard promotion block regardless of synthetic IC
    forced_fail = rep_rev.model_copy(
        update={
            "passed_gates": False,
            "gate_failures": ["oos_mean_ic=-0.0200 < min 0.0150"],
            "oos_ic_pvalue": 0.90,
        }
    )
    forced_pass = rep_mom.model_copy(
        update={
            "passed_gates": True,
            "gate_failures": [],
            "oos_ic_pvalue": 0.001,
            "oos_mean_ic": 0.05,
            "oos_icir": 0.6,
            "net_sharpe_at_base_cost": 1.0,
        }
    )

    catalog = AlphaCatalog(tmp_path / "catalog", registry_policy)
    catalog.register_batch(
        [
            (mom.definition, forced_pass, oos_mom),
            (rev.definition, forced_fail, oos_rev),
        ]
    )

    assert len(catalog.trials) == 2
    assert (tmp_path / "catalog" / "trials.jsonl").exists()

    # Failed signal CANNOT be promoted
    with pytest.raises(PromotionGateError, match="CANNOT be promoted"):
        catalog.promote("rev_3d", "1.0.0", approver="cto", approval_ticket="TKT-1")

    # Missing approval metadata MUST fail
    with pytest.raises(PromotionGateError, match="approver"):
        catalog.promote("mom_20_5", "1.0.0", approver="", approval_ticket="")

    # Passing signal with approval promotes cleanly
    promoted = catalog.promote("mom_20_5", "1.0.0", approver="cto", approval_ticket="TKT-2")
    assert promoted.status == "promoted"

    # Paper exporter checks
    exporter = PaperSignalExporter(catalog)
    weights = evaluator.position_researcher.build_weights(oos_mom)
    as_of_utc = weights.select("available_at_utc").max().item()

    # Execution enabled attempt MUST raise ExecutionDisabledError
    with pytest.raises(ExecutionDisabledError):
        exporter.export_paper_signals(
            signal_id="mom_20_5",
            version="1.0.0",
            weights_df=weights,
            as_of_utc=as_of_utc,
            dataset_manifest_digest=research_dataset.manifest.manifest_digest,
            request_live_execution=True,
        )

    # Unpromoted signal export MUST raise PromotionGateError
    with pytest.raises(PromotionGateError):
        exporter.export_paper_signals(
            signal_id="rev_3d",
            version="1.0.0",
            weights_df=weights,
            as_of_utc=as_of_utc,
            dataset_manifest_digest=research_dataset.manifest.manifest_digest,
        )

    # Exporting with cutoff earlier than weights available_at_utc MUST raise LookAheadViolationError
    with pytest.raises(LookAheadViolationError):
        exporter.export_paper_signals(
            signal_id="mom_20_5",
            version="1.0.0",
            weights_df=weights,
            as_of_utc=as_of_utc - timedelta(days=5),
            dataset_manifest_digest=research_dataset.manifest.manifest_digest,
        )

    # Valid paper-only export succeeds with SHA-256 digest and execution_enabled=False
    out_file = tmp_path / "paper_export.json"
    export = exporter.export_paper_signals(
        signal_id="mom_20_5",
        version="1.0.0",
        weights_df=weights,
        as_of_utc=as_of_utc,
        dataset_manifest_digest=research_dataset.manifest.manifest_digest,
        output_path=out_file,
    )
    assert export.mode == "paper_only"
    assert export.execution_enabled is False
    assert len(export.sha256_digest) == 64
    assert export.records_count > 0
    assert out_file.exists()
