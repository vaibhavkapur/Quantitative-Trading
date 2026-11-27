"""End-to-end integration test running M1-M4 AlphaResearchPipeline on deterministic fixtures."""

from __future__ import annotations

from pathlib import Path

from alpha_engine.config import EngineConfig, RegistryPolicy, UniversePolicy
from alpha_engine.pipeline import AlphaResearchPipeline


def test_end_to_end_pipeline_runs_m1_through_m4(
    saved_fixtures_dir: Path,
    tmp_path: Path,
    engine_config: EngineConfig,
    universe_policy: UniversePolicy,
    registry_policy: RegistryPolicy,
) -> None:
    catalog_dir = tmp_path / "catalog"
    export_dir = tmp_path / "exports"
    pipe = AlphaResearchPipeline(
        engine_config=engine_config,
        universe_policy=universe_policy,
        registry_policy=registry_policy,
        fixtures_dir=saved_fixtures_dir,
        catalog_dir=catalog_dir,
        include_xgboost=True,
    )
    res = pipe.run(export_dir=export_dir)

    assert res.dataset.validation.passed
    # 6 core factors + 3 legacy adapters + 3 core statistical/ML baselines (+1 optional XGBoost when libomp present)
    assert len(res.reports) in (12, 13)
    # Both passing/promoted candidates and rejected negative results should be present and recorded
    assert len(res.promoted_keys) >= 1
    assert len(res.rejected_keys) >= 1
    assert len(res.paper_exports) == len(res.promoted_keys)
    assert all(exp.execution_enabled is False and exp.mode == "paper_only" for exp in res.paper_exports)
    assert (catalog_dir / "trials.jsonl").exists()
    assert (catalog_dir / "catalog.json").exists()
    assert res.metrics["status"] == "healthy"
    assert res.metrics["counters"]["trials_recorded"] == len(res.reports)
