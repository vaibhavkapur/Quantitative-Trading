"""Reproducible M1-M4 demonstration script generating milestone reports and paper-only exports."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from alpha_engine.adapters import (
    AIQuantResearchAgentAdapter,
    QuantumOrderBookAdapter,
    QuantumPairsSpreadAdapter,
    QuantumStatArbAdapter,
)
from alpha_engine.data.fixtures import generate_synthetic_pit_dataset, save_dataset_fixtures
from alpha_engine.pipeline import AlphaResearchPipeline


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def run_demo(repo_root: Path | None = None) -> dict:
    root = repo_root or Path(__file__).resolve().parent.parent
    fixtures_dir = root / "data" / "fixtures"
    reports_dir = root / "reports"
    catalog_dir = reports_dir / "catalog"
    export_dir = reports_dir / "paper_exports"

    # Ensure deterministic fixtures exist and match seed=42
    bundle = generate_synthetic_pit_dataset(seed=42)
    save_dataset_fixtures(bundle, fixtures_dir)

    # Start a clean catalog directory for reproducible demo runs
    if catalog_dir.exists():
        shutil.rmtree(catalog_dir)
    if export_dir.exists():
        shutil.rmtree(export_dir)

    pipeline = AlphaResearchPipeline.from_repo_root(root, catalog_dir=catalog_dir)
    result = pipeline.run(
        approver="saat-research-committee",
        approval_ticket="PAPER-APPROVAL-2026-001",
        export_dir=export_dir,
    )

    # M1 Foundation & PIT Validation Report
    m1_report = {
        "milestone": "M1 — Foundation and design",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_manifest": result.dataset.manifest.model_dump(mode="json"),
        "pit_validation": {
            "passed": result.dataset.validation.passed,
            "checks_run": result.dataset.validation.checks_run,
            "metrics": result.dataset.validation.metrics,
        },
        "legacy_adapter_provenance": [
            QuantumStatArbAdapter().provenance.model_dump(mode="json"),
            QuantumPairsSpreadAdapter().provenance.model_dump(mode="json"),
            QuantumOrderBookAdapter().provenance.model_dump(mode="json"),
            AIQuantResearchAgentAdapter.provenance.model_dump(mode="json"),
        ],
    }
    _write_json(reports_dir / "m1_foundation_report.json", m1_report)

    # M2 Signals & Baselines Inventory Report
    m2_report = {
        "milestone": "M2 — Core implementation",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "candidates_evaluated": [
            {
                "signal_id": r.signal_id,
                "signal_version": r.signal_version,
                "definition_hash": r.definition_hash,
                "sample_days": r.sample_days,
                "universe_size": r.universe_size,
            }
            for r in result.reports
        ],
    }
    _write_json(reports_dir / "m2_signals_baselines_report.json", m2_report)

    # M3 Walk-Forward Evaluation Report (IC, ICIR, BH p-val, turnover, sector exposure, stability, costs)
    m3_report = {
        "milestone": "M3 — Integration and validation",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "walk_forward_config": pipeline.engine_config.walk_forward.model_dump(mode="json"),
        "cost_model_config": pipeline.engine_config.cost_model.model_dump(mode="json"),
        "evaluations": [r.model_dump(mode="json") for r in result.reports],
        "pairwise_correlation_matrix": result.correlation_matrix,
    }
    _write_json(reports_dir / "m3_walk_forward_evaluation.json", m3_report)

    # M4 Catalog Registration & Paper-Only Export Report
    m4_report = {
        "milestone": "M4 — Hardening and delivery",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "execution_mode": "paper_only",
        "execution_enabled": False,
        "promoted_signals": result.promoted_keys,
        "rejected_negative_results": [
            {
                "signal_key": f"{r.signal_id}@{r.signal_version}",
                "oos_mean_ic": r.oos_mean_ic,
                "oos_icir": r.oos_icir,
                "bh_adjusted_pvalue": r.bh_adjusted_pvalue,
                "net_sharpe_at_base_cost": r.net_sharpe_at_base_cost,
                "gate_failures": r.gate_failures,
            }
            for r in result.reports
            if f"{r.signal_id}@{r.signal_version}" in result.rejected_keys
        ],
        "paper_exports": [
            {
                "export_id": exp.export_id,
                "signal_id": exp.signal_id,
                "signal_version": exp.signal_version,
                "mode": exp.mode,
                "execution_enabled": exp.execution_enabled,
                "as_of_utc": exp.as_of_utc.isoformat(),
                "records_count": exp.records_count,
                "gross_exposure": exp.gross_exposure,
                "net_exposure": exp.net_exposure,
                "sha256_digest": exp.sha256_digest,
            }
            for exp in result.paper_exports
        ],
    }
    _write_json(reports_dir / "m4_catalog_and_paper_export.json", m4_report)
    _write_json(reports_dir / "engine_metrics.json", result.metrics)

    # Print summary table
    print("\n=== Advanced Alpha Generation Engine — M1–M4 Reproducible Demo ===")
    print(f"Dataset: {result.dataset.manifest.dataset_id} (digest={result.dataset.manifest.manifest_digest[:12]})")
    print(f"Execution Enabled: {pipeline.engine_config.execution_enabled} (mode={pipeline.engine_config.execution_mode})")
    print("-" * 114)
    print(
        f"{'Signal ID':26s} {'Ver':6s} {'OOS IC':>8s} {'ICIR':>7s} {'BH p':>7s} "
        f"{'Turn':>6s} {'SecExp':>7s} {'NetSh':>7s} {'Stable':>7s} {'Status':>10s}"
    )
    print("-" * 114)
    for r in sorted(result.reports, key=lambda x: x.oos_icir, reverse=True):
        key = f"{r.signal_id}@{r.signal_version}"
        status = "PROMOTED" if key in result.promoted_keys else "REJECTED"
        print(
            f"{r.signal_id:26s} {r.signal_version:6s} {r.oos_mean_ic:+8.4f} {r.oos_icir:+7.3f} "
            f"{r.bh_adjusted_pvalue:7.4f} {r.avg_daily_turnover:6.3f} {r.max_abs_sector_exposure:7.4f} "
            f"{r.net_sharpe_at_base_cost:+7.2f} {str(r.stability.is_stable):>7s} {status:>10s}"
        )
    print("-" * 114)
    print(f"Promoted ({len(result.promoted_keys)}): {result.promoted_keys}")
    print(f"Rejected / Negative Results Recorded ({len(result.rejected_keys)}): {result.rejected_keys}")
    print(f"Paper-only exports written to: {export_dir}")
    print(f"Reports written to: {reports_dir}\n")
    return m4_report


if __name__ == "__main__":
    run_demo()
