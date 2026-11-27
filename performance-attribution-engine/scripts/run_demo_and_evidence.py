#!/usr/bin/env python3
"""Reproducible demo, benchmark baseline, and automated validation evidence runner."""

from __future__ import annotations

import argparse
import io
import json
import platform
import statistics
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pae.api.service import AttributionService
from pae.config import AttributionConfig
from pae.domain.models import AttributionInputBatch
from pae.engine.pipeline import AttributionPipeline
from pae.reporting.exporter import write_report_bundle
from pae.security.rbac import Actor, Role
from pae.storage.sqlite_store import SQLiteAttributionStore


def run_demo(reports_dir: Path) -> Dict[str, Any]:
    """Execute end-to-end demo: v1 baseline -> review -> approve -> v2 fee correction -> exports."""
    fixtures_dir = ROOT / "data" / "fixtures"
    v1_raw = json.loads((fixtures_dir / "multi_strategy_multi_ccy_baseline.json").read_text(encoding="utf-8"))
    v2_raw = json.loads((fixtures_dir / "multi_strategy_fee_correction_v2.json").read_text(encoding="utf-8"))
    hw_raw = json.loads((fixtures_dir / "hand_worked_portfolio.json").read_text(encoding="utf-8"))

    v1_batch = AttributionInputBatch.from_dict(v1_raw)
    v2_batch = AttributionInputBatch.from_dict(v2_raw)
    hw_batch = AttributionInputBatch.from_dict(hw_raw)

    db_path = reports_dir / "demo_pae_state.sqlite3"
    if db_path.exists():
        db_path.unlink()

    cfg = AttributionConfig.from_file(ROOT / "config" / "default_config.json")
    store = SQLiteAttributionStore(str(db_path))
    svc = AttributionService(config=cfg, store=store)

    analyst = Actor("analyst-01", Role.ANALYST)
    reviewer = Actor("reviewer-01", Role.REVIEWER)
    approver = Actor("approver-01", Role.APPROVER)

    # 1. Run v1 baseline, review, approve
    v1_rec = svc.run_attribution(v1_batch, actor=analyst)
    v1_id = v1_rec["version_id"]
    svc.review_report(v1_id, actor=reviewer, comment="Verified reconciliation, Brinson, factor, and TCA figures.")
    v1_rec = svc.approve_report(v1_id, actor=approver, comment="Approved for synthetic prototype release.")

    # 2. Apply retroactive fee correction creating v2 superseding v1
    v2_rec = svc.run_attribution(
        v2_batch,
        actor=analyst,
        report_id=v1_rec["report_id"],
        correction_reason="Retroactive prime-broker commission rebate and fee restatement on 2026-01-09",
    )
    v2_id = v2_rec["version_id"]
    svc.review_report(v2_id, actor=reviewer, comment="Confirmed delta net return and fee rebate linkage.")
    v2_rec = svc.approve_report(v2_id, actor=approver, comment="Approved revised v2 report.")

    # 3. Also run hand-worked portfolio
    hw_rec = svc.run_attribution(hw_batch, actor=analyst)

    # 4. Write export bundles
    exports_dir = reports_dir / "demo_exports"
    if exports_dir.exists():
        for p in exports_dir.iterdir():
            if p.is_file():
                p.unlink()
    v1_files = write_report_bundle(v1_rec["report"], exports_dir)
    v2_files = write_report_bundle(v2_rec["report"], exports_dir)
    hw_files = write_report_bundle(hw_rec["report"], exports_dir)

    # 5. Baseline latency measurement (N=25 deterministic runs on 10-day multi-ccy fixture)
    pipeline = AttributionPipeline(cfg)
    latencies_ms: List[float] = []
    for _ in range(25):
        t0 = time.perf_counter()
        pipeline.run(v1_batch)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    audit_info = svc.get_audit_log(actor=approver)
    health = svc.health()
    store.close()

    return {
        "v1": {
            "version_id": v1_id,
            "reconciliation_status": v1_rec["reconciliation_status"],
            "workflow_status": v1_rec["workflow"]["status"],
            "cumulative_gross_return": v1_rec["report"]["returns"]["cumulative_gross_return"],
            "cumulative_net_return": v1_rec["report"]["returns"]["cumulative_net_return"],
            "linked_net_residual": v1_rec["report"]["reconciliation"]["linked_net_residual"],
            "brinson_status": v1_rec["report"]["brinson"]["status"],
            "factor_status": v1_rec["report"]["factors"]["status"],
            "tca_total_is_bps": v1_rec["report"]["execution_costs"]["total_is_bps_of_notional"],
            "exports": v1_files,
        },
        "v2": {
            "version_id": v2_id,
            "supersedes_version_id": v2_rec["supersedes_version_id"],
            "reconciliation_status": v2_rec["reconciliation_status"],
            "workflow_status": v2_rec["workflow"]["status"],
            "cumulative_gross_return": v2_rec["report"]["returns"]["cumulative_gross_return"],
            "cumulative_net_return": v2_rec["report"]["returns"]["cumulative_net_return"],
            "linked_net_residual": v2_rec["report"]["reconciliation"]["linked_net_residual"],
            "revision_diff": v2_rec["report"]["revision_diff"],
            "exports": v2_files,
        },
        "hand_worked": {
            "version_id": hw_rec["version_id"],
            "reconciliation_status": hw_rec["reconciliation_status"],
            "cumulative_gross_return": hw_rec["report"]["returns"]["cumulative_gross_return"],
            "cumulative_net_return": hw_rec["report"]["returns"]["cumulative_net_return"],
            "linked_net_residual": hw_rec["report"]["reconciliation"]["linked_net_residual"],
            "exports": hw_files,
        },
        "audit_chain_valid": audit_info["chain_valid"],
        "audit_entry_count": audit_info["entry_count"],
        "health": health,
        "baseline_benchmark": {
            "environment": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
            "workload": "10 trading days, 4 multi-currency instruments, 3 strategies, 4 trades, Brinson + 4-factor OLS + TCA",
            "sample_size": len(latencies_ms),
            "median_ms": round(statistics.median(latencies_ms), 3),
            "mean_ms": round(statistics.mean(latencies_ms), 3),
            "stdev_ms": round(statistics.stdev(latencies_ms), 3),
            "min_ms": round(min(latencies_ms), 3),
            "max_ms": round(max(latencies_ms), 3),
        },
    }


def run_unittest_suite(reports_dir: Path) -> Dict[str, Any]:
    """Run all unit, integration, and scenario tests and retain raw test_report.txt."""
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / "tests"), pattern="test_*.py")
    buf = io.StringIO()
    runner = unittest.TextTestRunner(stream=buf, verbosity=2)
    t0 = time.perf_counter()
    result = runner.run(suite)
    elapsed_s = time.perf_counter() - t0
    raw_output = buf.getvalue()

    (reports_dir / "test_report.txt").write_text(raw_output, encoding="utf-8")
    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "was_successful": result.wasSuccessful(),
        "elapsed_seconds": round(elapsed_s, 3),
    }


def write_validation_reports(reports_dir: Path, test_summary: Dict[str, Any], demo_summary: Dict[str, Any]) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "methodology_version": AttributionConfig.from_file(ROOT / "config" / "default_config.json").methodology_version,
        "tests": test_summary,
        "demo": demo_summary,
    }
    (reports_dir / "validation_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    bm = demo_summary["baseline_benchmark"]
    v1 = demo_summary["v1"]
    v2 = demo_summary["v2"]
    hw = demo_summary["hand_worked"]

    md = [
        "# Performance Attribution Engine — Validation & Benchmark Report",
        "",
        f"- **Generated (UTC):** `{payload['generated_at_utc']}`",
        f"- **Methodology Version:** `{payload['methodology_version']}`",
        f"- **Environment:** Python `{bm['environment']['python_version']}` on `{bm['environment']['platform']}`",
        f"- **Order Execution Enabled:** `{demo_summary['health']['order_execution_enabled']}` (strictly disabled)",
        "",
        "## 1. Automated Test Suite Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total Tests Executed | **{test_summary['tests_run']}** |",
        f"| Failures | **{test_summary['failures']}** |",
        f"| Errors | **{test_summary['errors']}** |",
        f"| Skipped | {test_summary['skipped']} |",
        f"| Overall Result | **{'PASS' if test_summary['was_successful'] else 'FAIL'}** |",
        f"| Runtime | {test_summary['elapsed_seconds']}s |",
        "",
        "Raw runner transcript is retained in [`reports/test_report.txt`](test_report.txt).",
        "",
        "## 2. Reproducible End-to-End Demo & Versioning Evidence",
        "",
        "| Artifact | Version ID | Reconciliation | Gross Return | Net Return | Linked Net Residual |",
        "|---|---|---|---:|---:|---:|",
        f"| Multi-Strategy v1 Baseline | `{v1['version_id']}` | `{v1['reconciliation_status']}` | {v1['cumulative_gross_return']*100:+.4f}% | {v1['cumulative_net_return']*100:+.4f}% | {v1['linked_net_residual']*10000:+.6f} bps |",
        f"| Multi-Strategy v2 Fee Correction | `{v2['version_id']}` | `{v2['reconciliation_status']}` | {v2['cumulative_gross_return']*100:+.4f}% | {v2['cumulative_net_return']*100:+.4f}% | {v2['linked_net_residual']*10000:+.6f} bps |",
        f"| Hand-Worked Portfolio | `{hw['version_id']}` | `{hw['reconciliation_status']}` | {hw['cumulative_gross_return']*100:+.4f}% | {hw['cumulative_net_return']*100:+.4f}% | {hw['linked_net_residual']*10000:+.6f} bps |",
        "",
        f"- **v2 Supersedes:** `{v2['supersedes_version_id']}`",
        f"- **v2 Delta Net Return:** `{v2['revision_diff']['delta_cumulative_net_return']*100:+.6f}%` (Delta Fees: `{v2['revision_diff']['delta_total_fees_base']:+.2f} USD`)",
        f"- **Audit Hash Chain Verified:** `{demo_summary['audit_chain_valid']}` ({demo_summary['audit_entry_count']} entries)",
        "",
        "## 3. Baseline Performance Measurement (Local Synthetic Prototype)",
        "",
        f"- **Workload:** {bm['workload']}",
        f"- **Sample Size:** N = {bm['sample_size']} runs",
        f"- **Latency (ms):** median `{bm['median_ms']} ms`, mean `{bm['mean_ms']} ms`, stdev `{bm['stdev_ms']} ms` (min `{bm['min_ms']} ms`, max `{bm['max_ms']} ms`)",
        "",
        "_Note: Measurements reflect synthetic local execution only; live/production throughput targets require representative production sizing._",
        "",
    ]
    (reports_dir / "validation_report.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-only", action="store_true")
    args = parser.parse_args()

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    demo_summary = run_demo(reports_dir)
    print(f"[demo] v1={demo_summary['v1']['version_id']} ({demo_summary['v1']['reconciliation_status']})")
    print(f"[demo] v2={demo_summary['v2']['version_id']} supersedes {demo_summary['v2']['supersedes_version_id']}")
    print(f"[demo] hand_worked={demo_summary['hand_worked']['version_id']} ({demo_summary['hand_worked']['reconciliation_status']})")

    if not args.demo_only:
        test_summary = run_unittest_suite(reports_dir)
        write_validation_reports(reports_dir, test_summary, demo_summary)
        print(
            f"[tests] ran={test_summary['tests_run']} failures={test_summary['failures']} "
            f"errors={test_summary['errors']} ({test_summary['elapsed_seconds']}s)"
        )
        if not test_summary["was_successful"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
