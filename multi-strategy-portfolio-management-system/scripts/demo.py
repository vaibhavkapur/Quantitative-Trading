"""Reproducible end-to-end demonstration and benchmark script for the Multi-Strategy PMS."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from pms.adapters.paper_oms import PaperFillReport
from pms.domain.enums import Role, Side
from pms.reporting.cod import write_cod_artifacts
from pms.service import PortfolioService


def run_demo(root: Path) -> dict[str, object]:
    evidence_dir = root / "reports" / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    db_path = evidence_dir / "pms_demo.sqlite3"
    if db_path.exists():
        db_path.unlink()

    t_start = time.perf_counter()
    service = PortfolioService.from_workspace(root, sqlite_path=db_path, load_fixtures=True)
    t_loaded = time.perf_counter()

    # 1. Verify Paper OMS deduplication on a synthetic late paper fill + replay duplicate
    paper_fill = PaperFillReport(
        order_id="ORD-DEMO-4001",
        execution_id="EXEC-DEMO-4001-1",
        account_id="ACCT-STRAT-MOM",
        instrument_id="EQ-AAPL-US",
        side=Side.BUY,
        quantity=Decimal("10"),
        price=Decimal("98.00"),
        currency="USD",
        fee=Decimal("0.50"),
        fill_time=datetime(2026, 1, 15, 20, 45, 0, tzinfo=UTC),
    )
    oms_first = service.ingest_paper_fill(paper_fill, role=Role.OPERATOR)
    oms_dup = service.ingest_paper_fill(paper_fill, role=Role.OPERATOR)

    # 2. Generate Close-of-Day report (before break resolution, showing open breaks + unpriced flag)
    cod_pkg = service.build_cod_report()
    write_cod_artifacts(
        cod_pkg,
        json_path=evidence_dir / "cod_report.json",
        markdown_path=evidence_dir / "cod_report.md",
    )

    # 3. Verify disaster recovery via materialized cache wipe + deterministic replay
    recovery = service.verify_recovery_via_replay()

    # 4. Measure replay latency over 50 iterations (explicit sample size & environment)
    replay_times_ms: list[float] = []
    for _ in range(50):
        t0 = time.perf_counter()
        service.replay()
        replay_times_ms.append((time.perf_counter() - t0) * 1000.0)
    replay_times_ms.sort()

    summary = {
        "demo_executed_at": datetime.now(UTC).isoformat(),
        "fund_id": cod_pkg.fund_id,
        "event_count": cod_pkg.ledger_event_count,
        "ledger_state_hash": cod_pkg.ledger_state_hash,
        "hash_chain_valid": cod_pkg.hash_chain_valid,
        "fund_nav_base": str(cod_pkg.valuation.fund_nav_base),
        "explained_total_pnl_base": str(cod_pkg.valuation.pnl.explained_total_pnl_base),
        "unexplained_residual_base": str(cod_pkg.valuation.pnl.unexplained_residual_base),
        "consolidation_passed": cod_pkg.valuation.consolidation_check.passed,
        "unpriced_instruments": list(cod_pkg.valuation.unpriced_instruments),
        "paper_oms_first": oms_first.model_dump(mode="json"),
        "paper_oms_duplicate": oms_dup.model_dump(mode="json"),
        "reconciliation_breaks_detected": len(cod_pkg.reconciliation.breaks) if cod_pkg.reconciliation else 0,
        "reconciliation_mutated_history": cod_pkg.reconciliation.mutated_history if cod_pkg.reconciliation else False,
        "recovery_check": recovery,
        "benchmark": {
            "sample_size": 50,
            "events_per_replay": cod_pkg.ledger_event_count,
            "bootstrap_ms": round((t_loaded - t_start) * 1000.0, 3),
            "replay_median_ms": round(replay_times_ms[25], 3),
            "replay_p95_ms": round(replay_times_ms[47], 3),
            "replay_max_ms": round(replay_times_ms[-1], 3),
            "note": "Local single-process SQLite prototype measurement; not a production throughput claim.",
        },
    }
    (evidence_dir / "demo_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    result = run_demo(project_root)
    print(json.dumps(result, indent=2, sort_keys=True))
