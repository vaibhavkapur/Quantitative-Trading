"""Reproducible end-to-end demo script generating baseline reports for all core scenarios."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from robustness_lab.pipeline import compare_reports_logical_equality, evaluate_from_directory

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "config" / "default_policy.json"
REPORTS_DIR = ROOT / "reports" / "baseline"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    scenarios = [
        ("known_signal", ROOT / "data" / "fixtures" / "valid" / "known_signal_experiment", None),
        ("null_overfit", ROOT / "data" / "fixtures" / "valid" / "null_overfit_experiment", None),
        ("cherry_picked", ROOT / "data" / "fixtures" / "valid" / "cherry_picked_experiment", None),
        ("insufficient_sample", ROOT / "data" / "fixtures" / "boundary" / "insufficient_sample_experiment", None),
        ("extreme_outlier", ROOT / "data" / "fixtures" / "boundary" / "extreme_outlier_experiment", "outlier_t3"),
    ]

    summary_rows: list[dict[str, object]] = []
    for name, exp_dir, cand in scenarios:
        payload = evaluate_from_directory(
            experiment_dir=exp_dir,
            policy_path=POLICY,
            candidate_trial_id=cand,
            output_dir=REPORTS_DIR,
            report_basename=name,
        )
        # Verify deterministic replay immediately
        rerun = evaluate_from_directory(
            experiment_dir=exp_dir,
            policy_path=POLICY,
            candidate_trial_id=payload["candidate_trial_id"],
        )
        matched, diffs = compare_reports_logical_equality(payload, rerun)
        if not matched:
            raise RuntimeError(f"Replay determinism check failed for {name}: {diffs}")

        diag = payload["diagnostics"]
        summary_rows.append(
            {
                "scenario": name,
                "run_id": payload["run_id"],
                "candidate": payload["candidate_trial_id"],
                "decision": payload["verdict"]["decision"],
                "evidence_complete": payload["verdict"]["evidence_complete"],
                "dsr": round(diag["dsr"]["deflated_sharpe_ratio"], 4) if diag["dsr"] else None,
                "pbo": round(diag["pbo"]["pbo"], 4) if diag["pbo"] else None,
                "oos_net_sharpe": round(diag["walk_forward"]["aggregate_oos_net_sharpe"], 4)
                if diag["walk_forward"]
                else None,
                "replay_matched": matched,
            }
        )

    index_path = REPORTS_DIR / "demo_summary.json"
    index_path.write_text(json.dumps(summary_rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary_rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
