"""Command-line interface for the Quantitative Strategy Robustness Lab."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

from .models import RobustnessLabError
from .pipeline import compare_reports_logical_equality, evaluate_from_directory
from .registry import ImmutableExperimentRegistry


def _configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="robustness-lab",
        description="Quantitative Strategy Robustness Lab CLI (offline research & promotion gating).",
    )
    parser.add_argument("--verbose", action="store_true", help="Emit structured JSON diagnostic logs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Run full robustness evaluation on an experiment directory.")
    p_eval.add_argument("--experiment-dir", type=Path, required=True, help="Path to experiment directory.")
    p_eval.add_argument("--policy", type=Path, required=True, help="Path to policy JSON config.")
    p_eval.add_argument("--candidate", type=str, default=None, help="Optional candidate trial_id override.")
    p_eval.add_argument("--output-dir", type=Path, default=None, help="Directory to write JSON and Markdown reports.")
    p_eval.add_argument("--report-name", type=str, default=None, help="Optional output filename stem.")

    # verify-registry
    p_ver = subparsers.add_parser("verify-registry", help="Verify hash chain integrity of a registry.jsonl ledger.")
    p_ver.add_argument("--ledger", type=Path, required=True, help="Path to registry.jsonl.")

    # replay
    p_rep = subparsers.add_parser("replay", help="Re-run an evaluation and verify deterministic match against a baseline JSON.")
    p_rep.add_argument("--experiment-dir", type=Path, required=True)
    p_rep.add_argument("--policy", type=Path, required=True)
    p_rep.add_argument("--baseline-report", type=Path, required=True)
    p_rep.add_argument("--candidate", type=str, default=None)

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    try:
        if args.command == "verify-registry":
            reg = ImmutableExperimentRegistry(args.ledger)
            valid, errors = reg.verify()
            out = {
                "ledger": str(args.ledger),
                "valid": valid,
                "head_hash": reg.head_hash() if valid else None,
                "errors": errors,
            }
            print(json.dumps(out, indent=2))
            return 0 if valid else 2

        if args.command == "evaluate":
            payload = evaluate_from_directory(
                experiment_dir=args.experiment_dir,
                policy_path=args.policy,
                candidate_trial_id=args.candidate,
                output_dir=args.output_dir,
                report_basename=args.report_name,
            )
            summary = {
                "run_id": payload["run_id"],
                "experiment_id": payload["experiment_id"],
                "candidate_trial_id": payload["candidate_trial_id"],
                "decision": payload["verdict"]["decision"],
                "evidence_complete": payload["verdict"]["evidence_complete"],
                "summary": payload["verdict"]["summary"],
            }
            print(json.dumps(summary, indent=2))
            return 0

        if args.command == "replay":
            if not args.baseline_report.exists():
                print(f"Error: baseline report not found: {args.baseline_report}", file=sys.stderr)
                return 2
            baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
            rerun = evaluate_from_directory(
                experiment_dir=args.experiment_dir,
                policy_path=args.policy,
                candidate_trial_id=args.candidate or baseline.get("candidate_trial_id"),
            )
            matched, diffs = compare_reports_logical_equality(baseline, rerun)
            print(json.dumps({"matched": matched, "run_id": rerun["run_id"], "diffs": diffs}, indent=2))
            return 0 if matched else 3

    except RobustnessLabError as exc:
        print(json.dumps({"error_type": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
