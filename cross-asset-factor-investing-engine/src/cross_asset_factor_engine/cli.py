"""Command-line interface for running, validating, and replaying cross-asset factor evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from cross_asset_factor_engine.config import load_config
from cross_asset_factor_engine.diagnostics import DiagnosticTracker
from cross_asset_factor_engine.errors import CrossAssetEngineError
from cross_asset_factor_engine.evaluation import run_walk_forward_evaluation
from cross_asset_factor_engine.io import load_dataset_from_dir
from cross_asset_factor_engine.reporting import (
    make_run_id,
    save_evaluation_report,
    verify_replay_equivalence,
)


def _resolve_relative_path(raw_path: str, base_dir: Path) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    candidate = base_dir / p
    if candidate.exists():
        return candidate.resolve()
    return (base_dir / p).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cafe",
        description="Cross-Asset Factor Investing Engine CLI (offline research prototype).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run walk-forward evaluation and write versioned reports.")
    run_p.add_argument("--config", required=True, help="Path to YAML configuration file.")
    run_p.add_argument("--output-dir", default=None, help="Override output directory for reports.")
    run_p.add_argument(
        "--project-root",
        default=None,
        help="Optional root directory for resolving relative dataset/output paths.",
    )

    val_p = sub.add_parser("validate", help="Validate canonical dataset directory against data contracts.")
    val_p.add_argument("--dataset", required=True, help="Path to dataset directory.")
    val_p.add_argument("--base-currency", default="USD", help="Portfolio base currency (default: USD).")

    rep_p = sub.add_parser("verify-replay", help="Execute two independent runs and assert deterministic replay.")
    rep_p.add_argument("--config", required=True, help="Path to YAML configuration file.")
    rep_p.add_argument("--project-root", default=None, help="Optional root directory for relative paths.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "validate":
            ds = load_dataset_from_dir(args.dataset, base_currency=args.base_currency)
            print(
                json.dumps(
                    {
                        "status": "VALID",
                        "dataset_hash": ds.dataset_hash,
                        "schema_version": ds.schema_version,
                        "n_assets": ds.contract_metadata.height,
                        "n_price_rows": ds.prices.height,
                        "n_fx_rows": ds.fx_rates.height,
                        "n_factor_input_rows": ds.factor_inputs.height,
                    },
                    indent=2,
                )
            )
            return 0

        cfg_path = Path(args.config).resolve()
        cfg = load_config(cfg_path)
        project_root = (
            Path(args.project_root).resolve()
            if getattr(args, "project_root", None)
            else cfg_path.parent.parent
        )
        dataset_dir = _resolve_relative_path(cfg.data.dataset_dir, project_root)
        dataset = load_dataset_from_dir(dataset_dir, base_currency=cfg.base_currency)

        if args.command == "run":
            diag = DiagnosticTracker(deterministic_clock=True)
            result = run_walk_forward_evaluation(dataset, cfg, diagnostics=diag)
            out_dir = (
                Path(args.output_dir).resolve()
                if args.output_dir
                else _resolve_relative_path(cfg.reporting.output_dir, project_root)
            )
            run_dir = save_evaluation_report(result, out_dir)
            print(
                json.dumps(
                    {
                        "status": "OK",
                        "run_id": make_run_id(result),
                        "report_dir": str(run_dir),
                        "strategy_metrics": result.strategy_metrics,
                    },
                    indent=2,
                )
            )
            return 0

        if args.command == "verify-replay":
            res_a = run_walk_forward_evaluation(dataset, cfg, diagnostics=DiagnosticTracker(deterministic_clock=True))
            res_b = run_walk_forward_evaluation(dataset, cfg, diagnostics=DiagnosticTracker(deterministic_clock=True))
            check = verify_replay_equivalence(res_a, res_b)
            print(json.dumps(check, indent=2))
            return 0 if check.get("equivalent") else 2

    except CrossAssetEngineError as exc:
        print(json.dumps({"status": "ERROR", "error": exc.to_dict()}, indent=2), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
