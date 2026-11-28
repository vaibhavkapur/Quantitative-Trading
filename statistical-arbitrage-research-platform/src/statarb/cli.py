"""Command-line interface for the Statistical Arbitrage Research Platform."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from statarb.config import load_config
from statarb.contracts import load_and_validate_inputs
from statarb.exceptions import StatArbError
from statarb.fixtures import write_all_fixtures
from statarb.pipeline import execute_research_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="statarb",
        description="Statistical Arbitrage Research Platform CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_fix = sub.add_parser("generate-fixtures", help="Generate deterministic synthetic fixtures")
    p_fix.add_argument(
        "--output-dir",
        default="data/fixtures",
        help="Target directory for generated fixtures (default: data/fixtures)",
    )

    p_val = sub.add_parser("validate", help="Validate configuration and input datasets")
    p_val.add_argument("--config", required=True, help="Path to TOML or JSON configuration file")
    p_val.add_argument("--project-root", default=".", help="Project root for relative data paths")

    p_run = sub.add_parser("run", help="Run walk-forward research pipeline and write reports")
    p_run.add_argument("--config", required=True, help="Path to TOML or JSON configuration file")
    p_run.add_argument("--project-root", default=".", help="Project root for relative data paths")
    p_run.add_argument(
        "--output-dir",
        default="reports",
        help="Directory for versioned machine-readable outputs (default: reports)",
    )
    p_run.add_argument(
        "--no-sweep",
        action="store_true",
        help="Skip multi-configuration z_entry trial sweep",
    )
    p_run.add_argument("--quiet", action="store_true", help="Suppress structured INFO logs")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "generate-fixtures":
            manifest = write_all_fixtures(args.output_dir)
            print(json.dumps(manifest, indent=2))
            return 0

        if args.command == "validate":
            root = Path(args.project_root).resolve()
            cfg = load_config(args.config)
            prices_path = Path(cfg.data.prices_path)
            if not prices_path.is_absolute():
                prices_path = (root / prices_path).resolve()
            borrow_path = Path(cfg.data.borrow_path)
            if not borrow_path.is_absolute():
                borrow_path = (root / borrow_path).resolve()
            _, _, report = load_and_validate_inputs(
                prices_path,
                borrow_path,
                max_missing_fraction=cfg.data.max_missing_fraction,
                missing_price_policy=cfg.data.missing_price_policy,
            )
            print(
                json.dumps(
                    {
                        "status": "VALID",
                        "config_hash": cfg.config_hash(),
                        "prices_rows": report.prices_rows,
                        "borrow_rows": report.borrow_rows,
                        "symbols": list(report.symbols),
                        "date_start": report.date_start,
                        "date_end": report.date_end,
                        "prices_hash": report.prices_hash,
                        "borrow_hash": report.borrow_hash,
                    },
                    indent=2,
                )
            )
            return 0

        if args.command == "run":
            report = execute_research_pipeline(
                args.config,
                project_root=args.project_root,
                output_dir=args.output_dir,
                run_trial_sweep=not args.no_sweep,
                quiet=args.quiet,
            )
            summary = {
                "run_id": report["run_id"],
                "config_hash": report["config_hash"],
                "performance": report["performance"],
                "uncertainty": report["uncertainty"],
                "reconciliation": report["reconciliation"],
                "benchmarks": report["benchmarks"],
            }
            print(json.dumps(summary, indent=2))
            return 0

    except StatArbError as exc:
        print(f"ERROR [{type(exc).__name__}]: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"UNEXPECTED ERROR [{type(exc).__name__}]: {exc}", file=sys.stderr)
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
