"""Command-line interface for the Volatility Forecasting Engine."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

from volatility_forecasting.contracts import (
    read_table,
    validate_intraday_bars_df,
    validate_labels_df,
    validate_returns_df,
)
from volatility_forecasting.errors import VolatilityEngineError
from volatility_forecasting.fixtures import write_all_fixtures
from volatility_forecasting.pipeline import run_pipeline


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vol-forecast",
        description="Causal realized-volatility forecasting engine CLI.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run chronological volatility forecasting pipeline.")
    p_run.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file.",
    )
    p_run.add_argument(
        "--base-dir",
        default=None,
        help="Optional project base directory for resolving relative paths.",
    )
    p_run.add_argument(
        "--no-export",
        action="store_true",
        help="Run evaluation without writing report files to disk.",
    )

    p_fix = sub.add_parser(
        "generate-fixtures",
        help="Generate deterministic synthetic, reference, boundary, and invalid fixtures.",
    )
    p_fix.add_argument(
        "--out-dir",
        default="data/fixtures",
        help="Output directory for generated fixture files.",
    )

    p_val = sub.add_parser(
        "validate-data",
        help="Validate input CSV/Parquet files against canonical data contracts.",
    )
    p_val.add_argument("--returns", required=True, help="Path to timestamped returns table.")
    p_val.add_argument("--intraday", default=None, help="Optional path to intraday bars table.")
    p_val.add_argument("--labels", default=None, help="Optional path to realized vol labels table.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    try:
        if args.command == "generate-fixtures":
            written = write_all_fixtures(args.out_dir)
            print(json.dumps({"status": "ok", "fixtures": written}, indent=2, sort_keys=True))
            return 0

        if args.command == "validate-data":
            ret_df = validate_returns_df(read_table(args.returns))
            summary = {"status": "ok", "returns_rows": ret_df.height}
            if args.intraday:
                bars_df = validate_intraday_bars_df(read_table(args.intraday))
                summary["intraday_rows"] = bars_df.height
            if args.labels:
                lbl_df = validate_labels_df(read_table(args.labels))
                summary["labels_rows"] = lbl_df.height
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0

        if args.command == "run":
            base_dir = Path(args.base_dir).resolve() if args.base_dir else None
            result = run_pipeline(
                args.config,
                base_dir=base_dir,
                export_reports=not args.no_export,
            )
            payload = {
                "status": "ok",
                "run_id": result.run_id,
                "config_fingerprint": result.config_fingerprint,
                "horizon_days": result.horizon_days,
                "counters": result.counters,
                "output_paths": result.output_paths,
                "model_comparison": result.model_comparison.select(
                    [
                        "model",
                        "qlike",
                        "qlike_vs_baseline_pct",
                        "mae_vol",
                        "mae_vol_vs_baseline_pct",
                        "mz_slope",
                        "min_pred_var",
                    ]
                ).to_dicts(),
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0

    except VolatilityEngineError as exc:
        print(
            json.dumps(
                {"status": "error", "error_type": exc.__class__.__name__, "message": str(exc)},
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
