"""Command-line interface for the Multi-Factor Asset Pricing Engine."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

from mfape.engine import AssetPricingEngine
from mfape.errors import MFAPEError
from mfape.io import load_config


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mfape",
        description="Multi-Factor Asset Pricing Engine — time-series, Fama-MacBeth, and OOS evaluation.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_p = subparsers.add_parser("run", help="Run the asset-pricing engine from a JSON config file.")
    run_p.add_argument("--config", required=True, help="Path to JSON configuration file.")
    run_p.add_argument(
        "--base-dir",
        default=".",
        help="Base directory for resolving relative paths in config (default: current directory).",
    )
    run_p.add_argument(
        "--no-write",
        action="store_true",
        help="Execute without writing report artifacts to disk.",
    )

    val_p = subparsers.add_parser("validate", help="Validate a JSON configuration file without running models.")
    val_p.add_argument("--config", required=True, help="Path to JSON configuration file.")

    rep_p = subparsers.add_parser("replay", help="Replay a previous run from replay_manifest.json.")
    rep_p.add_argument("--manifest", required=True, help="Path to replay_manifest.json.")
    rep_p.add_argument("--base-dir", default=".", help="Base directory for resolving relative input paths.")
    rep_p.add_argument(
        "--skip-fingerprint-check",
        action="store_true",
        help="Allow replay even if input file SHA-256 hashes changed.",
    )
    rep_p.add_argument(
        "--write",
        action="store_true",
        help="Rewrite outputs to the configured output_dir during replay.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    try:
        if args.command == "validate":
            cfg = load_config(args.config)
            payload = {
                "status": "valid",
                "run_label": cfg.run_label,
                "models": [m.name for m in cfg.models],
                "baseline_model_name": cfg.baseline_model_name,
            }
            print(json.dumps(payload, indent=2))
            return 0

        engine = AssetPricingEngine()

        if args.command == "run":
            output, written = engine.run_from_config_file(
                config_path=args.config,
                base_dir=Path(args.base_dir),
                write_outputs=not args.no_write,
            )
            summary = {
                "status": "ok",
                "run_id": output.run_id,
                "run_label": output.run_label,
                "models_evaluated": [m.spec.name for m in output.models],
                "written_artifacts": written,
            }
            print(json.dumps(summary, indent=2))
            return 0

        if args.command == "replay":
            output, written = engine.replay_from_manifest(
                manifest_path=args.manifest,
                base_dir=Path(args.base_dir),
                verify_fingerprints=not args.skip_fingerprint_check,
                write_outputs=args.write,
            )
            summary = {
                "status": "replayed_ok",
                "run_id": output.run_id,
                "run_label": output.run_label,
                "fingerprints_verified": not args.skip_fingerprint_check,
                "written_artifacts": written,
            }
            print(json.dumps(summary, indent=2))
            return 0

        parser.error(f"Unknown command: {args.command}")
        return 2
    except MFAPEError as exc:
        print(json.dumps(exc.to_dict(), indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
