"""Command-line interface for running the Multi-Strategy PMS locally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn

from pms.api import create_app
from pms.reporting.cod import render_cod_markdown
from pms.service import PortfolioService


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pms",
        description="SAAT Multi-Strategy Portfolio Management System CLI",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=_default_root(),
        help="Repository root directory containing config/ and data/fixtures/",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Optional SQLite database file path",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve_cmd = sub.add_parser("serve", help="Run the FastAPI server and web UI")
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8080)

    cod_cmd = sub.add_parser("cod", help="Generate Close-of-Day report")
    cod_cmd.add_argument("--format", choices=["json", "markdown"], default="markdown")

    sub.add_parser("verify-recovery", help="Clear read-model cache and verify deterministic replay")
    sub.add_parser("health", help="Print health, controls, and integrity summary")

    args = parser.parse_args(argv)
    service = PortfolioService.from_workspace(args.root, sqlite_path=args.db, load_fixtures=True)

    if args.command == "serve":
        app = create_app(service)
        uvicorn.run(app, host=args.host, port=args.port)
        return 0
    if args.command == "cod":
        pkg = service.build_cod_report()
        if args.format == "markdown":
            print(render_cod_markdown(pkg))
        else:
            print(json.dumps(pkg.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    if args.command == "verify-recovery":
        print(json.dumps(service.verify_recovery_via_replay(), indent=2, sort_keys=True))
        return 0
    if args.command == "health":
        print(json.dumps(service.health(), indent=2, sort_keys=True))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
