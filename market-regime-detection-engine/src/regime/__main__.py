import argparse
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

from .service import server
from .store import packed
from .workflow import run

ROOT = Path(__file__).resolve().parents[2]


def main():
    cli = argparse.ArgumentParser(description="Market regime detection engine")
    commands = cli.add_subparsers(dest="command", required=True)
    build = commands.add_parser("run", help="train, replay and publish a history")
    build.add_argument("--bars", type=Path, default=ROOT / "data/fixtures/daily.csv")
    build.add_argument("--settings", type=Path, default=ROOT / "config/daily.v1.json")
    build.add_argument("--report", type=Path)
    serve = commands.add_parser("serve", help="serve the read-only loopback API")
    serve.add_argument("--port", type=int, default=8765)
    for command in (build, serve):
        command.add_argument("--db", type=Path, default=ROOT / ".local/history.sqlite3")
    args = cli.parse_args()
    try:
        if args.command == "serve":
            with server(args.db, os.environ.get("REGIME_READ_TOKEN"), args.port) as http:
                print(f"Serving on http://127.0.0.1:{http.server_port}", flush=True)
                http.serve_forever()
        else:
            report = run(args.bars, args.settings, args.db)
            if args.report:
                value = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
                args.report.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(mode="w", dir=args.report.parent, delete=False) as out:
                    out.write(value)
                os.replace(out.name, args.report)
            print(packed({k: report[k] for k in ("run_id", "training_bars", "holdout_bars", "stability")}))
    except (ValueError, OSError, sqlite3.Error, OverflowError) as exc:
        print(packed({"error": "run_failed", "detail": str(exc)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
