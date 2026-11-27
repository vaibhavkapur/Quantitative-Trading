import hmac
import json
import re
import sqlite3
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from .domain import ContractError, timestamp
from .store import page, packed, reader


def dispatch(db_path, token, header, url):
    request = urlsplit(url)
    if request.path != "/health" and not hmac.compare_digest(
            (header or "").encode(), ("Bearer " + token).encode()):
        return 401, {"error": "unauthorized"}
    try:
        query = parse_qs(request.query, keep_blank_values=True, strict_parsing=True, max_num_fields=3) if request.query else {}
        with closing(reader(db_path)) as db:
            if request.path == "/health":
                count = db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
                return (200 if count else 503), {"ready": bool(count)}
            if request.path == "/v1/runs" and not query:
                return 200, {"run_ids": [r[0] for r in db.execute("SELECT id FROM runs ORDER BY id")]}
            match = re.fullmatch(r"/v1/runs/([0-9a-f]{64})/(history|features|report|metadata)", request.path)
            if not match:
                return 404, {"error": "not_found"}
            run_id, view = match.groups()
            run = db.execute("SELECT meta,report FROM runs WHERE id=?", (run_id,)).fetchone()
            if run is None:
                return 404, {"error": "run_not_found"}
            if view in ("report", "metadata"):
                if query:
                    raise ContractError("unexpected query parameters")
                return 200, json.loads(run["report" if view == "report" else "meta"])
            if set(query) - {"after", "as_of", "limit"} or any(len(v) != 1 for v in query.values()):
                raise ContractError("invalid page parameters")
            after = query.get("after", [""])[0]
            as_of = query.get("as_of", ["9999-12-31T23:59:59Z"])[0]
            limit = int(query.get("limit", ["100"])[0])
            if "after" in query:
                timestamp(after)
            timestamp(as_of)
            if not 1 <= limit <= 500 or after > as_of:
                raise ContractError("invalid page bounds")
            return 200, page(db, run_id, after, as_of, limit, view == "features")
    except (ContractError, ValueError) as exc:
        return 400, {"error": "invalid_request", "detail": str(exc)}
    except sqlite3.Error:
        return 503, {"error": "history_unavailable"}


def server(db_path, token, port):
    if not token or len(token) < 16 or not token.isascii() or any(ch.isspace() for ch in token):
        raise ContractError("REGIME_READ_TOKEN must contain at least 16 non-whitespace ASCII characters")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            status, value = dispatch(db_path, token, self.headers.get("Authorization"), self.path)
            body = packed(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if status == 401:
                self.send_header("WWW-Authenticate", "Bearer")
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
