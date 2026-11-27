"""Zero-dependency stdlib HTTP REST API & interactive dashboard server with in-process router."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from pae.api.service import AttributionService
from pae.config import AttributionConfig
from pae.domain.models import AttributionInputBatch
from pae.errors import PAEError
from pae.security.rbac import Actor
from pae.web.dashboard import render_dashboard_index, render_report_html


def dispatch_request(
    service: AttributionService,
    method: str,
    raw_path: str,
    headers: Optional[Dict[str, str]] = None,
    raw_body: Optional[str] = None,
) -> Tuple[int, str, str]:
    """Pure in-process HTTP request dispatcher returning (status_code, content_type, body_str).

    Used identically by ThreadingHTTPServer and sandboxed integration tests.
    """
    hdrs = {k.lower(): v for k, v in (headers or {}).items()}
    parsed = urlparse(raw_path)
    path = parsed.path.rstrip("/") or "/"
    qs = parse_qs(parsed.query)

    def get_actor() -> Actor:
        actor_id = hdrs.get("x-actor-id", "local-viewer")
        role_str = hdrs.get("x-actor-role", "VIEWER")
        return Actor.from_headers(actor_id, role_str)

    def json_resp(status: int, payload: Any) -> Tuple[int, str, str]:
        return status, "application/json", json.dumps(payload, indent=2, sort_keys=True)

    try:
        m = method.upper()
        if m == "GET":
            if path in {"/", "/dashboard"}:
                reports = service.store.list_reports()
                return 200, "text/html", render_dashboard_index(reports, service.health())
            if path == "/api/v1/health":
                return json_resp(200, service.health())
            if path == "/api/v1/reports":
                actor = get_actor()
                portfolio_id = qs.get("portfolio_id", [None])[0]
                report_id = qs.get("report_id", [None])[0]
                return json_resp(
                    200,
                    {"reports": service.list_reports(actor=actor, portfolio_id=portfolio_id, report_id=report_id)},
                )
            if path.startswith("/reports/") and path.endswith("/html"):
                version_id = path[len("/reports/") : -len("/html")]
                rec = service.store.get_report_record(version_id)
                return 200, "text/html", render_report_html(rec["report"])
            if path.startswith("/api/v1/reports/"):
                parts = path.split("/")
                version_id = parts[4]
                actor = get_actor()
                if len(parts) == 5:
                    return json_resp(200, service.get_report(version_id, actor=actor))
                if len(parts) == 6 and parts[5] == "export":
                    fmt = qs.get("format", ["json"])[0]
                    content = service.export_report(version_id, fmt, actor=actor)
                    ctype = {
                        "json": "application/json",
                        "md": "text/markdown",
                        "markdown": "text/markdown",
                        "csv": "text/csv",
                        "html": "text/html",
                    }.get(fmt.lower(), "text/plain")
                    return 200, ctype, content
            if path == "/api/v1/audit":
                actor = get_actor()
                return json_resp(200, service.get_audit_log(actor=actor))
            return json_resp(404, {"error_code": "NOT_FOUND", "message": f"Unknown route: {path}"})

        if m == "POST":
            actor = get_actor()
            body: Dict[str, Any] = json.loads(raw_body) if raw_body else {}
            if path == "/api/v1/attribution/run":
                batch_dict = body.get("batch", body)
                batch = AttributionInputBatch.from_dict(batch_dict)
                res = service.run_attribution(
                    batch,
                    actor=actor,
                    report_id=body.get("report_id"),
                    correction_reason=body.get("correction_reason"),
                )
                return json_resp(201, res)
            if path.startswith("/api/v1/reports/") and path.endswith("/review"):
                version_id = path[len("/api/v1/reports/") : -len("/review")]
                res = service.review_report(version_id, actor=actor, comment=body.get("comment", ""))
                return json_resp(200, res)
            if path.startswith("/api/v1/reports/") and path.endswith("/approve"):
                version_id = path[len("/api/v1/reports/") : -len("/approve")]
                res = service.approve_report(version_id, actor=actor, comment=body.get("comment", ""))
                return json_resp(200, res)
            return json_resp(404, {"error_code": "NOT_FOUND", "message": f"Unknown route: {path}"})

        return json_resp(405, {"error_code": "METHOD_NOT_ALLOWED", "message": f"Unsupported method: {m}"})
    except PAEError as exc:
        return json_resp(exc.http_status, exc.to_dict())
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        return json_resp(400, {"error_code": "BAD_REQUEST", "message": str(exc)})


def create_http_server(
    service: AttributionService,
    host: str = "127.0.0.1",
    port: int = 8090,
) -> ThreadingHTTPServer:
    """Create a ThreadingHTTPServer bound to the given AttributionService."""

    class PAEHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _handle(self, method: str) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else None
            hdrs = {k: v for k, v in self.headers.items()}
            status, ctype, content = dispatch_request(service, method, self.path, hdrs, raw_body)
            raw = content.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            self._handle("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._handle("POST")

    return ThreadingHTTPServer((host, port), PAEHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Performance Attribution Engine HTTP Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = AttributionConfig.from_file(args.config) if args.config else AttributionConfig.from_file()
    svc = AttributionService(config=cfg)
    srv = create_http_server(svc, host=args.host, port=args.port)
    print(f"PAE server listening on http://{args.host}:{args.port} (methodology={cfg.methodology_version})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
