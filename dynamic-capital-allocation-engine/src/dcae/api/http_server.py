"""Zero-dependency HTTP JSON API service for DCAE."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

from dcae.audit.store import SQLiteAuditStore
from dcae.domain.models import EngineConfig, FundSnapshot
from dcae.errors import DCAEError, ValidationError
from dcae.paper.ledger import PaperPortfolioLedger
from dcae.proposals.service import ProposalService


class DCAEServiceApp:
    """Framework-agnostic HTTP request dispatcher shared by stdlib HTTP and FastAPI."""

    def __init__(
        self,
        config: EngineConfig,
        audit_store: SQLiteAuditStore,
        paper_ledger: Optional[PaperPortfolioLedger] = None,
    ) -> None:
        self.config = config
        self.audit_store = audit_store
        self.proposal_service = ProposalService(config, audit_store)
        self.paper_ledger = paper_ledger

    def handle(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        parsed = urlparse(path)
        route = parsed.path.rstrip("/") or "/"
        payload = body or {}

        try:
            if method == "GET" and route == "/health":
                chain = self.audit_store.verify_chain()
                return 200, {
                    "status": "ok",
                    "environment": self.config.environment,
                    "config_version": self.config.config_version,
                    "live_execution_enabled": self.config.live_execution_enabled,
                    "paper_execution_enabled": self.config.paper_execution_enabled,
                    "audit_chain": chain,
                }

            if method == "GET" and route == "/v1/metrics":
                return 200, self.proposal_service.metrics.snapshot()

            if method == "GET" and route == "/v1/audit":
                chain = self.audit_store.verify_chain()
                events = [ev.to_dict() for ev in self.audit_store.list_events()]
                return 200, {"chain_integrity": chain, "events": events}

            if method == "POST" and route == "/v1/snapshots":
                snapshot = FundSnapshot.from_dict(payload["snapshot"])
                actor = str(payload.get("actor_id", "api_client"))
                manifest = self.proposal_service.register_snapshot(snapshot, actor_id=actor)
                return 201, {"manifest": manifest}

            if method == "POST" and route == "/v1/allocations/propose":
                if "snapshot" in payload:
                    snapshot = FundSnapshot.from_dict(payload["snapshot"])
                elif "snapshot_id" in payload:
                    loaded = self.audit_store.load_snapshot(str(payload["snapshot_id"]))
                    if loaded is None:
                        raise ValidationError(
                            f"Unknown snapshot_id: {payload['snapshot_id']}"
                        )
                    snapshot = loaded
                else:
                    raise ValidationError("Request must include 'snapshot' or 'snapshot_id'")

                proposal = self.proposal_service.create_proposal(
                    snapshot,
                    proposer_id=str(payload.get("proposer_id", "api_proposer")),
                    allocator_name=str(payload.get("allocator_name", "constrained_mv")),
                    idempotency_key=payload.get("idempotency_key"),
                    now_utc=payload.get("now_utc"),
                )
                return 201, proposal.to_dict()

            if method == "GET" and route == "/v1/proposals":
                return 200, {"proposals": self.audit_store.list_proposals()}

            if route.startswith("/v1/proposals/"):
                parts = route.split("/")
                # /v1/proposals/{id}
                if len(parts) == 4 and method == "GET":
                    prop = self.audit_store.load_proposal(parts[3])
                    if prop is None:
                        return 404, {"error_code": "NOT_FOUND", "message": "Proposal not found"}
                    return 200, prop
                # /v1/proposals/{id}/review
                if len(parts) == 5 and parts[4] == "review" and method == "POST":
                    updated = self.proposal_service.review_proposal(
                        parts[3],
                        reviewer_id=str(payload.get("reviewer_id", "")),
                        approve=bool(payload.get("approve", False)),
                        notes=str(payload.get("notes", "")),
                        reviewed_at_utc=payload.get("reviewed_at_utc"),
                    )
                    return 200, updated
                # /v1/proposals/{id}/execute-paper
                if len(parts) == 5 and parts[4] == "execute-paper" and method == "POST":
                    if self.paper_ledger is None:
                        raise ValidationError("Paper ledger is not attached to this service instance")
                    receipt = self.paper_ledger.apply_approved_proposal(
                        proposal_service=self.proposal_service,
                        proposal_id=parts[3],
                        executed_by=str(payload.get("executed_by", "")),
                        idempotency_key=payload.get("idempotency_key"),
                        executed_at_utc=payload.get("executed_at_utc"),
                        mode=str(payload.get("mode", "paper")),
                    )
                    return 200, receipt.to_dict()

            return 404, {"error_code": "NOT_FOUND", "message": f"Route not found: {method} {route}"}

        except DCAEError as err:
            return err.http_status, err.to_dict()
        except KeyError as exc:
            return 422, {
                "error_code": "MISSING_FIELD",
                "message": f"Missing required field: {exc}",
            }
        except ValueError as exc:
            return 422, {"error_code": "VALUE_ERROR", "message": str(exc)}


    def handle_json(
        self,
        method: str,
        path: str,
        raw_body: Optional[bytes] = None,
    ) -> Tuple[int, bytes]:
        """Raw JSON-over-HTTP dispatcher usable both by ThreadingHTTPServer and in-process tests."""
        body: Optional[Dict[str, Any]] = None
        if raw_body:
            try:
                body = json.loads(raw_body.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                err = json.dumps(
                    {"error_code": "INVALID_JSON", "message": str(exc)},
                    sort_keys=True,
                ).encode("utf-8")
                return 400, err
        status, resp = self.handle(method, path, body)
        return status, json.dumps(resp, sort_keys=True).encode("utf-8")


def make_http_server(
    app: DCAEServiceApp,
    host: str = "127.0.0.1",
    port: int = 0,
) -> ThreadingHTTPServer:
    """Create a stdlib ThreadingHTTPServer bound to host:port serving DCAEServiceApp."""

    class Handler(BaseHTTPRequestHandler):
        def _dispatch(self, method: str) -> None:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw_bytes = self.rfile.read(length) if length > 0 else None
            status, encoded = app.handle_json(method, self.path, raw_bytes)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch("POST")

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    return ThreadingHTTPServer((host, port), Handler)
