"""Optional FastAPI application adapter wrapping DCAEServiceApp."""

from __future__ import annotations

from typing import Any, Dict

from dcae.api.http_server import DCAEServiceApp
from dcae.errors import DCAEError


def create_fastapi_app(service: DCAEServiceApp) -> Any:
    """Build a FastAPI app if optional dependency `fastapi` is installed."""
    try:
        from fastapi import FastAPI, Request  # type: ignore[import-not-found]
        from fastapi.responses import JSONResponse  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DCAEError(
            "Optional dependency 'fastapi' is not installed. "
            "Install with `pip install -e .[api]` or use dcae.api.http_server.make_http_server.",
            code="OPTIONAL_DEPENDENCY_MISSING",
            http_status=500,
        ) from exc

    api = FastAPI(title="Dynamic Capital Allocation Engine", version="0.1.0")

    @api.get("/health")
    def health() -> JSONResponse:
        code, data = service.handle("GET", "/health")
        return JSONResponse(status_code=code, content=data)

    @api.get("/v1/metrics")
    def metrics() -> JSONResponse:
        code, data = service.handle("GET", "/v1/metrics")
        return JSONResponse(status_code=code, content=data)

    @api.get("/v1/audit")
    def audit() -> JSONResponse:
        code, data = service.handle("GET", "/v1/audit")
        return JSONResponse(status_code=code, content=data)

    @api.post("/v1/snapshots")
    async def register_snapshot(request: Request) -> JSONResponse:
        body: Dict[str, Any] = await request.json()
        code, data = service.handle("POST", "/v1/snapshots", body)
        return JSONResponse(status_code=code, content=data)

    @api.post("/v1/allocations/propose")
    async def propose(request: Request) -> JSONResponse:
        body: Dict[str, Any] = await request.json()
        code, data = service.handle("POST", "/v1/allocations/propose", body)
        return JSONResponse(status_code=code, content=data)

    @api.get("/v1/proposals")
    def list_proposals() -> JSONResponse:
        code, data = service.handle("GET", "/v1/proposals")
        return JSONResponse(status_code=code, content=data)

    @api.get("/v1/proposals/{proposal_id}")
    def get_proposal(proposal_id: str) -> JSONResponse:
        code, data = service.handle("GET", f"/v1/proposals/{proposal_id}")
        return JSONResponse(status_code=code, content=data)

    @api.post("/v1/proposals/{proposal_id}/review")
    async def review_proposal(proposal_id: str, request: Request) -> JSONResponse:
        body: Dict[str, Any] = await request.json()
        code, data = service.handle("POST", f"/v1/proposals/{proposal_id}/review", body)
        return JSONResponse(status_code=code, content=data)

    @api.post("/v1/proposals/{proposal_id}/execute-paper")
    async def execute_paper(proposal_id: str, request: Request) -> JSONResponse:
        body: Dict[str, Any] = await request.json()
        code, data = service.handle("POST", f"/v1/proposals/{proposal_id}/execute-paper", body)
        return JSONResponse(status_code=code, content=data)

    return api
