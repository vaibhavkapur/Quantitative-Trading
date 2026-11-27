"""Optional FastAPI ASGI application wrapping AttributionService.

Requires optional extras: `pip install .[api]`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pae.api.service import AttributionService
from pae.config import AttributionConfig
from pae.domain.models import AttributionInputBatch
from pae.errors import PAEError
from pae.security.rbac import Actor


def create_fastapi_app(service: Optional[AttributionService] = None) -> Any:
    """Build a FastAPI application delegating to AttributionService."""
    try:
        from fastapi import FastAPI, Header, HTTPException
        from fastapi.responses import HTMLResponse, PlainTextResponse
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI is not installed. Install optional dependencies via `pip install .[api]` "
            "or use the built-in stdlib server `pae.api.http_server`."
        ) from exc

    svc = service or AttributionService(config=AttributionConfig.from_file())
    app = FastAPI(
        title="Performance Attribution Engine API",
        version=svc.config.methodology_version,
        description="Reconciled return, contribution, Brinson, factor, and TCA attribution API.",
    )

    def _actor(x_actor_id: str, x_actor_role: str) -> Actor:
        try:
            return Actor.from_headers(x_actor_id, x_actor_role)
        except PAEError as exc:
            raise HTTPException(status_code=exc.http_status, detail=exc.to_dict()) from exc

    @app.get("/api/v1/health")
    def health() -> Dict[str, Any]:
        return svc.health()

    @app.get("/api/v1/reports")
    def list_reports(
        portfolio_id: Optional[str] = None,
        report_id: Optional[str] = None,
        x_actor_id: str = Header(default="viewer"),
        x_actor_role: str = Header(default="VIEWER"),
    ) -> Dict[str, Any]:
        try:
            return {
                "reports": svc.list_reports(
                    actor=_actor(x_actor_id, x_actor_role),
                    portfolio_id=portfolio_id,
                    report_id=report_id,
                )
            }
        except PAEError as exc:
            raise HTTPException(status_code=exc.http_status, detail=exc.to_dict()) from exc

    @app.post("/api/v1/attribution/run", status_code=201)
    def run_attribution(
        payload: Dict[str, Any],
        x_actor_id: str = Header(default="analyst"),
        x_actor_role: str = Header(default="ANALYST"),
    ) -> Dict[str, Any]:
        try:
            batch = AttributionInputBatch.from_dict(payload.get("batch", payload))
            return svc.run_attribution(
                batch,
                actor=_actor(x_actor_id, x_actor_role),
                report_id=payload.get("report_id"),
                correction_reason=payload.get("correction_reason"),
            )
        except PAEError as exc:
            raise HTTPException(status_code=exc.http_status, detail=exc.to_dict()) from exc

    @app.get("/api/v1/reports/{version_id}")
    def get_report(
        version_id: str,
        x_actor_id: str = Header(default="viewer"),
        x_actor_role: str = Header(default="VIEWER"),
    ) -> Dict[str, Any]:
        try:
            return svc.get_report(version_id, actor=_actor(x_actor_id, x_actor_role))
        except PAEError as exc:
            raise HTTPException(status_code=exc.http_status, detail=exc.to_dict()) from exc

    @app.post("/api/v1/reports/{version_id}/review")
    def review_report(
        version_id: str,
        payload: Optional[Dict[str, Any]] = None,
        x_actor_id: str = Header(default="reviewer"),
        x_actor_role: str = Header(default="REVIEWER"),
    ) -> Dict[str, Any]:
        try:
            return svc.review_report(
                version_id,
                actor=_actor(x_actor_id, x_actor_role),
                comment=(payload or {}).get("comment", ""),
            )
        except PAEError as exc:
            raise HTTPException(status_code=exc.http_status, detail=exc.to_dict()) from exc

    @app.post("/api/v1/reports/{version_id}/approve")
    def approve_report(
        version_id: str,
        payload: Optional[Dict[str, Any]] = None,
        x_actor_id: str = Header(default="approver"),
        x_actor_role: str = Header(default="APPROVER"),
    ) -> Dict[str, Any]:
        try:
            return svc.approve_report(
                version_id,
                actor=_actor(x_actor_id, x_actor_role),
                comment=(payload or {}).get("comment", ""),
            )
        except PAEError as exc:
            raise HTTPException(status_code=exc.http_status, detail=exc.to_dict()) from exc

    @app.get("/api/v1/reports/{version_id}/export")
    def export_report(
        version_id: str,
        format: str = "md",
        x_actor_id: str = Header(default="viewer"),
        x_actor_role: str = Header(default="VIEWER"),
    ) -> Any:
        try:
            content = svc.export_report(version_id, format, actor=_actor(x_actor_id, x_actor_role))
            if format.lower() == "html":
                return HTMLResponse(content)
            return PlainTextResponse(content)
        except PAEError as exc:
            raise HTTPException(status_code=exc.http_status, detail=exc.to_dict()) from exc

    return app
