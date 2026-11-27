"""FastAPI REST API and web UI server for the Multi-Strategy PMS."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Header, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from pms.adapters.broker_feed import BrokerStatement
from pms.adapters.paper_oms import PaperFillReport
from pms.domain.enums import Role
from pms.domain.errors import (
    AuthorizationError,
    ControlViolationError,
    DuplicateEventError,
    ImmutabilityViolationError,
    MasterDataError,
    PMSError,
    ValidationError,
)
from pms.reporting.cod import render_cod_markdown
from pms.service import PortfolioService
from pms.valuation.pricing import MarketSnapshot


class ResolveBreakRequest(BaseModel):
    reason: str = Field(min_length=3)
    unit_cost: Decimal = Decimal("0")


def _parse_role(x_role: str | None) -> Role:
    raw = (x_role or Role.VIEWER.value).strip().upper()
    try:
        return Role(raw)
    except ValueError as exc:
        raise AuthorizationError(f"Unknown role header X-Role={x_role!r}") from exc


def create_app(service: PortfolioService) -> FastAPI:
    app = FastAPI(
        title="SAAT Multi-Strategy Portfolio Management System",
        version="0.1.0",
        description=(
            "Authoritative event-sourced multi-strategy portfolio accounting, "
            "lot tracking, mark-to-market valuation, consolidation, and broker reconciliation."
        ),
    )
    app.state.pms = service

    @app.exception_handler(PMSError)
    async def _handle_pms_error(_: Request, exc: PMSError) -> JSONResponse:
        status_map: dict[type[PMSError], int] = {
            DuplicateEventError: 409,
            AuthorizationError: 403,
            ControlViolationError: 403,
            ImmutabilityViolationError: 409,
            MasterDataError: 422,
            ValidationError: 400,
        }
        status_code = status_map.get(type(exc), 400)
        return JSONResponse(status_code=status_code, content=exc.to_dict())

    @app.get("/", response_class=HTMLResponse)
    def serve_ui() -> HTMLResponse:
        html_path = Path(__file__).parent / "web" / "index.html"
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

    @app.get("/api/v1/health")
    def get_health() -> dict[str, Any]:
        return service.health()

    @app.get("/api/v1/master")
    def get_master() -> dict[str, Any]:
        return {
            "schema_version": service.master.schema_version,
            "effective_at": service.master.effective_at,
            "fund": service.master.fund.model_dump(mode="json"),
            "accounts": [a.model_dump(mode="json") for a in service.master.list_accounts()],
            "instruments": [i.model_dump(mode="json") for i in service.master.list_instruments()],
        }

    @app.get("/api/v1/events")
    def list_events(as_of: Annotated[datetime | None, Query()] = None) -> dict[str, Any]:
        events = service.store.list_events(as_of=as_of)
        return {
            "count": len(events),
            "events": [e.model_dump(mode="json") for e in events],
        }

    @app.post("/api/v1/events", status_code=201)
    def post_event(
        payload: dict[str, Any],
        x_role: Annotated[str | None, Header()] = "OPERATOR",
    ) -> dict[str, Any]:
        stored = service.ingest_event(payload, role=_parse_role(x_role))
        return stored.model_dump(mode="json")

    @app.get("/api/v1/journals")
    def list_journals(as_of: Annotated[datetime | None, Query()] = None) -> dict[str, Any]:
        snap, journals = service.replay(as_of=as_of, include_journals=True)
        return {
            "event_count": snap.event_count,
            "state_hash": snap.state_hash,
            "trial_balance": {
                ccy: {k: str(v) for k, v in vals.items()}
                for ccy, vals in snap.trial_balance.items()
            },
            "journals": [j.model_dump(mode="json") for j in journals],
        }

    @app.get("/api/v1/portfolio/valuation")
    def get_valuation(
        snapshot_id: Annotated[str | None, Query()] = None,
        as_of: Annotated[datetime | None, Query()] = None,
    ) -> dict[str, Any]:
        report = service.get_valuation(snapshot_id=snapshot_id, as_of=as_of)
        return report.model_dump(mode="json")

    @app.get("/api/v1/portfolio/lots")
    def get_lots(
        account_id: Annotated[str | None, Query()] = None,
        instrument_id: Annotated[str | None, Query()] = None,
    ) -> dict[str, Any]:
        snap, _ = service.replay()
        lots: list[dict[str, Any]] = []
        for aid, acct_state in snap.accounts.items():
            if account_id and aid != account_id:
                continue
            for iid, pos in acct_state.positions.items():
                if instrument_id and iid != instrument_id:
                    continue
                for lot in pos.lots:
                    lots.append(lot.model_dump(mode="json"))
        return {"count": len(lots), "lots": lots}

    @app.post("/api/v1/markets/snapshots", status_code=201)
    def post_market_snapshot(
        snapshot: MarketSnapshot,
        x_role: Annotated[str | None, Header()] = "OPERATOR",
    ) -> dict[str, Any]:
        saved = service.register_market_snapshot(snapshot, role=_parse_role(x_role))
        return saved.model_dump(mode="json")

    @app.post("/api/v1/oms/paper-fills")
    def post_paper_fill(
        report: PaperFillReport,
        x_role: Annotated[str | None, Header()] = "OPERATOR",
    ) -> dict[str, Any]:
        res = service.ingest_paper_fill(report, role=_parse_role(x_role))
        return res.model_dump(mode="json")

    @app.post("/api/v1/oms/live-orders")
    def post_live_order() -> dict[str, Any]:
        service.paper_oms.submit_live_order()
        return {"status": "UNREACHABLE"}

    @app.get("/api/v1/reconciliation/runs")
    def list_reconciliation_runs() -> dict[str, Any]:
        runs = service.reconciliation.list_runs()
        return {
            "count": len(runs),
            "runs": [r.model_dump(mode="json") for r in runs],
        }

    @app.post("/api/v1/reconciliation/run")
    def post_reconciliation_run(
        statement: BrokerStatement | None = None,
        x_role: Annotated[str | None, Header()] = "OPERATOR",
    ) -> dict[str, Any]:
        run = service.run_reconciliation(statement, role=_parse_role(x_role))
        return run.model_dump(mode="json")

    @app.post("/api/v1/reconciliation/breaks/{break_id}/resolve")
    def resolve_break(
        break_id: str,
        body: ResolveBreakRequest,
        x_actor: Annotated[str, Header()] = "ops_user",
        x_role: Annotated[str | None, Header()] = "VIEWER",
    ) -> dict[str, Any]:
        resolved, stored = service.resolve_break(
            break_id=break_id,
            actor=x_actor,
            role=_parse_role(x_role),
            reason=body.reason,
            unit_cost=body.unit_cost,
        )
        return {
            "break": resolved.model_dump(mode="json"),
            "adjustment_event": stored.model_dump(mode="json"),
        }

    @app.get("/api/v1/reports/cod")
    def get_cod_report(
        format: Annotated[str, Query(pattern="^(json|markdown)$")] = "json",
        snapshot_id: Annotated[str | None, Query()] = None,
    ) -> Any:
        pkg = service.build_cod_report(snapshot_id=snapshot_id)
        if format == "markdown":
            return PlainTextResponse(content=render_cod_markdown(pkg), media_type="text/markdown")
        return pkg.model_dump(mode="json")

    @app.post("/api/v1/recovery/verify")
    def post_recovery_verify() -> dict[str, Any]:
        return service.verify_recovery_via_replay()

    return app
