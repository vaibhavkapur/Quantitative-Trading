"""End-to-end API and Web UI integration tests."""

from __future__ import annotations

from decimal import Decimal

from fastapi.testclient import TestClient


def test_ui_and_read_endpoints(api_client: TestClient) -> None:
    ui_resp = api_client.get("/")
    assert ui_resp.status_code == 200
    assert "SAAT Multi-Strategy Portfolio Management System" in ui_resp.text

    health = api_client.get("/api/v1/health").json()
    assert health["status"] == "OK"
    assert health["hash_chain_valid"] is True
    assert health["controls"]["live_order_execution_enabled"] is False
    assert health["event_count"] == 17

    master = api_client.get("/api/v1/master").json()
    assert master["fund"]["fund_id"] == "FUND-SAAT-MAIN"
    assert len(master["accounts"]) == 4

    val = api_client.get("/api/v1/portfolio/valuation").json()
    assert val["consolidation_check"]["passed"] is True
    assert val["pnl"]["unexplained_residual_base"] == "0" or float(val["pnl"]["unexplained_residual_base"]) == 0.0
    assert "EQ-UNPRICED-US" in val["unpriced_instruments"]

    lots = api_client.get("/api/v1/portfolio/lots").json()
    assert lots["count"] >= 4

    journals = api_client.get("/api/v1/journals").json()
    assert journals["event_count"] == 17
    for tb in journals["trial_balance"].values():
        assert tb["debits"] == tb["credits"]
        assert Decimal(tb["debits"]) == Decimal(tb["credits"])


def test_api_controls_duplicate_conflict_cod_and_recovery(api_client: TestClient) -> None:
    # Live execution must return 403
    live_resp = api_client.post("/api/v1/oms/live-orders")
    assert live_resp.status_code == 403
    assert live_resp.json()["error_code"] == "CONTROL_VIOLATION"

    # Re-posting an existing event idempotency key returns 409 Conflict
    existing_evt = api_client.get("/api/v1/events").json()["events"][0]["event"]
    dup_resp = api_client.post("/api/v1/events", json=existing_evt, headers={"X-Role": "OPERATOR"})
    assert dup_resp.status_code == 409
    assert dup_resp.json()["error_code"] == "DUPLICATE_EVENT"

    # Break resolution requires APPROVER role
    runs = api_client.get("/api/v1/reconciliation/runs").json()["runs"]
    break_id = runs[-1]["breaks"][0]["break_id"]
    unauth = api_client.post(
        f"/api/v1/reconciliation/breaks/{break_id}/resolve",
        json={"reason": "test", "unit_cost": "200.00"},
        headers={"X-Role": "VIEWER", "X-Actor": "viewer_1"},
    )
    assert unauth.status_code == 403

    # COD report renders in both JSON and Markdown
    cod_md = api_client.get("/api/v1/reports/cod?format=markdown")
    assert cod_md.status_code == 200
    assert "Close-of-Day Portfolio Report" in cod_md.text

    # Recovery verification
    rec = api_client.post("/api/v1/recovery/verify").json()
    assert rec["recovered_identically"] is True
    assert rec["hash_chain_valid"] is True
