"""Shared pytest fixtures for the Multi-Strategy PMS test suite."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pms.api import create_app
from pms.domain.master import MasterRegistry
from pms.ledger.engine import LedgerEngine
from pms.ledger.store import InMemoryEventStore, SQLiteEventStore
from pms.service import PortfolioService
from pms.valuation.pricing import MarketSnapshot

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture()
def master() -> MasterRegistry:
    return MasterRegistry.from_file(PROJECT_ROOT / "config" / "accounts_and_instruments.json")


@pytest.fixture()
def memory_ledger(master: MasterRegistry) -> LedgerEngine:
    return LedgerEngine(master, InMemoryEventStore())


@pytest.fixture()
def sqlite_store(tmp_path: Path) -> SQLiteEventStore:
    return SQLiteEventStore(tmp_path / "test_pms.sqlite3")


@pytest.fixture()
def baseline_market() -> MarketSnapshot:
    return MarketSnapshot.from_file(PROJECT_ROOT / "data" / "fixtures" / "market_marks.json")


@pytest.fixture()
def service_with_fixtures(tmp_path: Path) -> PortfolioService:
    return PortfolioService.from_workspace(
        PROJECT_ROOT,
        sqlite_path=tmp_path / "service_pms.sqlite3",
        load_fixtures=True,
    )


@pytest.fixture()
def api_client(service_with_fixtures: PortfolioService) -> TestClient:
    app = create_app(service_with_fixtures)
    return TestClient(app)
