"""Integration tests for SQLite immutability triggers, hash chain, and replay recovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from pms.domain.errors import ImmutabilityViolationError
from pms.ledger.store import SQLiteEventStore
from pms.service import PortfolioService


def test_sqlite_triggers_prevent_history_mutation(
    service_with_fixtures: PortfolioService,
) -> None:
    assert isinstance(service_with_fixtures.store, SQLiteEventStore)
    valid, tip_hash = service_with_fixtures.store.verify_hash_chain()
    assert valid is True
    assert len(tip_hash) == 64

    with pytest.raises(ImmutabilityViolationError):
        service_with_fixtures.store.attempt_illegal_mutation("EVT-0001")


def test_cache_wipe_and_deterministic_replay_recovery(
    project_root: Path, tmp_path: Path
) -> None:
    db_file = tmp_path / "recovery_test.sqlite3"
    svc1 = PortfolioService.from_workspace(project_root, sqlite_path=db_file, load_fixtures=True)
    snap_before, _ = svc1.replay()

    recovery = svc1.verify_recovery_via_replay()
    assert recovery["recovered_identically"] is True
    assert recovery["cache_cleared"] is True
    assert recovery["hash_chain_valid"] is True
    assert recovery["before_state_hash"] == recovery["after_replay_1_state_hash"]
    assert recovery["after_replay_1_state_hash"] == recovery["after_replay_2_state_hash"]

    # Re-open SQLite from cold process without re-loading fixtures and replay again
    svc2 = PortfolioService.from_workspace(project_root, sqlite_path=db_file, load_fixtures=False)
    snap_cold, _ = svc2.replay()
    assert snap_cold.state_hash == snap_before.state_hash
    assert snap_cold.accounts == snap_before.accounts
