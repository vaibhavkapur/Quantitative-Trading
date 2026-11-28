"""Immutable experiment registry tests: failure tracking, hash chaining, tamper detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from robustness_lab.models import (
    ContractValidationError,
    CostModel,
    RegistryIntegrityError,
    TrialRecord,
    TrialStatus,
)
from robustness_lab.registry import GENESIS_HASH, ImmutableExperimentRegistry


def _make_record(seq: int, status: TrialStatus = TrialStatus.COMPLETED) -> TrialRecord:
    return TrialRecord(
        trial_id=f"trial_{seq}",
        sequence_number=seq,
        experiment_id="exp-reg-test",
        strategy_name="demo",
        parameters={"k": seq},
        status=status,
        created_at_utc=f"2026-01-01T00:{seq:02d}:00Z",
        dataset_version="v1",
        code_version="0.1.0",
        cost_model=CostModel(),
        series_id=f"trial_{seq}" if status == TrialStatus.COMPLETED else None,
        failure_reason="Solver diverged" if status != TrialStatus.COMPLETED else None,
    )


def test_registry_tracks_all_statuses_and_verifies_chain(tmp_path: Path) -> None:
    reg = ImmutableExperimentRegistry(tmp_path / "registry.jsonl")
    e1 = reg.append(_make_record(1, TrialStatus.COMPLETED))
    e2 = reg.append(_make_record(2, TrialStatus.FAILED))
    e3 = reg.append(_make_record(3, TrialStatus.REJECTED))
    e4 = reg.append(_make_record(4, TrialStatus.ABORTED))

    assert e1.prev_hash == GENESIS_HASH
    assert e2.prev_hash == e1.entry_hash
    assert e3.prev_hash == e2.entry_hash
    assert e4.prev_hash == e3.entry_hash

    valid, errs = reg.verify()
    assert valid is True
    assert errs == []
    assert len(reg.list_records()) == 4
    assert len(reg.list_records(status=TrialStatus.FAILED)) == 1


def test_failed_trial_requires_failure_reason() -> None:
    with pytest.raises(ContractValidationError, match="failure_reason"):
        TrialRecord(
            trial_id="bad_fail",
            sequence_number=1,
            experiment_id="exp",
            strategy_name="s",
            parameters={"k": 1},
            status=TrialStatus.FAILED,
            created_at_utc="2026-01-01T00:00:00Z",
            dataset_version="v1",
            code_version="0.1.0",
            cost_model=CostModel(),
        )


def test_registry_detects_tampering_and_blocks_appends(tmp_path: Path) -> None:
    ledger = tmp_path / "registry.jsonl"
    reg = ImmutableExperimentRegistry(ledger)
    reg.append(_make_record(1, TrialStatus.COMPLETED))
    reg.append(_make_record(2, TrialStatus.FAILED))
    reg.append(_make_record(3, TrialStatus.COMPLETED))

    # Mutate entry 1 retroactively
    text = ledger.read_text(encoding="utf-8").replace('"k":1', '"k":99')
    ledger.write_text(text, encoding="utf-8")

    valid, errs = reg.verify()
    assert valid is False
    assert any("tamper detected" in e for e in errs)

    # Appending to a tampered registry must raise RegistryIntegrityError
    with pytest.raises(RegistryIntegrityError):
        reg.append(_make_record(4, TrialStatus.COMPLETED))


def test_registry_detects_deleted_middle_entry_and_duplicate_ids(tmp_path: Path) -> None:
    ledger = tmp_path / "registry.jsonl"
    reg = ImmutableExperimentRegistry(ledger)
    reg.append(_make_record(1, TrialStatus.COMPLETED))
    reg.append(_make_record(2, TrialStatus.FAILED))
    reg.append(_make_record(3, TrialStatus.COMPLETED))

    # Reject duplicate trial_id or non-monotonic sequence_number on append
    with pytest.raises(ContractValidationError, match="already exists"):
        reg.append(_make_record(1, TrialStatus.COMPLETED))
    with pytest.raises(ContractValidationError, match="exceed last logged sequence"):
        rec_low = TrialRecord(
            trial_id="trial_low",
            sequence_number=2,
            experiment_id="exp-reg-test",
            strategy_name="demo",
            parameters={"k": 22},
            status=TrialStatus.COMPLETED,
            created_at_utc="2026-01-01T00:05:00Z",
            dataset_version="v1",
            code_version="0.1.0",
            cost_model=CostModel(),
            series_id="trial_low",
        )
        reg.append(rec_low)

    # Delete middle failed trial to simulate cherry-picking cover-up
    lines = ledger.read_text(encoding="utf-8").splitlines()
    ledger.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")
    valid, errs = reg.verify()
    assert valid is False
    assert any("prev_hash mismatch" in e for e in errs)


def test_invalid_tampered_fixture_fails_verification(fixtures_dir: Path) -> None:
    reg = ImmutableExperimentRegistry(fixtures_dir / "invalid" / "tampered_registry.jsonl")
    valid, errs = reg.verify()
    assert valid is False
    assert len(errs) >= 1
