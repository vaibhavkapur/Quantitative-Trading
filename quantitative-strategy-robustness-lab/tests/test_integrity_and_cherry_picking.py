"""Integration tests for duplicate trial detection and cherry-picking audits."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from robustness_lab.integrity import audit_trial_integrity
from robustness_lab.io_adapters import load_parameter_grid, load_return_series_csv
from robustness_lab.models import CostModel, ReturnFrequency, ReturnSeriesSpec, TrialRecord, TrialStatus
from robustness_lab.registry import ImmutableExperimentRegistry


def test_cherry_picked_fixture_flagged_for_gaps_duplicates_and_grid_coverage(fixtures_dir: Path) -> None:
    exp = fixtures_dir / "valid" / "cherry_picked_experiment"
    reg = ImmutableExperimentRegistry(exp / "registry.jsonl")
    valid, errs = reg.verify()
    assert valid is True  # hash chain itself is intact, but content reveals cherry-picking

    records = reg.list_records()
    grid = load_parameter_grid(exp / "parameter_grid.json")
    series_by_id = {
        p.stem: load_return_series_csv(p, p.stem, "v1")
        for p in sorted((exp / "series").glob("*.csv"))
    }

    audit = audit_trial_integrity(
        registry_valid=valid,
        registry_errors=errs,
        all_records=records,
        series_by_id=series_by_id,
        parameter_grid=grid,
        duplicate_correlation_threshold=0.995,
        max_allowed_duplicate_fraction=0.15,
        min_grid_coverage_fraction=0.85,
    )
    assert audit.passed is False
    assert len(audit.sequence_gaps) > 0
    assert len(audit.exact_duplicate_pairs) >= 1
    assert audit.grid_coverage_fraction is not None and audit.grid_coverage_fraction < 0.85
    assert any("cherry-picking" in f.lower() for f in audit.cherry_picking_flags)


def test_near_duplicate_correlation_and_hash_mismatch_detected() -> None:
    rng = np.random.default_rng(801)
    n_obs = 120
    ts = tuple(f"2024-01-{i+1:03d}" for i in range(n_obs))
    base = rng.normal(0.0005, 0.01, size=n_obs)
    near_copy = base + rng.normal(0.0, 1e-4, size=n_obs)  # corr > 0.999

    s1 = ReturnSeriesSpec("s1", "v1", ReturnFrequency.DAILY, ts, tuple(float(x) for x in base))
    s2 = ReturnSeriesSpec("s2", "v1", ReturnFrequency.DAILY, ts, tuple(float(x) for x in near_copy))

    r1 = TrialRecord(
        trial_id="t1",
        sequence_number=1,
        experiment_id="e",
        strategy_name="s",
        parameters={"k": 1},
        status=TrialStatus.COMPLETED,
        created_at_utc="2026-01-01T00:01:00Z",
        dataset_version="v1",
        code_version="0.1",
        cost_model=CostModel(),
        series_id="s1",
        return_series_sha256="deadbeef" * 8,  # Deliberate hash mismatch
    )
    r2 = TrialRecord(
        trial_id="t2",
        sequence_number=2,
        experiment_id="e",
        strategy_name="s",
        parameters={"k": 2},
        status=TrialStatus.COMPLETED,
        created_at_utc="2026-01-01T00:02:00Z",
        dataset_version="v1",
        code_version="0.1",
        cost_model=CostModel(),
        series_id="s2",
        return_series_sha256=s2.content_sha256,
    )

    audit = audit_trial_integrity(
        registry_valid=True,
        registry_errors=[],
        all_records=[r1, r2],
        series_by_id={"s1": s1, "s2": s2},
        duplicate_correlation_threshold=0.995,
        max_allowed_duplicate_fraction=0.15,
    )
    assert audit.passed is False
    assert len(audit.near_duplicate_pairs) == 1
    assert any("hash mismatch" in f.lower() for f in audit.cherry_picking_flags)


def test_clean_experiment_with_logged_failed_trial_passes_integrity(fixtures_dir: Path) -> None:
    exp = fixtures_dir / "valid" / "known_signal_experiment"
    reg = ImmutableExperimentRegistry(exp / "registry.jsonl")
    valid, errs = reg.verify()
    grid = load_parameter_grid(exp / "parameter_grid.json")
    series_by_id = {
        p.stem: load_return_series_csv(p, p.stem, "v1")
        for p in sorted((exp / "series").glob("*.csv"))
    }
    audit = audit_trial_integrity(
        registry_valid=valid,
        registry_errors=errs,
        all_records=reg.list_records(),
        series_by_id=series_by_id,
        parameter_grid=grid,
    )
    assert audit.passed is True
    assert audit.failed_or_rejected_trials == 1
    assert audit.grid_coverage_fraction == 1.0
    assert audit.sequence_gaps == ()
