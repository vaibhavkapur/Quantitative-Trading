"""Trial integrity, duplicate detection, and cherry-picking audit module.

Detects:
  - Tampered or non-monotonic immutable experiment registry
  - Sequence number gaps (unreported or deleted intermediate trials)
  - Return series hash mismatches vs logged registry entries
  - Exact duplicate return series or duplicate parameter configurations
  - Near-duplicate return series (pairwise correlation >= threshold)
  - Incomplete parameter grid coverage (selective reporting / cherry-picking)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .models import (
    IntegrityAuditResult,
    ParameterGridSpec,
    ReturnSeriesSpec,
    TrialRecord,
    TrialStatus,
    canonical_json,
)


def audit_trial_integrity(
    registry_valid: bool,
    registry_errors: list[str],
    all_records: list[TrialRecord],
    series_by_id: dict[str, ReturnSeriesSpec],
    parameter_grid: Optional[ParameterGridSpec] = None,
    duplicate_correlation_threshold: float = 0.995,
    max_allowed_duplicate_fraction: float = 0.15,
    min_grid_coverage_fraction: float = 0.85,
) -> IntegrityAuditResult:
    """Audit trial records and return series for duplicates, tampering, and cherry-picking."""
    flags: list[str] = []

    if not registry_valid:
        flags.append(f"Registry cryptographic/structural verification failed: {'; '.join(registry_errors)}")

    total_logged = len(all_records)
    completed = [r for r in all_records if r.status == TrialStatus.COMPLETED]
    failed_or_rejected = [r for r in all_records if r.status != TrialStatus.COMPLETED]

    if total_logged == 0:
        flags.append("Experiment registry contains zero trials.")

    # 1. Check sequence number continuity (1..max_seq) to detect deleted trials
    seq_nums = sorted(r.sequence_number for r in all_records)
    sequence_gaps: list[int] = []
    if seq_nums:
        expected_range = set(range(1, seq_nums[-1] + 1))
        sequence_gaps = sorted(expected_range.difference(seq_nums))
        if sequence_gaps:
            flags.append(
                f"Detected {len(sequence_gaps)} missing sequence number(s) in ledger "
                f"(possible omitted/cherry-picked trials): {sequence_gaps[:10]}"
            )

    # 2. Verify return series SHA-256 against logged record hashes
    for rec in completed:
        if rec.series_id not in series_by_id:
            flags.append(f"Completed trial '{rec.trial_id}' references missing series_id '{rec.series_id}'.")
            continue
        spec = series_by_id[rec.series_id]
        if rec.return_series_sha256 and rec.return_series_sha256 != spec.content_sha256:
            flags.append(
                f"Return series hash mismatch for trial '{rec.trial_id}': "
                f"logged={rec.return_series_sha256[:12]}... vs actual={spec.content_sha256[:12]}..."
            )

    # 3. Exact duplicate detection (same return hash OR same parameter config among completed trials)
    exact_pairs: list[tuple[str, str]] = []
    near_pairs: list[tuple[str, str, float]] = []
    flagged_duplicate_ids: set[str] = set()

    present_completed = [r for r in completed if r.series_id in series_by_id]
    n_present = len(present_completed)

    # Pre-extract return arrays for pairwise checks
    arrays: list[np.ndarray] = [series_by_id[r.series_id].gross_array() for r in present_completed]
    hashes: list[str] = [series_by_id[r.series_id].content_sha256 for r in present_completed]
    param_keys: list[str] = [canonical_json(r.parameters) for r in present_completed]

    for i in range(n_present):
        for j in range(i + 1, n_present):
            tid_i = present_completed[i].trial_id
            tid_j = present_completed[j].trial_id
            if hashes[i] == hashes[j] or (param_keys[i] == param_keys[j] and present_completed[i].parameters):
                exact_pairs.append((tid_i, tid_j))
                flagged_duplicate_ids.add(tid_j)
                continue
            arr_i, arr_j = arrays[i], arrays[j]
            if arr_i.size == arr_j.size and arr_i.size >= 4:
                std_i = float(np.std(arr_i, ddof=1))
                std_j = float(np.std(arr_j, ddof=1))
                if std_i > 1e-15 and std_j > 1e-15:
                    corr = float(np.corrcoef(arr_i, arr_j)[0, 1])
                    if corr >= duplicate_correlation_threshold:
                        near_pairs.append((tid_i, tid_j, corr))
                        flagged_duplicate_ids.add(tid_j)

    dup_fraction = (len(flagged_duplicate_ids) / float(n_present)) if n_present > 0 else 0.0
    if exact_pairs:
        flags.append(f"Detected {len(exact_pairs)} exact duplicate trial pair(s).")
    if near_pairs:
        flags.append(
            f"Detected {len(near_pairs)} near-duplicate trial pair(s) with correlation >= {duplicate_correlation_threshold:.3f}."
        )
    if dup_fraction > max_allowed_duplicate_fraction:
        flags.append(
            f"Duplicate trial fraction ({dup_fraction:.2%}) exceeds allowed limit ({max_allowed_duplicate_fraction:.2%})."
        )

    # 4. Parameter grid completeness check (counts both completed AND logged failures)
    expected_grid_trials: Optional[int] = None
    covered_grid_trials: Optional[int] = None
    grid_coverage_fraction: Optional[float] = None

    if parameter_grid is not None:
        expected_configs = parameter_grid.config_keys()
        expected_grid_trials = len(expected_configs)
        logged_configs = {canonical_json(r.parameters) for r in all_records}
        covered = expected_configs.intersection(logged_configs)
        covered_grid_trials = len(covered)
        grid_coverage_fraction = covered_grid_trials / float(expected_grid_trials) if expected_grid_trials > 0 else 1.0
        if grid_coverage_fraction < min_grid_coverage_fraction:
            missing_cnt = expected_grid_trials - covered_grid_trials
            flags.append(
                f"Parameter grid coverage is {grid_coverage_fraction:.1%} "
                f"({covered_grid_trials}/{expected_grid_trials}; {missing_cnt} unlogged configuration(s)), "
                f"below required {min_grid_coverage_fraction:.1%} — potential cherry-picking."
            )

    passed = (
        registry_valid
        and total_logged > 0
        and len(sequence_gaps) == 0
        and dup_fraction <= max_allowed_duplicate_fraction
        and (grid_coverage_fraction is None or grid_coverage_fraction >= min_grid_coverage_fraction)
        and not any("hash mismatch" in f or "missing series_id" in f for f in flags)
    )

    return IntegrityAuditResult(
        registry_valid=registry_valid,
        registry_errors=tuple(registry_errors),
        total_logged_trials=total_logged,
        completed_trials=len(completed),
        failed_or_rejected_trials=len(failed_or_rejected),
        exact_duplicate_pairs=tuple(exact_pairs),
        near_duplicate_pairs=tuple(near_pairs),
        duplicate_fraction=dup_fraction,
        sequence_gaps=tuple(sequence_gaps),
        expected_grid_trials=expected_grid_trials,
        covered_grid_trials=covered_grid_trials,
        grid_coverage_fraction=grid_coverage_fraction,
        cherry_picking_flags=tuple(flags),
        passed=passed,
    )
