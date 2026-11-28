"""End-to-end robustness evaluation orchestrator with structured logging and replay."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .bootstrap import run_block_bootstrap
from .costs import CostAdjustedSeries, apply_transaction_costs
from .integrity import audit_trial_integrity
from .io_adapters import (
    align_return_series_matrix,
    load_parameter_grid,
    load_policy_config,
    load_return_series_csv,
)
from .models import (
    ContractValidationError,
    InsufficientSampleError,
    ParameterGridSpec,
    ReturnSeriesSpec,
    TrialRecord,
    TrialStatus,
    canonical_json,
)
from .pbo import compute_pbo_cscv
from .perturbation import evaluate_parameter_perturbation
from .promotion import evaluate_promotion
from .registry import ImmutableExperimentRegistry
from .reporting import build_report_payload, compute_run_id, write_report_artifacts
from .statistics import compute_dsr, compute_sharpe_ratio, diagnose_outliers
from .walk_forward import run_walk_forward

logger = logging.getLogger("robustness_lab")


def _log_event(event: str, **fields: Any) -> None:
    logger.info(json.dumps({"event": event, **fields}, sort_keys=True))


def evaluate_experiment(
    registry: ImmutableExperimentRegistry,
    series_by_id: dict[str, ReturnSeriesSpec],
    policy: dict[str, Any],
    candidate_trial_id: Optional[str] = None,
    parameter_grid: Optional[ParameterGridSpec] = None,
    output_dir: Optional[Path] = None,
    report_basename: Optional[str] = None,
) -> dict[str, Any]:
    """Run the full robustness suite against a logged experiment registry.

    Returns the machine-readable report payload dictionary.
    """
    seed = int(policy["seed"])
    registry_valid, registry_errors = registry.verify()
    all_records = registry.load_entries() if registry_valid else []
    records: list[TrialRecord] = [e.record for e in all_records]

    # If registry failed verify() due to hash mismatch (but JSON parseable), load records for audit visibility
    if not registry_valid and not records and registry.ledger_path.exists():
        try:
            records = [e.record for e in registry.load_entries()]
        except Exception:
            records = []

    head_hash = registry.head_hash() if registry_valid else "INVALID_REGISTRY"
    completed_records = [r for r in records if r.status == TrialStatus.COMPLETED]
    failed_records = [r for r in records if r.status != TrialStatus.COMPLETED]
    experiment_id = records[0].experiment_id if records else "unknown_experiment"

    _log_event(
        "evaluation_started",
        experiment_id=experiment_id,
        total_trials=len(records),
        completed_trials=len(completed_records),
        failed_trials=len(failed_records),
        registry_valid=registry_valid,
    )

    # 1. Integrity and cherry-picking audit
    int_cfg = policy["integrity"]
    integrity = audit_trial_integrity(
        registry_valid=registry_valid,
        registry_errors=registry_errors,
        all_records=records,
        series_by_id=series_by_id,
        parameter_grid=parameter_grid,
        duplicate_correlation_threshold=float(int_cfg["duplicate_correlation_threshold"]),
        max_allowed_duplicate_fraction=float(int_cfg["max_allowed_duplicate_fraction"]),
        min_grid_coverage_fraction=float(int_cfg["min_grid_coverage_fraction"]),
    )

    # Identify completed trials whose return series are available
    available_completed = [r for r in completed_records if r.series_id in series_by_id]
    if not available_completed:
        # Cannot compute return-based diagnostics; produce incomplete-evidence REJECT report
        cand_id = candidate_trial_id or "none"
        run_id = compute_run_id(experiment_id, cand_id, head_hash, policy, seed)
        verdict = evaluate_promotion(
            candidate_trial_id=cand_id,
            policy=policy,
            integrity=integrity,
            dsr=None,
            pbo=None,
            bootstrap=None,
            walk_forward=None,
            perturbation=None,
            outliers=None,
        )
        counters = {
            "n_trials_logged": len(records),
            "n_trials_completed": len(completed_records),
            "n_trials_failed_or_rejected": len(failed_records),
            "n_diagnostics_completed": 1,
            "n_diagnostics_missing": 6,
            "n_warnings": len(verdict.review_notes),
        }
        payload = build_report_payload(
            run_id=run_id,
            experiment_id=experiment_id,
            candidate_trial_id=cand_id,
            registry_head_hash=head_hash,
            policy=policy,
            seed=seed,
            provenance="unknown",
            cost_summary={},
            integrity=integrity,
            dsr=None,
            pbo=None,
            bootstrap=None,
            walk_forward=None,
            perturbation=None,
            outliers=None,
            verdict=verdict,
            counters=counters,
        )
        if output_dir is not None:
            write_report_artifacts(payload, output_dir, basename=report_basename)
        return payload

    # Align timestamps and apply each trial's own CostModel
    ordered_series_ids = [r.series_id for r in available_completed if r.series_id is not None]
    timestamps, aligned_specs = align_return_series_matrix(series_by_id, ordered_series_ids)
    ann_factor = aligned_specs[0].frequency.periods_per_year
    provenance = aligned_specs[0].provenance

    cost_adjusted: dict[str, CostAdjustedSeries] = {}
    trial_net_sharpes: dict[str, float] = {}
    net_cols: list[np.ndarray] = []
    gross_cols: list[np.ndarray] = []
    trial_ids: list[str] = []

    for rec, spec in zip(available_completed, aligned_specs):
        adj = apply_transaction_costs(spec, rec.cost_model)
        cost_adjusted[rec.trial_id] = adj
        _, net_sr = compute_sharpe_ratio(adj.net_returns, annualization_factor=ann_factor)
        trial_net_sharpes[rec.trial_id] = net_sr
        net_cols.append(adj.net_returns)
        gross_cols.append(adj.gross_returns)
        trial_ids.append(rec.trial_id)

    net_mat = np.column_stack(net_cols)
    gross_mat = np.column_stack(gross_cols)
    benchmark_arr = aligned_specs[0].benchmark_array()

    # Select candidate trial (either explicitly requested or highest net Sharpe)
    if candidate_trial_id is not None:
        matched = [r for r in available_completed if r.trial_id == candidate_trial_id]
        if not matched:
            raise ContractValidationError(
                f"Requested candidate_trial_id '{candidate_trial_id}' not found among completed trials"
            )
        cand_record = matched[0]
    else:
        best_tid = max(trial_ids, key=lambda tid: trial_net_sharpes[tid])
        cand_record = next(r for r in available_completed if r.trial_id == best_tid)

    cand_adj = cost_adjusted[cand_record.trial_id]
    run_id = compute_run_id(experiment_id, cand_record.trial_id, head_hash, policy, seed)

    # 2. DSR diagnostics
    dsr_cfg = policy["dsr"]
    min_obs = int(policy.get("min_observations", 60))
    dsr_res = None
    try:
        dsr_res = compute_dsr(
            candidate_trial_id=cand_record.trial_id,
            candidate_returns=cand_adj.net_returns,
            all_trial_returns=net_mat,
            annualization_factor=ann_factor,
            benchmark_sr_annual=float(dsr_cfg["benchmark_sharpe"]),
            min_observations=min_obs,
            use_autocorr_adjustment=bool(dsr_cfg["use_autocorrelation_adjustment"]),
            max_autocorr_lags=int(dsr_cfg["max_autocorr_lags"]),
            extra_unobserved_failed_trials=len(failed_records),
        )
    except InsufficientSampleError as exc:
        _log_event("diagnostic_skipped_insufficient_sample", diagnostic="dsr", reason=str(exc))

    # 3. CSCV PBO diagnostics
    pbo_cfg = policy["pbo"]
    pbo_res = None
    try:
        pbo_res = compute_pbo_cscv(
            trial_returns_matrix=net_mat,
            n_partitions=int(pbo_cfg["n_partitions"]),
            annualization_factor=ann_factor,
            max_combinations=int(pbo_cfg["max_combinations"]),
            seed=seed,
        )
    except InsufficientSampleError as exc:
        _log_event("diagnostic_skipped_insufficient_sample", diagnostic="pbo", reason=str(exc))

    # 4. Block bootstrap
    boot_cfg = policy["bootstrap"]
    boot_res = None
    try:
        boot_res = run_block_bootstrap(
            returns=cand_adj.net_returns,
            n_resamples=int(boot_cfg["n_resamples"]),
            block_length=int(boot_cfg["block_length"]),
            method=str(boot_cfg["method"]),
            confidence_level=float(boot_cfg["confidence_level"]),
            annualization_factor=ann_factor,
            seed=seed,
            min_observations=min_obs,
        )
    except InsufficientSampleError as exc:
        _log_event("diagnostic_skipped_insufficient_sample", diagnostic="bootstrap", reason=str(exc))

    # 5. Walk-forward OOS
    wf_cfg = policy["walk_forward"]
    wf_res = None
    try:
        wf_res = run_walk_forward(
            timestamps=timestamps,
            trial_ids=trial_ids,
            net_returns_matrix=net_mat,
            gross_returns_matrix=gross_mat,
            benchmark_returns=benchmark_arr,
            mode=str(wf_cfg["mode"]),
            train_window=int(wf_cfg["train_window"]),
            test_window=int(wf_cfg["test_window"]),
            step_size=int(wf_cfg["step_size"]),
            purge_gap=int(wf_cfg["purge_gap"]),
            annualization_factor=ann_factor,
        )
    except InsufficientSampleError as exc:
        _log_event("diagnostic_skipped_insufficient_sample", diagnostic="walk_forward", reason=str(exc))

    # 6. Parameter perturbation
    pert_cfg = policy["perturbation"]
    pert_res = evaluate_parameter_perturbation(
        center_trial=cand_record,
        completed_trials=available_completed,
        trial_net_sharpes=trial_net_sharpes,
        relative_step=float(pert_cfg["relative_step"]),
        parameter_grid=parameter_grid,
    )

    # 7. Outlier diagnostics
    outlier_res = None
    try:
        outlier_res = diagnose_outliers(
            returns=cand_adj.net_returns,
            annualization_factor=ann_factor,
            zscore_threshold=float(int_cfg["max_outlier_zscore_warning"]),
            kurtosis_threshold=float(int_cfg["max_excess_kurtosis_warning"]),
        )
    except InsufficientSampleError as exc:
        _log_event("diagnostic_skipped_insufficient_sample", diagnostic="outliers", reason=str(exc))

    # 8. Promotion verdict
    verdict = evaluate_promotion(
        candidate_trial_id=cand_record.trial_id,
        policy=policy,
        integrity=integrity,
        dsr=dsr_res,
        pbo=pbo_res,
        bootstrap=boot_res,
        walk_forward=wf_res,
        perturbation=pert_res,
        outliers=outlier_res,
    )

    diag_list = [integrity, dsr_res, pbo_res, boot_res, wf_res, pert_res, outlier_res]
    n_present = sum(1 for d in diag_list if d is not None)
    counters = {
        "n_trials_logged": len(records),
        "n_trials_completed": len(completed_records),
        "n_trials_failed_or_rejected": len(failed_records),
        "n_observations": int(net_mat.shape[0]),
        "n_diagnostics_completed": n_present,
        "n_diagnostics_missing": len(diag_list) - n_present,
        "n_walk_forward_folds": wf_res.n_folds if wf_res is not None else 0,
        "n_pbo_splits": pbo_res.n_combinations_evaluated if pbo_res is not None else 0,
        "n_bootstrap_resamples": boot_res.n_resamples if boot_res is not None else 0,
        "n_warnings": len(verdict.review_notes),
    }

    cost_summary = {
        "cost_model": cand_record.cost_model.to_dict(),
        "total_gross_compounded": cand_adj.total_gross_compounded,
        "total_net_compounded": cand_adj.total_net_compounded,
        "mean_period_cost_bps": cand_adj.mean_period_cost_bps,
        "annualized_cost_drag": cand_adj.annualized_cost_drag,
    }

    payload = build_report_payload(
        run_id=run_id,
        experiment_id=experiment_id,
        candidate_trial_id=cand_record.trial_id,
        registry_head_hash=head_hash,
        policy=policy,
        seed=seed,
        provenance=provenance,
        cost_summary=cost_summary,
        integrity=integrity,
        dsr=dsr_res,
        pbo=pbo_res,
        bootstrap=boot_res,
        walk_forward=wf_res,
        perturbation=pert_res,
        outliers=outlier_res,
        verdict=verdict,
        counters=counters,
    )

    if output_dir is not None:
        json_path, md_path = write_report_artifacts(payload, output_dir, basename=report_basename)
        _log_event("report_written", run_id=run_id, json_path=str(json_path), md_path=str(md_path))

    _log_event(
        "evaluation_completed",
        run_id=run_id,
        candidate_trial_id=cand_record.trial_id,
        decision=verdict.decision.value,
        evidence_complete=verdict.evidence_complete,
    )
    return payload


def evaluate_from_directory(
    experiment_dir: Path,
    policy_path: Path,
    candidate_trial_id: Optional[str] = None,
    output_dir: Optional[Path] = None,
    report_basename: Optional[str] = None,
) -> dict[str, Any]:
    """Load a versioned experiment directory (registry.jsonl, series/*.csv, grid.json) and evaluate."""
    exp_dir = Path(experiment_dir)
    if not exp_dir.exists():
        raise ContractValidationError(f"Experiment directory does not exist: {exp_dir}")

    policy = load_policy_config(policy_path)
    registry = ImmutableExperimentRegistry(exp_dir / "registry.jsonl")

    grid_path = exp_dir / "parameter_grid.json"
    grid = load_parameter_grid(grid_path) if grid_path.exists() else None

    manifest_path = exp_dir / "manifest.json"
    frequency = "daily"
    provenance = "synthetic_fixture"
    dataset_version = "v1"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frequency = str(manifest.get("frequency", "daily"))
        provenance = str(manifest.get("provenance", "synthetic_fixture"))
        dataset_version = str(manifest.get("dataset_version", "v1"))

    series_dir = exp_dir / "series"
    series_by_id: dict[str, ReturnSeriesSpec] = {}
    if series_dir.exists():
        for csv_path in sorted(series_dir.glob("*.csv")):
            sid = csv_path.stem
            series_by_id[sid] = load_return_series_csv(
                csv_path,
                series_id=sid,
                version=dataset_version,
                frequency=frequency,
                provenance=provenance,
            )

    return evaluate_experiment(
        registry=registry,
        series_by_id=series_by_id,
        policy=policy,
        candidate_trial_id=candidate_trial_id,
        parameter_grid=grid,
        output_dir=output_dir,
        report_basename=report_basename,
    )


def compare_reports_logical_equality(
    report_a: dict[str, Any],
    report_b: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Compare two report payloads ignoring volatile wall-clock timestamp fields and tuple/list reprs."""
    diffs: list[str] = []
    for key in ("run_id", "experiment_id", "candidate_trial_id", "registry_head_hash", "verdict", "diagnostics", "counters"):
        if canonical_json(report_a.get(key)) != canonical_json(report_b.get(key)):
            diffs.append(f"Mismatch in top-level section '{key}'")
    return len(diffs) == 0, diffs
