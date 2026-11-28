"""Versioned machine-readable reporting, runtime metadata capture, and replay verification."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import polars as pl
import scipy

from cross_asset_factor_engine import SCHEMA_VERSION, __version__
from cross_asset_factor_engine.evaluation import EvaluationOutput
from cross_asset_factor_engine.io import write_csv, write_json


def make_run_id(output: EvaluationOutput) -> str:
    """Stable deterministic identifier derived from config hash, dataset hash, and schema version."""
    return f"{output.config.config_name}_{output.config.config_hash()[:8]}_{output.dataset_hash[:8]}"


def capture_runtime_metadata() -> dict[str, Any]:
    """Capture runtime environment versions for reproducibility audits."""
    return {
        "engine_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "polars_version": pl.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }


def save_evaluation_report(
    output: EvaluationOutput,
    output_root: str | Path,
    *,
    deterministic_timestamp: bool = False,
) -> Path:
    """Persist versioned machine-readable JSON/CSV reports and diagnostics into reports/runs/<run_id>/."""
    run_id = make_run_id(output)
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    generated_at = (
        "deterministic-replay"
        if deterministic_timestamp
        else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    summary_payload: dict[str, Any] = {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "data_provenance": output.dataset_provenance,
        "dataset_hash": output.dataset_hash,
        "config_hash": output.config.config_hash(),
        "runtime_environment": capture_runtime_metadata(),
        "config": output.config.to_dict(),
        "folds": output.folds,
        "strategy_metrics": output.strategy_metrics,
        "class_attribution": output.class_attribution,
        "constraint_audit": output.constraint_audit,
        "provenance_audit": output.provenance_audit,
        "return_correlations": output.return_correlations,
        "diagnostics_counters": output.diagnostics.counters,
    }
    write_json(run_dir / "summary.json", summary_payload)

    # Tabular strategy comparison CSV
    strat_rows = [
        {"strategy": strat, **metrics}
        for strat, metrics in output.strategy_metrics.items()
    ]
    write_csv(run_dir / "strategy_comparison.csv", pl.DataFrame(strat_rows))

    # Tabular asset-class attribution CSV
    cls_rows = [
        {"strategy": strat, "asset_class": cls_name, **vals}
        for strat, cls_map in output.class_attribution.items()
        for cls_name, vals in cls_map.items()
    ]
    write_csv(run_dir / "class_attribution.csv", pl.DataFrame(cls_rows))

    # Daily returns CSV
    write_csv(run_dir / "daily_returns.csv", output.daily_returns_df)

    # Detailed weights CSV
    if output.config.reporting.write_detailed_weights:
        write_csv(run_dir / "portfolio_weights.csv", output.weights_df)

    # Signal provenance audit CSV
    if output.config.reporting.write_signal_audit:
        audit_df = output.oos_signals.select(
            [
                "signal_date",
                "as_of_ts",
                "source_max_publication_ts",
                "asset_id",
                "asset_class",
                "factor_name",
                "raw_value",
                "z_score",
                "signal_status",
                "missing_reason",
                "provenance_hash",
            ]
        )
        write_csv(run_dir / "signal_provenance_audit.csv", audit_df)

    # Structured diagnostics log
    (run_dir / "diagnostics.jsonl").write_text(
        output.diagnostics.to_jsonl() + "\n", encoding="utf-8"
    )
    return run_dir


def verify_replay_equivalence(
    first: EvaluationOutput,
    second: EvaluationOutput,
    *,
    atol: float = 1.0e-12,
) -> dict[str, Any]:
    """Verify two runs with identical inputs and config produce logically matching outputs."""
    id_a = make_run_id(first)
    id_b = make_run_id(second)
    if id_a != id_b:
        return {"equivalent": False, "reason": f"run_id mismatch: {id_a} != {id_b}"}

    if first.strategy_metrics != second.strategy_metrics:
        return {"equivalent": False, "reason": "strategy_metrics differ"}

    if first.class_attribution != second.class_attribution:
        return {"equivalent": False, "reason": "class_attribution differs"}

    ret_a = first.daily_returns_df.sort(["strategy", "date"])["net_return"].to_numpy()
    ret_b = second.daily_returns_df.sort(["strategy", "date"])["net_return"].to_numpy()
    max_ret_diff = float(np.max(np.abs(ret_a - ret_b))) if len(ret_a) > 0 else 0.0
    if max_ret_diff > atol:
        return {
            "equivalent": False,
            "reason": f"daily net_return max diff {max_ret_diff} exceeds atol {atol}",
        }

    w_a = first.weights_df.sort(["strategy", "decision_date", "asset_id"])["weight"].to_numpy()
    w_b = second.weights_df.sort(["strategy", "decision_date", "asset_id"])["weight"].to_numpy()
    max_w_diff = float(np.max(np.abs(w_a - w_b))) if len(w_a) > 0 else 0.0
    if max_w_diff > atol:
        return {
            "equivalent": False,
            "reason": f"weights max diff {max_w_diff} exceeds atol {atol}",
        }

    return {
        "equivalent": True,
        "run_id": id_a,
        "max_return_diff": max_ret_diff,
        "max_weight_diff": max_w_diff,
    }
