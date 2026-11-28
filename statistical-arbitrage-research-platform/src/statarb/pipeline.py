"""End-to-end research pipeline orchestrator, structured logging, and versioned report output."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import scipy
import statsmodels

from statarb._version import __version__
from statarb.backtest import BacktestResult, run_walk_forward_backtest
from statarb.config import PlatformConfig, load_config
from statarb.contracts import ValidationReport, load_and_validate_inputs
from statarb.evaluation import (
    compute_deflated_sharpe_ratio,
    compute_oos_uncertainty,
    compute_performance_summary,
    compute_stability_report,
    run_benchmark_comparisons,
)


def _make_logger() -> logging.Logger:
    logger = logging.getLogger("statarb.pipeline")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def _log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    payload = {
        "ts_utc": datetime.now(UTC).isoformat(),
        "event": event,
        **fields,
    }
    logger.info(json.dumps(payload, sort_keys=True, default=str))


def capture_runtime_metadata() -> dict[str, str]:
    return {
        "platform_version": __version__,
        "python_version": sys.version.split()[0],
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "numpy_version": np.__version__,
        "polars_version": pl.__version__,
        "statsmodels_version": statsmodels.__version__,
        "scipy_version": scipy.__version__,
    }


def compute_run_id(cfg: PlatformConfig, validation: ValidationReport) -> str:
    """Deterministic run identifier derived from config hash, input hashes, and seed."""
    token = (
        f"{cfg.run_name}|{cfg.config_hash()}|{validation.prices_hash}|"
        f"{validation.borrow_hash}|{cfg.seed}"
    )
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]
    return f"{cfg.run_name}_{digest}"


def _run_config_trial_sweep(
    prices: pl.DataFrame,
    borrow: pl.DataFrame,
    base_cfg: PlatformConfig,
) -> list[dict[str, Any]]:
    """Evaluate all attempted z_entry configurations and record full trial ledger."""
    trials: list[dict[str, Any]] = []
    seen_z = sorted(set(list(base_cfg.evaluation.sweep_z_entries) + [base_cfg.signals.z_entry]))
    for z_val in seen_z:
        trial_dict = base_cfg.to_dict()
        trial_dict["signals"]["z_entry"] = float(z_val)
        trial_cfg = PlatformConfig.model_validate(trial_dict)
        res = run_walk_forward_backtest(prices, borrow, trial_cfg)
        perf = compute_performance_summary(res, trial_cfg)
        trials.append(
            {
                "config_hash": trial_cfg.config_hash(),
                "z_entry": float(z_val),
                "is_primary": abs(float(z_val) - base_cfg.signals.z_entry) < 1e-12,
                "sharpe_net": round(perf.sharpe_net, 4),
                "total_return_net": round(perf.total_return_net, 6),
                "max_drawdown_net": round(perf.max_drawdown_net, 6),
                "annualized_turnover": round(perf.annualized_turnover, 4),
                "n_trades": perf.n_trades,
            }
        )
    return trials


def execute_research_pipeline(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    run_trial_sweep: bool = True,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run full validated research pipeline and optionally persist raw reports."""
    logger = _make_logger()
    if quiet:
        logger.setLevel(logging.WARNING)

    cfg_path = Path(config_path).resolve()
    root = Path(project_root).resolve() if project_root is not None else cfg_path.parent.parent
    cfg = load_config(cfg_path)

    prices_path = Path(cfg.data.prices_path)
    if not prices_path.is_absolute():
        prices_path = (root / prices_path).resolve()
    borrow_path = Path(cfg.data.borrow_path)
    if not borrow_path.is_absolute():
        borrow_path = (root / borrow_path).resolve()

    _log_event(
        logger,
        "pipeline_start",
        run_name=cfg.run_name,
        config_hash=cfg.config_hash(),
        prices_path=str(prices_path),
        borrow_path=str(borrow_path),
    )

    prices, borrow, val_report = load_and_validate_inputs(
        prices_path,
        borrow_path,
        max_missing_fraction=cfg.data.max_missing_fraction,
        missing_price_policy=cfg.data.missing_price_policy,
    )
    run_id = compute_run_id(cfg, val_report)

    result: BacktestResult = run_walk_forward_backtest(prices, borrow, cfg)
    perf = compute_performance_summary(result, cfg)

    oos_returns = np.asarray([r.daily_return_net for r in result.daily_equity], dtype=np.float64)
    uncertainty = compute_oos_uncertainty(oos_returns, cfg.evaluation, seed=cfg.seed)
    stability = compute_stability_report(result, prices)
    benchmarks = run_benchmark_comparisons(prices, borrow, cfg, result)

    # Multiple-testing accounting: both candidate-level per fold and config-trial level (DSR)
    trial_ledger = (
        _run_config_trial_sweep(prices, borrow, cfg)
        if run_trial_sweep
        else [
            {
                "config_hash": cfg.config_hash(),
                "z_entry": cfg.signals.z_entry,
                "is_primary": True,
                "sharpe_net": round(perf.sharpe_net, 4),
                "total_return_net": round(perf.total_return_net, 6),
                "max_drawdown_net": round(perf.max_drawdown_net, 6),
                "annualized_turnover": round(perf.annualized_turnover, 4),
                "n_trades": perf.n_trades,
            }
        ]
    )
    trial_sharpes = [t["sharpe_net"] for t in trial_ledger]
    trial_var = float(np.var(trial_sharpes, ddof=1)) if len(trial_sharpes) > 1 else None
    dsr = compute_deflated_sharpe_ratio(
        oos_returns,
        n_trials=len(trial_ledger),
        ann_days=float(cfg.evaluation.annualization_days),
        rf_annual=cfg.evaluation.risk_free_rate_annual,
        trial_sharpe_variance=trial_var,
    )

    total_pairs_tested = sum(f.selection.n_pairs_tested for f in result.folds)
    total_baskets_tested = sum(f.selection.n_baskets_tested for f in result.folds)
    total_raw_passed = sum(f.selection.n_passed_raw_pvalue for f in result.folds)
    total_adj_passed = sum(f.selection.n_passed_adjusted_pvalue for f in result.folds)

    fold_summaries = []
    for f in result.folds:
        fold_summaries.append(
            {
                "fold": f.fold.to_dict(),
                "eligible_symbols": list(f.universe.eligible_symbols),
                "excluded_symbols": {k: list(v) for k, v in f.universe.excluded_reasons.items()},
                "n_pairs_tested": f.selection.n_pairs_tested,
                "n_baskets_tested": f.selection.n_baskets_tested,
                "n_passed_raw_pvalue": f.selection.n_passed_raw_pvalue,
                "n_passed_adjusted_pvalue": f.selection.n_passed_adjusted_pvalue,
                "selected_spreads": [m.to_dict() for m in f.selection.selected_models],
                "oos_net_return": round(f.oos_net_return, 6),
                "oos_gross_return": round(f.oos_gross_return, 6),
                "n_trades": f.n_trades,
            }
        )

    report: dict[str, Any] = {
        "schema_version": cfg.schema_version,
        "run_id": run_id,
        "run_name": cfg.run_name,
        "config_hash": cfg.config_hash(),
        "data_provenance": "synthetic_fixture",
        "runtime_environment": capture_runtime_metadata(),
        "input_validation": {
            "contract_version": val_report.contract_version,
            "prices_rows": val_report.prices_rows,
            "borrow_rows": val_report.borrow_rows,
            "symbols": list(val_report.symbols),
            "dropped_symbols": list(val_report.dropped_symbols),
            "date_start": val_report.date_start,
            "date_end": val_report.date_end,
            "corporate_action_events": val_report.corporate_action_events,
            "prices_hash": val_report.prices_hash,
            "borrow_hash": val_report.borrow_hash,
        },
        "config": cfg.to_dict(),
        "performance": perf.to_dict(),
        "uncertainty": uncertainty,
        "multiple_testing": {
            "method": cfg.selection.multiple_testing_method,
            "fdr_alpha": cfg.selection.fdr_alpha,
            "total_pair_hypotheses_tested": total_pairs_tested,
            "total_basket_hypotheses_tested": total_baskets_tested,
            "total_passed_raw_pvalue": total_raw_passed,
            "total_passed_adjusted_pvalue": total_adj_passed,
            "deflated_sharpe": dsr,
            "attempted_configurations": trial_ledger,
        },
        "stability": stability,
        "benchmarks": benchmarks,
        "reconciliation": result.reconciliation.to_dict(),
        "diagnostics": result.diagnostics,
        "folds": fold_summaries,
    }

    if output_dir is not None:
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        summary_path = out_root / f"{run_id}_summary.json"
        summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        # Also write latest pointer for deterministic runbook consumption
        latest_path = out_root / f"{cfg.run_name}_latest.json"
        latest_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        # Persist equity curve & trade ledger CSVs
        eq_df = pl.DataFrame([r.to_dict() for r in result.daily_equity])
        eq_df.write_csv(out_root / f"{run_id}_equity_curve.csv")
        if result.trades:
            tr_df = pl.DataFrame([t.to_dict() for t in result.trades])
            tr_df.write_csv(out_root / f"{run_id}_trades.csv")

        _log_event(
            logger,
            "pipeline_complete",
            run_id=run_id,
            summary_path=str(summary_path),
            sharpe_net=perf.sharpe_net,
            reconciliation_passed=result.reconciliation.passed,
        )

    return report
