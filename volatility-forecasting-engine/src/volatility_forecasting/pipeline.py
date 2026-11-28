"""End-to-end reproducible volatility forecasting evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import polars as pl
import scipy
import sklearn

from volatility_forecasting.config import EngineConfig, ModelSpec, load_config, validate_config
from volatility_forecasting.contracts import (
    CONTRACT_VERSION,
    compute_file_sha256,
    read_table,
    validate_nonnegative_forecasts,
)
from volatility_forecasting.evaluation import (
    apply_forecast_intervals,
    assign_regimes,
    compute_regime_diagnostics,
    fit_conformal_multipliers,
    fit_regime_thresholds,
    interval_calibration_metrics,
    mae_loss,
    mincer_zarnowitz_regression,
    qlike_loss,
    rmse_loss,
    variance_to_vol,
)
from volatility_forecasting.features import FeatureBuildDiagnostics, build_causal_dataset
from volatility_forecasting.models import BaseVolatilityModel, create_model
from volatility_forecasting.splits import ChronologicalSplit, generate_chronological_splits

logger = logging.getLogger("volatility_forecasting")


@dataclass
class RunResult:
    """Typed output bundle returned by `run_pipeline`."""

    run_id: str
    config_fingerprint: str
    input_fingerprints: dict[str, str]
    horizon_days: int
    splits: list[ChronologicalSplit]
    feature_diagnostics: FeatureBuildDiagnostics
    model_comparison: pl.DataFrame
    forecast_intervals: pl.DataFrame
    regime_diagnostics: pl.DataFrame
    attempted_configurations: list[dict[str, Any]]
    counters: dict[str, int]
    manifest: dict[str, Any]
    output_paths: dict[str, str] = field(default_factory=dict)


def _resolve_path(path_str: str, base_dir: Path | None) -> Path:
    p = Path(path_str)
    if p.is_absolute() or base_dir is None:
        return p
    return (base_dir / p).resolve()


def _build_run_id(config_fp: str, input_fps: dict[str, str], run_name: str) -> str:
    payload = json.dumps(
        {"run_name": run_name, "config_fp": config_fp, "inputs": input_fps},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    safe_name = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in run_name)
    return f"{safe_name}-{digest}"


def _runtime_environment() -> dict[str, str]:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "polars_version": pl.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "contract_version": CONTRACT_VERSION,
    }


def evaluate_models_chronologically(
    dataset: pl.DataFrame,
    cfg: EngineConfig,
    horizon_days: int | None = None,
    *,
    custom_models: list[BaseVolatilityModel] | None = None,
) -> tuple[
    list[ChronologicalSplit],
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    list[dict[str, Any]],
    dict[str, int],
]:
    """Run chronological split evaluation with conformal intervals and regime diagnostics.

    Isolated from filesystem I/O so tests and downstream libraries can invoke it directly.
    """
    validate_config(cfg)
    h = int(horizon_days if horizon_days is not None else cfg.features.primary_horizon)
    target_col = f"target_rv_h{h}"
    ann = cfg.annualization_factor
    v_floor = cfg.evaluation.variance_floor
    levels = cfg.evaluation.interval_levels

    splits = generate_chronological_splits(dataset, cfg.splits, horizon_days=h)

    model_specs: list[ModelSpec] = list(cfg.models)
    attempted_configs: list[dict[str, Any]] = []
    counters: dict[str, int] = {
        "n_splits": len(splits),
        "n_models": len(custom_models) if custom_models is not None else len(model_specs),
        "n_fit_calls": 0,
        "n_oos_predictions": 0,
        "n_leakage_checks_passed": len(splits),
        "n_nonnegativity_checks_passed": 0,
    }

    # Accumulators per model across all OOS splits
    pred_rows: list[dict[str, Any]] = []
    model_eval_store: dict[str, dict[str, list[Any]]] = {}

    model_items: list[tuple[str, str, dict[str, Any], BaseVolatilityModel | None]]
    if custom_models is not None:
        model_items = [
            (m.name, m.__class__.__name__, dict(m.params), m) for m in custom_models
        ]
    else:
        model_items = [(s.name, s.type, dict(s.params), None) for s in model_specs]

    for m_name, m_type, m_params, _ in model_items:
        attempted_configs.append({"name": m_name, "type": m_type, "params": m_params})
        model_eval_store[m_name] = {
            "y_true_var": [],
            "y_pred_var": [],
            "regimes": [],
            "intervals_vol": {lvl: {"lower": [], "upper": []} for lvl in levels},
            "split_qlikes": [],
            "split_maes_vol": [],
        }

    # Identify baseline model name (first naive model, else first model)
    baseline_name = next(
        (name for name, mtype, _, _ in model_items if mtype == "naive"),
        model_items[0][0],
    )

    for split in splits:
        train_df = dataset[list(split.train_indices)]
        inner_train_df = dataset[list(split.inner_train_indices)]
        cal_df = dataset[list(split.cal_indices)]
        test_df = dataset[list(split.test_indices)]

        # Context for strictly causal recursive filters (EWMA/GARCH): all rows strictly before test_start
        first_test_ts = test_df["timestamp"].min()
        context_for_test = dataset.filter(pl.col("timestamp") < first_test_ts)
        first_cal_ts = cal_df["timestamp"].min()
        context_for_cal = dataset.filter(pl.col("timestamp") < first_cal_ts)

        # Regime thresholds fit strictly on training window realized variance
        regime_thresholds = fit_regime_thresholds(
            train_df["rv_1d"].to_numpy(), cfg.evaluation.regime_quantiles
        )
        test_regimes = assign_regimes(test_df["rv_1d"].to_numpy(), regime_thresholds)

        y_cal_var = cal_df[target_col].to_numpy().astype(np.float64)
        y_test_var = test_df[target_col].to_numpy().astype(np.float64)
        y_test_vol = variance_to_vol(y_test_var, ann)

        test_assets = test_df["asset_id"].to_list()
        test_ts = [str(ts) for ts in test_df["timestamp"].to_list()]

        for m_name, m_type, m_params, custom_inst in model_items:
            spec = ModelSpec(name=m_name, type=m_type, params=m_params)

            # Step 1: Fit on inner_train -> predict calibration tail to fit conformal intervals
            if custom_inst is None:
                cal_model = create_model(spec, seed=cfg.seed)
                cal_model.fit(inner_train_df, horizon=h)
                cal_pred_var = cal_model.predict_variance(
                    cal_df,
                    horizon=h,
                    context_df=context_for_cal,
                    strict_nonnegative=cfg.evaluation.strict_nonnegative,
                )
            else:
                custom_inst.fit(inner_train_df, horizon=h)
                cal_pred_var = custom_inst.predict_variance(
                    cal_df,
                    horizon=h,
                    context_df=context_for_cal,
                    strict_nonnegative=cfg.evaluation.strict_nonnegative,
                )
            counters["n_fit_calls"] += 1

            conf_table = fit_conformal_multipliers(
                y_cal_var, cal_pred_var, levels=levels, variance_floor=v_floor
            )

            # Step 2: Refit on full training window -> predict OOS test split
            model = custom_inst if custom_inst is not None else create_model(spec, seed=cfg.seed)
            model.fit(train_df, horizon=h)
            counters["n_fit_calls"] += 1

            y_pred_var = model.predict_variance(
                test_df,
                horizon=h,
                context_df=context_for_test,
                strict_nonnegative=cfg.evaluation.strict_nonnegative,
            )
            if cfg.evaluation.strict_nonnegative:
                validate_nonnegative_forecasts(y_pred_var, model_name=m_name, quantity="variance")
                counters["n_nonnegativity_checks_passed"] += 1

            y_pred_vol = variance_to_vol(y_pred_var, ann)
            intervals = apply_forecast_intervals(
                y_pred_var, conf_table, annualization_factor=ann
            )

            store = model_eval_store[m_name]
            store["y_true_var"].append(y_test_var)
            store["y_pred_var"].append(y_pred_var)
            store["regimes"].append(test_regimes)
            store["split_qlikes"].append(
                qlike_loss(y_test_var, y_pred_var, variance_floor=v_floor)
            )
            store["split_maes_vol"].append(mae_loss(y_test_vol, y_pred_vol))

            for lvl in levels:
                store["intervals_vol"][lvl]["lower"].append(intervals[lvl]["lower_vol"])
                store["intervals_vol"][lvl]["upper"].append(intervals[lvl]["upper_vol"])

            counters["n_oos_predictions"] += len(y_pred_var)

            for i in range(len(y_pred_var)):
                row: dict[str, Any] = {
                    "split_id": split.split_id,
                    "horizon_days": h,
                    "model": m_name,
                    "asset_id": test_assets[i],
                    "timestamp": test_ts[i],
                    "regime": str(test_regimes[i]),
                    "actual_var": float(y_test_var[i]),
                    "pred_var": float(y_pred_var[i]),
                    "actual_vol": float(y_test_vol[i]),
                    "pred_vol": float(y_pred_vol[i]),
                }
                for lvl in levels:
                    pct = int(round(lvl * 100))
                    row[f"lower_vol_{pct}"] = float(intervals[lvl]["lower_vol"][i])
                    row[f"upper_vol_{pct}"] = float(intervals[lvl]["upper_vol"][i])
                pred_rows.append(row)

    # Aggregate OOS comparison table
    raw_metrics: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []

    for m_name, m_type, _, _ in model_items:
        store = model_eval_store[m_name]
        y_true_var = np.concatenate(store["y_true_var"])
        y_pred_var = np.concatenate(store["y_pred_var"])
        regimes_all = np.concatenate(store["regimes"])
        y_true_vol = variance_to_vol(y_true_var, ann)
        y_pred_vol = variance_to_vol(y_pred_var, ann)

        qlike_val = qlike_loss(y_true_var, y_pred_var, variance_floor=v_floor)
        mae_vol_val = mae_loss(y_true_vol, y_pred_vol)
        rmse_vol_val = rmse_loss(y_true_vol, y_pred_vol)
        mae_var_val = mae_loss(y_true_var, y_pred_var)
        bias_vol_val = float(np.mean(y_pred_vol - y_true_vol))
        mz = mincer_zarnowitz_regression(y_true_var, y_pred_var)

        metric_row: dict[str, Any] = {
            "model": m_name,
            "model_type": m_type,
            "horizon_days": h,
            "n_oos": int(len(y_true_var)),
            "n_splits": len(splits),
            "min_pred_var": float(np.min(y_pred_var)),
            "qlike": qlike_val,
            "qlike_split_std": float(np.std(store["split_qlikes"])),
            "mae_vol": mae_vol_val,
            "rmse_vol": rmse_vol_val,
            "mae_var": mae_var_val,
            "bias_vol": bias_vol_val,
            "mz_intercept": mz["mz_intercept"],
            "mz_slope": mz["mz_slope"],
            "mz_r2": mz["mz_r2"],
        }

        lower_80: np.ndarray | None = None
        upper_80: np.ndarray | None = None
        for lvl in levels:
            pct = int(round(lvl * 100))
            lo_v = np.concatenate(store["intervals_vol"][lvl]["lower"])
            hi_v = np.concatenate(store["intervals_vol"][lvl]["upper"])
            cal_m = interval_calibration_metrics(y_true_vol, lo_v, hi_v, nominal_level=lvl)
            metric_row[f"coverage_{pct}"] = cal_m["empirical_coverage"]
            metric_row[f"width_vol_{pct}"] = cal_m["mean_width_vol"]
            metric_row[f"winkler_{pct}"] = cal_m["winkler_score_vol"]
            if abs(lvl - 0.80) < 1e-6:
                lower_80 = lo_v
                upper_80 = hi_v

        raw_metrics.append(metric_row)
        regime_rows.extend(
            compute_regime_diagnostics(
                model_name=m_name,
                regimes=regimes_all,
                y_true_var=y_true_var,
                y_pred_var=y_pred_var,
                lower_80_vol=lower_80,
                upper_80_vol=upper_80,
                annualization_factor=ann,
                variance_floor=v_floor,
            )
        )

    # Compute relative skill vs naive baseline
    baseline_row = next(r for r in raw_metrics if r["model"] == baseline_name)
    base_qlike = max(float(baseline_row["qlike"]), 1e-12)
    base_mae = max(float(baseline_row["mae_vol"]), 1e-12)
    for r in raw_metrics:
        r["baseline_model"] = baseline_name
        r["qlike_vs_baseline_pct"] = (
            100.0 * (base_qlike - float(r["qlike"])) / base_qlike
            if float(baseline_row["qlike"]) > 1e-12
            else 0.0
        )
        r["mae_vol_vs_baseline_pct"] = (
            100.0 * (base_mae - float(r["mae_vol"])) / base_mae
            if float(baseline_row["mae_vol"]) > 1e-12
            else 0.0
        )

    comparison_df = pl.DataFrame(raw_metrics).sort("qlike")
    intervals_df = pl.DataFrame(pred_rows)
    regime_df = pl.DataFrame(regime_rows)

    return splits, comparison_df, intervals_df, regime_df, attempted_configs, counters


def run_pipeline(
    config: EngineConfig | str | Path,
    *,
    base_dir: Path | None = None,
    export_reports: bool = True,
) -> RunResult:
    """Execute full reproducible pipeline from config to versioned report artifacts."""
    if isinstance(config, (str, Path)):
        cfg_path = Path(config).resolve()
        cfg = load_config(cfg_path)
        root_dir = base_dir if base_dir is not None else cfg_path.parent.parent
    else:
        cfg = config
        validate_config(cfg)
        root_dir = base_dir if base_dir is not None else Path.cwd()

    returns_path = _resolve_path(cfg.data.returns_path, root_dir)
    intraday_path = (
        _resolve_path(cfg.data.intraday_bars_path, root_dir)
        if cfg.data.intraday_bars_path
        else None
    )
    labels_path = (
        _resolve_path(cfg.data.labels_path, root_dir) if cfg.data.labels_path else None
    )

    input_fps: dict[str, str] = {"returns": compute_file_sha256(returns_path)}
    returns_df = read_table(returns_path)

    intraday_df: pl.DataFrame | None = None
    if intraday_path is not None:
        input_fps["intraday_bars"] = compute_file_sha256(intraday_path)
        intraday_df = read_table(intraday_path)

    labels_df: pl.DataFrame | None = None
    if labels_path is not None:
        input_fps["labels"] = compute_file_sha256(labels_path)
        labels_df = read_table(labels_path)

    config_fp = cfg.fingerprint()
    run_id = _build_run_id(config_fp, input_fps, cfg.run_name)

    logger.info(
        "Starting run_id=%s config_fp=%s horizon=%d",
        run_id,
        config_fp,
        cfg.features.primary_horizon,
    )

    dataset, feat_diag = build_causal_dataset(
        returns_df,
        intraday_df,
        labels_df,
        horizons=cfg.features.horizons,
        annualization_factor=cfg.annualization_factor,
        expected_bars_per_session=cfg.expected_bars_per_session,
        include_overnight_gap=cfg.features.include_overnight_gap,
        min_bar_coverage_warn=cfg.features.min_bar_coverage_warn,
        variance_floor=cfg.features.variance_floor,
    )

    splits, comparison_df, intervals_df, regime_df, attempted_cfgs, counters = (
        evaluate_models_chronologically(dataset, cfg, horizon_days=cfg.features.primary_horizon)
    )

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "run_name": cfg.run_name,
        "config_version": cfg.version,
        "config_fingerprint": config_fp,
        "seed": cfg.seed,
        "horizon_days": cfg.features.primary_horizon,
        "annualization_factor": cfg.annualization_factor,
        "input_fingerprints": input_fps,
        "feature_diagnostics": {
            "n_rows": feat_diag.n_rows,
            "n_assets": feat_diag.n_assets,
            "n_sessions_with_intraday": feat_diag.n_sessions_with_intraday,
            "n_sessions_fallback_daily": feat_diag.n_sessions_fallback_daily,
            "n_sessions_low_coverage": feat_diag.n_sessions_low_coverage,
            "n_sessions_with_overnight_gap": feat_diag.n_sessions_with_overnight_gap,
            "min_bar_coverage": feat_diag.min_bar_coverage,
        },
        "splits": [s.summary_dict() for s in splits],
        "attempted_configurations": attempted_cfgs,
        "counters": counters,
        "runtime_environment": _runtime_environment(),
        "config": cfg.to_dict(),
    }

    output_paths: dict[str, str] = {}
    if export_reports:
        reports_dir = _resolve_path(cfg.output.reports_dir, root_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)

        comp_path = reports_dir / "model_comparison.csv"
        regime_path = reports_dir / "regime_diagnostics.csv"
        splits_path = reports_dir / "splits.json"
        manifest_path = reports_dir / "run_manifest.json"
        summary_path = reports_dir / "run_summary.json"

        comparison_df.write_csv(comp_path)
        regime_df.write_csv(regime_path)
        splits_path.write_text(
            json.dumps([s.summary_dict() for s in splits], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        summary_payload = {
            "run_id": run_id,
            "config_fingerprint": config_fp,
            "horizon_days": cfg.features.primary_horizon,
            "all_forecasts_nonnegative": bool(
                (comparison_df["min_pred_var"] >= 0.0).all()
            ),
            "counters": counters,
            "models": comparison_df.to_dicts(),
        }
        summary_path.write_text(
            json.dumps(summary_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        output_paths = {
            "model_comparison": str(comp_path),
            "regime_diagnostics": str(regime_path),
            "splits": str(splits_path),
            "run_manifest": str(manifest_path),
            "run_summary": str(summary_path),
        }

        if cfg.output.save_predictions:
            pred_path = reports_dir / "forecast_intervals.csv"
            intervals_df.write_csv(pred_path)
            output_paths["forecast_intervals"] = str(pred_path)

    return RunResult(
        run_id=run_id,
        config_fingerprint=config_fp,
        input_fingerprints=input_fps,
        horizon_days=cfg.features.primary_horizon,
        splits=splits,
        feature_diagnostics=feat_diag,
        model_comparison=comparison_df,
        forecast_intervals=intervals_df,
        regime_diagnostics=regime_df,
        attempted_configurations=attempted_cfgs,
        counters=counters,
        manifest=manifest,
        output_paths=output_paths,
    )
