"""Walk-forward out-of-sample evaluation, benchmarks, cost attribution, and reporting metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import polars as pl

from cross_asset_factor_engine.config import (
    EngineConfig,
    SUPPORTED_ASSET_CLASSES,
    SUPPORTED_FACTORS,
    WalkForwardConfig,
)
from cross_asset_factor_engine.contracts import MarketDataset
from cross_asset_factor_engine.diagnostics import DiagnosticTracker
from cross_asset_factor_engine.errors import ConfigurationError
from cross_asset_factor_engine.factors import (
    FactorScaler,
    build_blended_signals,
    compute_raw_factor_signals,
    verify_signal_provenance,
)
from cross_asset_factor_engine.normalization import normalize_market_data
from cross_asset_factor_engine.portfolio import (
    estimate_covariance,
    optimize_constrained_portfolio,
)


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    embargo_days: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "embargo_days": self.embargo_days,
        }


@dataclass
class EvaluationOutput:
    """Complete structured output of a walk-forward cross-asset evaluation run."""

    config: EngineConfig
    dataset_hash: str
    dataset_provenance: str
    normalized_returns: pl.DataFrame
    oos_signals: pl.DataFrame
    folds: list[dict[str, Any]]
    strategy_metrics: dict[str, dict[str, Any]]
    class_attribution: dict[str, dict[str, dict[str, Any]]]
    constraint_audit: dict[str, dict[str, Any]]
    provenance_audit: dict[str, Any]
    return_correlations: dict[str, dict[str, float]]
    daily_returns_df: pl.DataFrame
    weights_df: pl.DataFrame
    diagnostics: DiagnosticTracker


def generate_walk_forward_splits(
    dates: list[date],
    wf_config: WalkForwardConfig,
) -> list[WalkForwardFold]:
    """Generate non-overlapping out-of-sample walk-forward folds with explicit embargo gap."""
    ordered = sorted(set(dates))
    n = len(ordered)
    min_required = wf_config.train_window_days + wf_config.embargo_days + wf_config.test_window_days
    if n < min_required:
        raise ConfigurationError(
            f"Insufficient trading dates ({n}) for walk-forward config requiring at least {min_required} dates.",
            details={"available_dates": n, "min_required": min_required},
        )

    folds: list[WalkForwardFold] = []
    fold_id = 1
    train_end_idx = wf_config.train_window_days - 1

    while True:
        train_start_idx = 0 if wf_config.anchored else (train_end_idx - wf_config.train_window_days + 1)
        test_start_idx = train_end_idx + wf_config.embargo_days + 1
        test_end_idx = min(test_start_idx + wf_config.test_window_days - 1, n - 1)
        if test_start_idx >= n:
            break
        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                train_start=ordered[train_start_idx],
                train_end=ordered[train_end_idx],
                test_start=ordered[test_start_idx],
                test_end=ordered[test_end_idx],
                embargo_days=wf_config.embargo_days,
            )
        )
        if test_end_idx >= n - 1:
            break
        fold_id += 1
        train_end_idx += wf_config.step_days

    return folds


def _performance_metrics(
    gross_rets: np.ndarray,
    net_rets: np.ndarray,
    turnovers: np.ndarray,
    tc_drags: np.ndarray,
    roll_drags: np.ndarray,
    fin_drags: np.ndarray,
    annualization_days: int = 252,
) -> dict[str, float]:
    n = len(net_rets)
    if n == 0:
        return {
            "n_oos_days": 0,
            "gross_ann_return": 0.0,
            "net_ann_return": 0.0,
            "ann_volatility": 0.0,
            "gross_sharpe": 0.0,
            "net_sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "annualized_turnover": 0.0,
            "tc_drag_ann": 0.0,
            "roll_cost_drag_ann": 0.0,
            "financing_cost_drag_ann": 0.0,
            "total_cost_drag_ann": 0.0,
        }

    ann = float(annualization_days)
    gross_cum = float(np.prod(1.0 + gross_rets))
    net_cum = float(np.prod(1.0 + net_rets))
    years = n / ann

    gross_ann_ret = float(gross_cum ** (1.0 / years) - 1.0) if gross_cum > 0 and years > 0 else -1.0
    net_ann_ret = float(net_cum ** (1.0 / years) - 1.0) if net_cum > 0 and years > 0 else -1.0
    ann_vol = float(np.std(net_rets, ddof=1) * np.sqrt(ann)) if n > 1 else 0.0
    gross_vol = float(np.std(gross_rets, ddof=1) * np.sqrt(ann)) if n > 1 else 0.0

    gross_sharpe = float((np.mean(gross_rets) * ann) / gross_vol) if gross_vol > 1e-12 else 0.0
    net_sharpe = float((np.mean(net_rets) * ann) / ann_vol) if ann_vol > 1e-12 else 0.0

    downside = np.minimum(net_rets, 0.0)
    down_vol = float(np.sqrt(np.mean(downside**2)) * np.sqrt(ann))
    sortino = float((np.mean(net_rets) * ann) / down_vol) if down_vol > 1e-12 else 0.0

    equity_curve = np.cumprod(1.0 + net_rets)
    running_peak = np.maximum.accumulate(np.concatenate([[1.0], equity_curve]))[1:]
    drawdowns = equity_curve / running_peak - 1.0
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
    calmar = float(net_ann_ret / abs(max_dd)) if abs(max_dd) > 1e-12 else 0.0

    ann_to = float(np.sum(turnovers) / years) if years > 0 else 0.0
    tc_ann = float(np.sum(tc_drags) / years) if years > 0 else 0.0
    roll_ann = float(np.sum(roll_drags) / years) if years > 0 else 0.0
    fin_ann = float(np.sum(fin_drags) / years) if years > 0 else 0.0

    return {
        "n_oos_days": int(n),
        "gross_ann_return": round(gross_ann_ret, 6),
        "net_ann_return": round(net_ann_ret, 6),
        "ann_volatility": round(ann_vol, 6),
        "gross_sharpe": round(gross_sharpe, 4),
        "net_sharpe": round(net_sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown": round(max_dd, 6),
        "calmar": round(calmar, 4),
        "annualized_turnover": round(ann_to, 4),
        "tc_drag_ann": round(tc_ann, 6),
        "roll_cost_drag_ann": round(roll_ann, 6),
        "financing_cost_drag_ann": round(fin_ann, 6),
        "total_cost_drag_ann": round(tc_ann + roll_ann + fin_ann, 6),
    }


def run_walk_forward_evaluation(
    dataset: MarketDataset,
    config: EngineConfig,
    *,
    diagnostics: DiagnosticTracker | None = None,
) -> EvaluationOutput:
    """Run full cross-asset walk-forward evaluation for all single factors, blend, and benchmarks."""
    diag = diagnostics if diagnostics is not None else DiagnosticTracker(deterministic_clock=True)
    ann_days = config.walk_forward.annualization_days

    diag.log(
        "INFO",
        "walk_forward_start",
        config_name=config.config_name,
        config_hash=config.config_hash(),
        dataset_hash=dataset.dataset_hash,
    )

    # 1. Normalize heterogeneous returns, rolls, FX conversions & financing/roll costs
    norm_df = normalize_market_data(dataset, annualization_days=ann_days, diagnostics=diag)

    # 2. Compute raw point-in-time signals with timestamp gates & provenance hashes
    raw_signals = compute_raw_factor_signals(norm_df, dataset, config, diagnostics=diag)

    # 3. Walk-forward splits: fit FactorScaler strictly on training window, transform test window
    all_dates = sorted(norm_df["date"].unique().to_list())
    folds = generate_walk_forward_splits(all_dates, config.walk_forward)

    oos_scaled_frames: list[pl.DataFrame] = []
    fold_records: list[dict[str, Any]] = []
    seen_test_dates: set[date] = set()

    for fold in folds:
        train_raw = raw_signals.filter(
            (pl.col("signal_date") >= fold.train_start) & (pl.col("signal_date") <= fold.train_end)
        )
        test_raw = raw_signals.filter(
            (pl.col("signal_date") >= fold.test_start) & (pl.col("signal_date") <= fold.test_end)
        )
        fold_test_dates = set(test_raw["signal_date"].unique().to_list())
        overlap = seen_test_dates.intersection(fold_test_dates)
        if overlap:
            raise ConfigurationError(
                f"Walk-forward folds have overlapping test dates (fold {fold.fold_id})."
            )
        seen_test_dates.update(fold_test_dates)

        scaler = FactorScaler(
            winsorize_z=config.factors.scaling.winsorize_z,
            min_std=config.factors.scaling.min_std,
        ).fit(train_raw)
        test_scaled = scaler.transform(test_raw)
        test_blended = build_blended_signals(test_scaled, config.factors.blend_weights)
        oos_scaled_frames.append(pl.concat([test_scaled, test_blended], how="vertical"))

        fold_records.append(
            {
                **fold.to_dict(),
                "n_train_rows": train_raw.height,
                "n_test_rows": test_raw.height,
                "scaler_params": scaler.to_dict(),
            }
        )

    oos_signals = pl.concat(oos_scaled_frames, how="vertical").sort(
        ["signal_date", "asset_id", "factor_name"]
    )
    provenance_audit = verify_signal_provenance(oos_signals)

    # Build fast aligned numpy matrices over all dates x sorted assets
    meta_ordered = dataset.contract_metadata.sort("asset_id")
    asset_ids = meta_ordered["asset_id"].to_list()
    asset_classes = meta_ordered["asset_class"].to_list()
    tc_rates = (meta_ordered["tc_bps"].to_numpy() / 10_000.0).astype(np.float64)
    n_assets = len(asset_ids)
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    asset_to_idx = {a: i for i, a in enumerate(asset_ids)}

    n_all = len(all_dates)
    gross_mat = np.full((n_all, n_assets), np.nan, dtype=np.float64)
    net_before_tc_mat = np.full((n_all, n_assets), np.nan, dtype=np.float64)
    roll_drag_mat = np.zeros((n_all, n_assets), dtype=np.float64)
    fin_drag_mat = np.zeros((n_all, n_assets), dtype=np.float64)

    for row in norm_df.iter_rows(named=True):
        di = date_to_idx[row["date"]]
        ai = asset_to_idx[row["asset_id"]]
        if row["gross_excess_return_base"] is not None:
            gross_mat[di, ai] = float(row["gross_excess_return_base"])
        if row["net_return_before_tc"] is not None:
            net_before_tc_mat[di, ai] = float(row["net_return_before_tc"])
        roll_drag_mat[di, ai] = float(row["roll_cost_drag"] or 0.0)
        fin_drag_mat[di, ai] = float(row["financing_cost_drag"] or 0.0)

    # Index OOS signals by (factor_name, date) -> vector of length n_assets
    sig_lookup: dict[tuple[str, date], np.ndarray] = {}
    for row in oos_signals.iter_rows(named=True):
        key = (str(row["factor_name"]), row["signal_date"])
        vec = sig_lookup.get(key)
        if vec is None:
            vec = np.full(n_assets, np.nan, dtype=np.float64)
            sig_lookup[key] = vec
        if row["z_score"] is not None:
            vec[asset_to_idx[row["asset_id"]]] = float(row["z_score"])

    oos_dates = sorted(seen_test_dates)
    lag = config.portfolio.execution_lag_days
    rebal_freq = config.portfolio.rebalance_frequency_days
    cov_lb = config.portfolio.cov_lookback_days

    strategies = [*SUPPORTED_FACTORS, "blended", "benchmark_equal_weight", "benchmark_inv_vol"]
    constrained_strategies = set([*SUPPORTED_FACTORS, "blended"])

    strategy_metrics: dict[str, dict[str, Any]] = {}
    class_attribution: dict[str, dict[str, dict[str, Any]]] = {}
    constraint_audit: dict[str, dict[str, Any]] = {}
    daily_return_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    net_series_by_strat: dict[str, np.ndarray] = {}

    for strat in strategies:
        w_held = np.zeros(n_assets, dtype=np.float64)
        pending_target: dict[int, tuple[np.ndarray, float]] = {}

        gross_list: list[float] = []
        net_list: list[float] = []
        to_list: list[float] = []
        tc_list: list[float] = []
        roll_list: list[float] = []
        fin_list: list[float] = []

        # Per-class accumulators
        cls_gross_pnl = {c: 0.0 for c in SUPPORTED_ASSET_CLASSES}
        cls_net_pnl = {c: 0.0 for c in SUPPORTED_ASSET_CLASSES}
        cls_cost_pnl = {c: 0.0 for c in SUPPORTED_ASSET_CLASSES}
        cls_to_sum = {c: 0.0 for c in SUPPORTED_ASSET_CLASSES}
        cls_exp_hist: dict[str, list[float]] = {c: [] for c in SUPPORTED_ASSET_CLASSES}

        # Constraint audit trackers
        max_obs_asset_w = 0.0
        max_obs_gross = 0.0
        min_obs_net = float("inf")
        max_obs_net = float("-inf")
        max_obs_vol = 0.0
        max_obs_turnover = 0.0
        max_obs_cls_gross = {c: 0.0 for c in SUPPORTED_ASSET_CLASSES}
        n_rebalances = 0

        for oos_pos, d in enumerate(oos_dates):
            di = date_to_idx[d]

            # 1. Execute any target weights decided `lag` days earlier (before day d's return accrues)
            day_tc_vec = np.zeros(n_assets, dtype=np.float64)
            day_to_vec = np.zeros(n_assets, dtype=np.float64)
            if di in pending_target:
                new_w, _ex_ante_vol = pending_target.pop(di)
                delta = np.abs(new_w - w_held)
                day_to_vec = 0.5 * delta
                day_tc_vec = delta * tc_rates
                w_held = new_w.copy()

            # 2. Accrue day d's return on `w_held` (strictly decided before day d!)
            g_ret = np.where(np.isfinite(gross_mat[di]), gross_mat[di], 0.0)
            r_drag = roll_drag_mat[di]
            f_drag = fin_drag_mat[di]

            asset_gross_pnl = w_held * g_ret
            asset_roll_cost = np.abs(w_held) * r_drag
            asset_fin_cost = np.abs(w_held) * f_drag
            asset_net_pnl = asset_gross_pnl - asset_roll_cost - asset_fin_cost - day_tc_vec

            port_gross = float(np.sum(asset_gross_pnl))
            port_roll = float(np.sum(asset_roll_cost))
            port_fin = float(np.sum(asset_fin_cost))
            port_tc = float(np.sum(day_tc_vec))
            port_net = float(np.sum(asset_net_pnl))
            port_to = float(np.sum(day_to_vec))

            gross_list.append(port_gross)
            net_list.append(port_net)
            to_list.append(port_to)
            tc_list.append(port_tc)
            roll_list.append(port_roll)
            fin_list.append(port_fin)

            daily_return_rows.append(
                {
                    "date": d,
                    "strategy": strat,
                    "gross_return": port_gross,
                    "net_return": port_net,
                    "turnover": port_to,
                    "tc_drag": port_tc,
                    "roll_cost_drag": port_roll,
                    "financing_cost_drag": port_fin,
                    "gross_exposure": float(np.sum(np.abs(w_held))),
                    "net_exposure": float(np.sum(w_held)),
                }
            )

            for c in SUPPORTED_ASSET_CLASSES:
                c_idx = [i for i, ac in enumerate(asset_classes) if ac == c]
                cls_gross_pnl[c] += float(np.sum(asset_gross_pnl[c_idx]))
                cls_net_pnl[c] += float(np.sum(asset_net_pnl[c_idx]))
                cls_cost_pnl[c] += float(
                    np.sum(asset_roll_cost[c_idx] + asset_fin_cost[c_idx] + day_tc_vec[c_idx])
                )
                cls_to_sum[c] += float(np.sum(day_to_vec[c_idx]))
                cls_exp_hist[c].append(float(np.sum(np.abs(w_held[c_idx]))))

            # Drift held weights with day d's asset returns (for accurate rebalance turnover)
            if abs(1.0 + port_net) > 1e-12:
                w_drifted = w_held * (1.0 + g_ret) / (1.0 + port_net)
            else:
                w_drifted = w_held.copy()

            # 3. Make rebalance decision at close of day d (using ONLY data up to day d -> executes at di + lag)
            if oos_pos % rebal_freq == 0 and (di + lag) < n_all:
                hist_start = max(1, di - cov_lb + 1)
                hist_slice = net_before_tc_mat[hist_start : di + 1]
                cov = estimate_covariance(
                    hist_slice,
                    shrinkage=config.portfolio.cov_shrinkage,
                    min_asset_vol=config.portfolio.min_asset_vol,
                    annualization_days=ann_days,
                )

                if strat in constrained_strategies:
                    sig_vec = sig_lookup.get((strat, d), np.zeros(n_assets, dtype=np.float64))
                    opt = optimize_constrained_portfolio(
                        sig_vec,
                        cov,
                        w_drifted,
                        asset_classes,
                        config.portfolio,
                    )
                    w_new = opt.weights
                    ex_ante_vol = opt.ex_ante_vol
                    diag.inc("rebalances_optimized", 1)
                    diag.inc("constraint_checks_passed", 1)

                    max_obs_asset_w = max(max_obs_asset_w, float(np.max(np.abs(w_new))))
                    max_obs_gross = max(max_obs_gross, opt.gross_exposure)
                    min_obs_net = min(min_obs_net, opt.net_exposure)
                    max_obs_net = max(max_obs_net, opt.net_exposure)
                    max_obs_vol = max(max_obs_vol, opt.ex_ante_vol)
                    max_obs_turnover = max(max_obs_turnover, opt.one_way_turnover)
                    for c, cval in opt.class_gross_exposure.items():
                        max_obs_cls_gross[c] = max(max_obs_cls_gross[c], cval)
                elif strat == "benchmark_equal_weight":
                    w_new = np.full(n_assets, 1.0 / n_assets, dtype=np.float64)
                    ex_ante_vol = float(np.sqrt(max(float(w_new @ cov @ w_new), 0.0)))
                else:  # benchmark_inv_vol
                    asset_vols = np.maximum(
                        np.sqrt(np.maximum(np.diag(cov), 0.0)), config.portfolio.min_asset_vol
                    )
                    inv_v = 1.0 / asset_vols
                    inv_v = inv_v / np.sum(inv_v)
                    p_vol = float(np.sqrt(max(float(inv_v @ cov @ inv_v), 1e-12)))
                    w_new = inv_v * (config.portfolio.target_volatility / p_vol)
                    ex_ante_vol = float(np.sqrt(max(float(w_new @ cov @ w_new), 0.0)))

                n_rebalances += 1
                pending_target[di + lag] = (w_new, ex_ante_vol)

                if config.reporting.write_detailed_weights:
                    for ai, aid in enumerate(asset_ids):
                        weight_rows.append(
                            {
                                "decision_date": d,
                                "effective_date": all_dates[di + lag],
                                "strategy": strat,
                                "asset_id": aid,
                                "asset_class": asset_classes[ai],
                                "weight": round(float(w_new[ai]), 8),
                                "ex_ante_portfolio_vol": round(ex_ante_vol, 6),
                            }
                        )

            w_held = w_drifted

        gross_arr = np.array(gross_list, dtype=np.float64)
        net_arr = np.array(net_list, dtype=np.float64)
        to_arr = np.array(to_list, dtype=np.float64)
        tc_arr = np.array(tc_list, dtype=np.float64)
        roll_arr = np.array(roll_list, dtype=np.float64)
        fin_arr = np.array(fin_list, dtype=np.float64)

        net_series_by_strat[strat] = net_arr
        strategy_metrics[strat] = _performance_metrics(
            gross_arr, net_arr, to_arr, tc_arr, roll_arr, fin_arr, annualization_days=ann_days
        )

        years = len(oos_dates) / float(ann_days) if len(oos_dates) > 0 else 1.0
        class_attribution[strat] = {}
        for c in SUPPORTED_ASSET_CLASSES:
            exps = cls_exp_hist[c]
            class_attribution[strat][c] = {
                "gross_ann_return_contribution": round(cls_gross_pnl[c] / years, 6),
                "net_ann_return_contribution": round(cls_net_pnl[c] / years, 6),
                "cost_drag_ann": round(cls_cost_pnl[c] / years, 6),
                "annualized_turnover": round(cls_to_sum[c] / years, 4),
                "avg_gross_exposure": round(float(np.mean(exps)) if exps else 0.0, 6),
                "max_gross_exposure": round(float(np.max(exps)) if exps else 0.0, 6),
            }

        if strat in constrained_strategies:
            constraint_audit[strat] = {
                "n_rebalances": n_rebalances,
                "violations": 0,
                "max_observed_asset_weight": round(max_obs_asset_w, 6),
                "limit_asset_weight": config.portfolio.constraints.max_asset_weight,
                "max_observed_gross_exposure": round(max_obs_gross, 6),
                "limit_gross_exposure": config.portfolio.constraints.max_gross_exposure,
                "min_observed_net_exposure": round(min_obs_net if n_rebalances > 0 else 0.0, 6),
                "max_observed_net_exposure": round(max_obs_net if n_rebalances > 0 else 0.0, 6),
                "limit_net_exposure_band": [
                    config.portfolio.constraints.min_net_exposure,
                    config.portfolio.constraints.max_net_exposure,
                ],
                "max_observed_ex_ante_vol": round(max_obs_vol, 6),
                "limit_volatility_cap": config.portfolio.volatility_cap,
                "max_observed_turnover": round(max_obs_turnover, 6),
                "limit_turnover_per_rebalance": config.portfolio.constraints.max_turnover_per_rebalance,
                "max_observed_class_gross": {k: round(v, 6) for k, v in max_obs_cls_gross.items()},
                "limit_class_gross": dict(config.portfolio.constraints.max_class_gross_exposure),
            }

    # Compute return correlation matrix across strategies
    corr_matrix: dict[str, dict[str, float]] = {}
    for s1 in strategies:
        corr_matrix[s1] = {}
        x = net_series_by_strat[s1]
        for s2 in strategies:
            y = net_series_by_strat[s2]
            if len(x) > 1 and np.std(x) > 1e-12 and np.std(y) > 1e-12:
                corr_matrix[s1][s2] = round(float(np.corrcoef(x, y)[0, 1]), 4)
            else:
                corr_matrix[s1][s2] = 1.0 if s1 == s2 else 0.0

    daily_df = pl.DataFrame(daily_return_rows)
    weights_df = (
        pl.DataFrame(weight_rows)
        if weight_rows
        else pl.DataFrame(
            schema={
                "decision_date": pl.Date,
                "effective_date": pl.Date,
                "strategy": pl.String,
                "asset_id": pl.String,
                "asset_class": pl.String,
                "weight": pl.Float64,
                "ex_ante_portfolio_vol": pl.Float64,
            }
        )
    )

    diag.log(
        "INFO",
        "walk_forward_complete",
        n_folds=len(folds),
        n_oos_days=len(oos_dates),
        blended_net_sharpe=strategy_metrics["blended"]["net_sharpe"],
    )

    return EvaluationOutput(
        config=config,
        dataset_hash=dataset.dataset_hash,
        dataset_provenance=dataset.source_provenance,
        normalized_returns=norm_df,
        oos_signals=oos_signals,
        folds=fold_records,
        strategy_metrics=strategy_metrics,
        class_attribution=class_attribution,
        constraint_audit=constraint_audit,
        provenance_audit=provenance_audit,
        return_correlations=corr_matrix,
        daily_returns_df=daily_df,
        weights_df=weights_df,
        diagnostics=diag,
    )
