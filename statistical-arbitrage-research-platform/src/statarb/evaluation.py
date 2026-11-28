"""Statistical evaluation, uncertainty, stability, multiple testing, and benchmarks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import norm

from statarb.backtest import BacktestResult, run_walk_forward_backtest
from statarb.cointegration import compute_adf_test
from statarb.config import EvaluationConfig, PlatformConfig


@dataclass(frozen=True)
class PerformanceSummary:
    n_oos_days: int
    total_return_net: float
    total_return_gross: float
    cagr_net: float
    cagr_gross: float
    annualized_vol_net: float
    sharpe_net: float
    sharpe_gross: float
    sortino_net: float
    max_drawdown_net: float
    calmar_net: float
    annualized_turnover: float
    total_commission: float
    total_slippage: float
    total_borrow_cost: float
    total_cost_drag_bps: float
    n_trades: int
    positive_day_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_oos_days": self.n_oos_days,
            "total_return_net": round(self.total_return_net, 6),
            "total_return_gross": round(self.total_return_gross, 6),
            "cagr_net": round(self.cagr_net, 6),
            "cagr_gross": round(self.cagr_gross, 6),
            "annualized_vol_net": round(self.annualized_vol_net, 6),
            "sharpe_net": round(self.sharpe_net, 4),
            "sharpe_gross": round(self.sharpe_gross, 4),
            "sortino_net": round(self.sortino_net, 4),
            "max_drawdown_net": round(self.max_drawdown_net, 6),
            "calmar_net": round(self.calmar_net, 4),
            "annualized_turnover": round(self.annualized_turnover, 4),
            "total_commission": round(self.total_commission, 4),
            "total_slippage": round(self.total_slippage, 4),
            "total_borrow_cost": round(self.total_borrow_cost, 4),
            "total_cost_drag_bps": round(self.total_cost_drag_bps, 2),
            "n_trades": self.n_trades,
            "positive_day_rate": round(self.positive_day_rate, 4),
        }


def _sharpe_ratio(returns: np.ndarray, ann_days: float, rf_annual: float = 0.0) -> float:
    if returns.size < 2:
        return 0.0
    rf_daily = rf_annual / ann_days
    excess = returns - rf_daily
    sd = float(np.std(excess, ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float((np.mean(excess) / sd) * math.sqrt(ann_days))


def _max_drawdown_from_returns(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    wealth = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(wealth)
    dd = (wealth / np.maximum(peak, 1e-12)) - 1.0
    return float(np.min(dd))


def compute_performance_summary(
    result: BacktestResult,
    cfg: PlatformConfig,
) -> PerformanceSummary:
    """Compute OOS net and gross performance metrics from a BacktestResult."""
    eq_records = result.daily_equity
    n_days = len(eq_records)
    ann_days = float(cfg.evaluation.annualization_days)
    rf = cfg.evaluation.risk_free_rate_annual

    if n_days == 0:
        return PerformanceSummary(
            n_oos_days=0,
            total_return_net=0.0,
            total_return_gross=0.0,
            cagr_net=0.0,
            cagr_gross=0.0,
            annualized_vol_net=0.0,
            sharpe_net=0.0,
            sharpe_gross=0.0,
            sortino_net=0.0,
            max_drawdown_net=0.0,
            calmar_net=0.0,
            annualized_turnover=0.0,
            total_commission=0.0,
            total_slippage=0.0,
            total_borrow_cost=0.0,
            total_cost_drag_bps=0.0,
            n_trades=len(result.trades),
            positive_day_rate=0.0,
        )

    ret_net = np.asarray([r.daily_return_net for r in eq_records], dtype=np.float64)
    ret_gross = np.asarray([r.daily_return_gross for r in eq_records], dtype=np.float64)
    turnover = np.asarray([r.turnover for r in eq_records], dtype=np.float64)

    total_net = float(np.prod(1.0 + ret_net) - 1.0)
    total_gross = float(np.prod(1.0 + ret_gross) - 1.0)
    years = n_days / ann_days
    cagr_net = (
        float((1.0 + total_net) ** (1.0 / years) - 1.0)
        if (1.0 + total_net) > 0 and years > 0
        else -1.0
    )
    cagr_gross = (
        float((1.0 + total_gross) ** (1.0 / years) - 1.0)
        if (1.0 + total_gross) > 0 and years > 0
        else -1.0
    )
    ann_vol = float(np.std(ret_net, ddof=1) * math.sqrt(ann_days)) if n_days > 1 else 0.0

    sharpe_net = _sharpe_ratio(ret_net, ann_days, rf)
    sharpe_gross = _sharpe_ratio(ret_gross, ann_days, rf)

    downside = ret_net[ret_net < 0.0]
    downside_std = (
        float(np.std(downside, ddof=1) * math.sqrt(ann_days)) if downside.size > 1 else 0.0
    )
    sortino_net = (
        float(((np.mean(ret_net) * ann_days) - rf) / downside_std) if downside_std > 1e-12 else 0.0
    )

    max_dd = _max_drawdown_from_returns(ret_net)
    calmar = float(cagr_net / abs(max_dd)) if abs(max_dd) > 1e-12 else 0.0

    tot_comm = float(sum(r.commission for r in eq_records))
    tot_slip = float(sum(r.slippage for r in eq_records))
    tot_borrow = float(sum(r.borrow_cost for r in eq_records))
    cost_drag_bps = (total_gross - total_net) * 10_000.0
    pos_rate = float(np.mean(ret_net > 0.0))

    return PerformanceSummary(
        n_oos_days=n_days,
        total_return_net=total_net,
        total_return_gross=total_gross,
        cagr_net=cagr_net,
        cagr_gross=cagr_gross,
        annualized_vol_net=ann_vol,
        sharpe_net=sharpe_net,
        sharpe_gross=sharpe_gross,
        sortino_net=sortino_net,
        max_drawdown_net=max_dd,
        calmar_net=calmar,
        annualized_turnover=float(np.mean(turnover) * ann_days),
        total_commission=tot_comm,
        total_slippage=tot_slip,
        total_borrow_cost=tot_borrow,
        total_cost_drag_bps=cost_drag_bps,
        n_trades=len(result.trades),
        positive_day_rate=pos_rate,
    )


def compute_oos_uncertainty(
    returns: np.ndarray,
    eval_cfg: EvaluationConfig,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Quantify OOS return and Sharpe uncertainty via circular block bootstrap & analytical SE."""
    r = np.asarray(returns, dtype=np.float64)
    n = r.size
    ann_days = float(eval_cfg.annualization_days)
    alpha = 1.0 - eval_cfg.confidence_level
    if n < 5:
        return {
            "confidence_level": eval_cfg.confidence_level,
            "bootstrap_samples": 0,
            "sharpe_point": 0.0,
            "sharpe_se_analytical": None,
            "sharpe_ci_low": None,
            "sharpe_ci_high": None,
            "ann_return_ci_low": None,
            "ann_return_ci_high": None,
        }

    point_sharpe = _sharpe_ratio(r, ann_days, eval_cfg.risk_free_rate_annual)
    # Analytical non-normal Sharpe standard error (Lo 2002 / Mertens 2002)
    centered = r - np.mean(r)
    sd = float(np.std(r, ddof=1))
    if sd > 1e-12:
        skew = float(np.mean((centered / sd) ** 3))
        kurt = float(np.mean((centered / sd) ** 4))
        sr_daily = point_sharpe / math.sqrt(ann_days)
        var_sr_daily = max(
            1e-12,
            (1.0 - skew * sr_daily + ((kurt - 1.0) / 4.0) * (sr_daily**2)) / max(n - 1, 1),
        )
        sharpe_se_ann = float(math.sqrt(var_sr_daily) * math.sqrt(ann_days))
    else:
        skew = 0.0
        kurt = 3.0
        sharpe_se_ann = 0.0

    # Circular moving-block bootstrap for autocorrelation-aware CIs
    rng = np.random.default_rng(seed)
    block = min(eval_cfg.bootstrap_block_size, n)
    n_blocks = int(math.ceil(n / block))
    boot_sharpes = np.empty(eval_cfg.bootstrap_samples, dtype=np.float64)
    boot_ann_rets = np.empty(eval_cfg.bootstrap_samples, dtype=np.float64)

    for b in range(eval_cfg.bootstrap_samples):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]
        sample = r[idx]
        boot_sharpes[b] = _sharpe_ratio(sample, ann_days, eval_cfg.risk_free_rate_annual)
        boot_ann_rets[b] = float(np.mean(sample) * ann_days)

    return {
        "confidence_level": eval_cfg.confidence_level,
        "bootstrap_samples": eval_cfg.bootstrap_samples,
        "bootstrap_block_size": block,
        "sharpe_point": round(point_sharpe, 4),
        "sharpe_se_analytical": round(sharpe_se_ann, 4),
        "sharpe_ci_low": round(float(np.quantile(boot_sharpes, alpha / 2.0)), 4),
        "sharpe_ci_high": round(float(np.quantile(boot_sharpes, 1.0 - alpha / 2.0)), 4),
        "ann_return_point": round(float(np.mean(r) * ann_days), 6),
        "ann_return_ci_low": round(float(np.quantile(boot_ann_rets, alpha / 2.0)), 6),
        "ann_return_ci_high": round(float(np.quantile(boot_ann_rets, 1.0 - alpha / 2.0)), 6),
        "return_skewness": round(skew, 4),
        "return_kurtosis": round(kurt, 4),
    }


def compute_deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    *,
    ann_days: float = 252.0,
    rf_annual: float = 0.0,
    trial_sharpe_variance: float | None = None,
) -> dict[str, Any]:
    """Compute Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) across attempted trials."""
    r = np.asarray(returns, dtype=np.float64)
    n = r.size
    if n < 10 or float(np.std(r, ddof=1)) <= 1e-12:
        return {
            "n_trials": n_trials,
            "observed_sharpe_annual": 0.0,
            "expected_max_null_sharpe_annual": 0.0,
            "deflated_sharpe_prob": 0.0,
        }

    sr_ann = _sharpe_ratio(r, ann_days, rf_annual)
    sr_daily = sr_ann / math.sqrt(ann_days)

    centered = r - np.mean(r)
    sd = float(np.std(r, ddof=1))
    skew = float(np.mean((centered / sd) ** 3))
    kurt = float(np.mean((centered / sd) ** 4))

    # Expected maximum Sharpe under null across N trials (Euler-Mascheroni approximation)
    gamma_em = 0.5772156649
    if n_trials <= 1:
        sr0_daily = 0.0
    else:
        z1 = float(norm.ppf(1.0 - 1.0 / n_trials))
        z2 = float(norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
        sigma_sr = (
            math.sqrt(max(trial_sharpe_variance, 1e-12)) / math.sqrt(ann_days)
            if trial_sharpe_variance is not None and trial_sharpe_variance > 0.0
            else 1.0 / math.sqrt(max(n - 1, 1))
        )
        sr0_daily = sigma_sr * ((1.0 - gamma_em) * z1 + gamma_em * z2)

    denom_inner = max(1e-12, 1.0 - skew * sr_daily + ((kurt - 1.0) / 4.0) * (sr_daily**2))
    z_dsr = ((sr_daily - sr0_daily) * math.sqrt(max(n - 1, 1))) / math.sqrt(denom_inner)
    dsr_prob = float(norm.cdf(z_dsr))

    return {
        "n_trials": n_trials,
        "observed_sharpe_annual": round(sr_ann, 4),
        "expected_max_null_sharpe_annual": round(sr0_daily * math.sqrt(ann_days), 4),
        "deflated_sharpe_z": round(float(z_dsr), 4),
        "deflated_sharpe_prob": round(dsr_prob, 4),
    }


def compute_stability_report(
    result: BacktestResult,
    prices: pl.DataFrame,
) -> dict[str, Any]:
    """Assess fold-to-fold stability, hedge-ratio drift, pair turnover, and OOS stationarity."""
    fold_net_rets = [f.oos_net_return for f in result.folds]
    positive_folds = sum(1 for r in fold_net_rets if r > 0.0)

    # Track pair persistence & hedge ratio drift across consecutive folds
    prev_ids: set[str] | None = None
    jaccard_overlaps: list[float] = []
    beta_history: dict[str, list[float]] = {}

    # Check OOS ADF stationarity retention of selected spreads during their test windows
    oos_adf_tests = 0
    oos_adf_retained_10pct = 0

    for fold_rec in result.folds:
        curr_ids = {m.spread_id for m in fold_rec.selection.selected_models}
        if prev_ids is not None:
            union = prev_ids | curr_ids
            inter = prev_ids & curr_ids
            jaccard_overlaps.append(len(inter) / len(union) if union else 1.0)
        prev_ids = curr_ids

        test_slice = prices.filter(
            (pl.col("date") >= fold_rec.fold.test_start)
            & (pl.col("date") <= fold_rec.fold.test_end)
        )
        piv = test_slice.pivot(on="symbol", index="date", values="close_adj").sort("date")

        for m in fold_rec.selection.selected_models:
            if m.kind == "pair" and len(m.weights) == 2:
                beta_history.setdefault(m.spread_id, []).append(-m.weights[1])
            if all(s in piv.columns for s in m.symbols):
                sub = piv.select(list(m.symbols)).drop_nulls()
                if sub.height >= 20:
                    mat = sub.to_numpy().astype(np.float64)
                    oos_spread = m.compute_spread_series(mat)
                    _, oos_pval = compute_adf_test(oos_spread)
                    oos_adf_tests += 1
                    if oos_pval <= 0.10:
                        oos_adf_retained_10pct += 1

    hedge_ratio_drift: dict[str, dict[str, float]] = {}
    for sid, betas in sorted(beta_history.items()):
        if len(betas) >= 2:
            hedge_ratio_drift[sid] = {
                "n_folds_selected": len(betas),
                "mean_beta": round(float(np.mean(betas)), 4),
                "std_beta": round(float(np.std(betas, ddof=1)), 4),
                "max_abs_drift": round(float(max(betas) - min(betas)), 4),
            }

    return {
        "n_folds": len(result.folds),
        "positive_oos_folds": positive_folds,
        "positive_oos_fold_rate": round(positive_folds / len(result.folds), 4)
        if result.folds
        else 0.0,
        "fold_net_return_mean": round(float(np.mean(fold_net_rets)), 6) if fold_net_rets else 0.0,
        "fold_net_return_std": round(float(np.std(fold_net_rets, ddof=1)), 6)
        if len(fold_net_rets) > 1
        else 0.0,
        "mean_fold_pair_jaccard_stability": round(float(np.mean(jaccard_overlaps)), 4)
        if jaccard_overlaps
        else None,
        "oos_stationarity_tests": oos_adf_tests,
        "oos_stationarity_retained_p10_rate": round(oos_adf_retained_10pct / oos_adf_tests, 4)
        if oos_adf_tests > 0
        else None,
        "hedge_ratio_drift": hedge_ratio_drift,
    }


def compute_equal_weight_benchmark(
    prices: pl.DataFrame,
    oos_dates: list[date],
    cfg: PlatformConfig,
) -> dict[str, Any]:
    """Compute equal-weight active-universe buy-and-hold benchmark on exact OOS dates."""
    if not oos_dates:
        return {
            "name": "equal_weight_universe",
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }

    date_set = set(oos_dates)
    sub = (
        prices.filter(pl.col("is_active"))
        .sort(["symbol", "date"])
        .with_columns(
            (pl.col("close_adj") / pl.col("close_adj").shift(1).over("symbol") - 1.0).alias("ret")
        )
        .filter(pl.col("date").is_in(list(date_set)) & pl.col("ret").is_not_null())
    )
    daily = sub.group_by("date").agg(pl.col("ret").mean().alias("ew_ret")).sort("date")
    rets = daily["ew_ret"].to_numpy().astype(np.float64)
    ann_days = float(cfg.evaluation.annualization_days)
    tot_ret = float(np.prod(1.0 + rets) - 1.0) if rets.size > 0 else 0.0
    years = rets.size / ann_days if rets.size > 0 else 1.0
    cagr = (
        float((1.0 + tot_ret) ** (1.0 / years) - 1.0) if (1.0 + tot_ret) > 0 and years > 0 else -1.0
    )
    return {
        "name": "equal_weight_universe",
        "n_oos_days": int(rets.size),
        "total_return": round(tot_ret, 6),
        "cagr": round(cagr, 6),
        "annualized_vol": round(float(np.std(rets, ddof=1) * math.sqrt(ann_days)), 6)
        if rets.size > 1
        else 0.0,
        "sharpe": round(_sharpe_ratio(rets, ann_days, cfg.evaluation.risk_free_rate_annual), 4),
        "max_drawdown": round(_max_drawdown_from_returns(rets), 6),
    }


def run_benchmark_comparisons(
    prices: pl.DataFrame,
    borrow: pl.DataFrame,
    cfg: PlatformConfig,
    primary_result: BacktestResult,
) -> dict[str, Any]:
    """Compare primary net strategy against frictionless zero-cost oracle and EW benchmark."""
    oos_dates = [r.date for r in primary_result.daily_equity]
    ew_bench = compute_equal_weight_benchmark(prices, oos_dates, cfg)

    zero_cost_res = run_walk_forward_backtest(prices, borrow, cfg, zero_cost=True)
    zero_cost_perf = compute_performance_summary(zero_cost_res, cfg)

    primary_perf = compute_performance_summary(primary_result, cfg)

    return {
        "equal_weight_universe": ew_bench,
        "zero_cost_strategy": {
            "name": "zero_cost_frictionless_strategy",
            "total_return_net": round(zero_cost_perf.total_return_net, 6),
            "cagr_net": round(zero_cost_perf.cagr_net, 6),
            "sharpe_net": round(zero_cost_perf.sharpe_net, 4),
            "max_drawdown_net": round(zero_cost_perf.max_drawdown_net, 6),
            "n_trades": zero_cost_perf.n_trades,
        },
        "primary_vs_zero_cost_delta": {
            "total_return_cost_impact": round(
                primary_perf.total_return_net - zero_cost_perf.total_return_net, 6
            ),
            "sharpe_cost_impact": round(primary_perf.sharpe_net - zero_cost_perf.sharpe_net, 4),
        },
        "primary_vs_equal_weight_delta": {
            "excess_total_return": round(
                primary_perf.total_return_net - ew_bench["total_return"], 6
            ),
            "sharpe_delta": round(primary_perf.sharpe_net - ew_bench["sharpe"], 4),
        },
    }
