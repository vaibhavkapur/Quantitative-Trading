"""Pure domain algorithms for cointegration testing, hedge ratios, and spread selection.

This module is strictly isolated from file I/O. All selection routines enforce
training-window timestamp boundaries to prevent train/test leakage.
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import numpy as np
import polars as pl
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from statarb.config import SelectionConfig
from statarb.exceptions import LeakageViolationError, UnsupportedInputError


@dataclass(frozen=True)
class SpreadModel:
    """Frozen cointegrating spread estimated strictly on a training window."""

    spread_id: str
    kind: Literal["pair", "basket"]
    symbols: tuple[str, ...]
    weights: tuple[float, ...]
    intercept: float
    spread_mean: float
    spread_std: float
    use_log_prices: bool
    estimator: str
    adf_statistic: float
    adf_pvalue: float
    adjusted_pvalue: float
    half_life_days: float
    correlation: float
    train_start: date
    train_end: date

    def transform_price(self, price: float) -> float:
        return math.log(price) if self.use_log_prices else float(price)

    def compute_spread_value(self, price_map: dict[str, float]) -> float:
        val = -self.intercept
        for sym, w in zip(self.symbols, self.weights, strict=True):
            val += w * self.transform_price(price_map[sym])
        return float(val)

    def compute_spread_series(self, price_matrix: np.ndarray) -> np.ndarray:
        """Compute spread over a (T, K) price matrix aligned with self.symbols."""
        arr = np.log(price_matrix) if self.use_log_prices else price_matrix.astype(np.float64)
        w = np.asarray(self.weights, dtype=np.float64)
        return arr @ w - self.intercept

    def to_dict(self) -> dict[str, Any]:
        return {
            "spread_id": self.spread_id,
            "kind": self.kind,
            "symbols": list(self.symbols),
            "weights": [round(float(w), 6) for w in self.weights],
            "intercept": round(float(self.intercept), 6),
            "spread_mean": round(float(self.spread_mean), 6),
            "spread_std": round(float(self.spread_std), 6),
            "use_log_prices": self.use_log_prices,
            "estimator": self.estimator,
            "adf_statistic": round(float(self.adf_statistic), 6),
            "adf_pvalue": round(float(self.adf_pvalue), 6),
            "adjusted_pvalue": round(float(self.adjusted_pvalue), 6),
            "half_life_days": round(float(self.half_life_days), 4)
            if math.isfinite(self.half_life_days)
            else None,
            "correlation": round(float(self.correlation), 6),
            "train_start": str(self.train_start),
            "train_end": str(self.train_end),
        }


@dataclass(frozen=True)
class CandidateDiagnostic:
    spread_id: str
    kind: Literal["pair", "basket"]
    symbols: tuple[str, ...]
    adf_statistic: float
    adf_pvalue: float
    adjusted_pvalue: float
    half_life_days: float
    correlation: float
    selected: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class FoldSelectionResult:
    train_start: date
    train_end: date
    selected_models: tuple[SpreadModel, ...]
    candidates: tuple[CandidateDiagnostic, ...]
    n_pairs_tested: int
    n_baskets_tested: int
    n_passed_raw_pvalue: int
    n_passed_adjusted_pvalue: int


def estimate_ols_hedge_ratio(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Estimate y = alpha + beta * x via Ordinary Least Squares (statsmodels)."""
    y_arr = np.asarray(y, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    if y_arr.ndim != 1 or x_arr.ndim != 1 or y_arr.size != x_arr.size:
        raise ValueError("y and x must be 1-D arrays of equal length")
    if y_arr.size < 10:
        raise ValueError("Need at least 10 observations for OLS hedge ratio estimation")
    design = sm.add_constant(x_arr, has_constant="add")
    fit = sm.OLS(y_arr, design).fit()
    alpha = float(fit.params[0])
    beta = float(fit.params[1])
    return alpha, beta


def estimate_tls_hedge_ratio(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Estimate y = alpha + beta * x via Total Least Squares (orthogonal regression via SVD)."""
    y_arr = np.asarray(y, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    if y_arr.ndim != 1 or x_arr.ndim != 1 or y_arr.size != x_arr.size:
        raise ValueError("y and x must be 1-D arrays of equal length")
    if y_arr.size < 10:
        raise ValueError("Need at least 10 observations for TLS hedge ratio estimation")

    x_mean = float(np.mean(x_arr))
    y_mean = float(np.mean(y_arr))
    centered = np.column_stack([x_arr - x_mean, y_arr - y_mean])
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    v_small = vt[-1]  # normal vector to best-fit line: v0*(x - mx) + v1*(y - my) = 0
    if abs(v_small[1]) < 1e-12:
        return estimate_ols_hedge_ratio(y_arr, x_arr)
    beta = float(-v_small[0] / v_small[1])
    alpha = float(y_mean - beta * x_mean)
    return alpha, beta


def estimate_half_life(spread: np.ndarray) -> float:
    """Estimate Ornstein-Uhlenbeck mean-reversion half-life in bars from discrete AR(1).

    Fits delta_s_t = c + lam * s_{t-1} + eps_t.
    For stationary mean reversion (-1 < lam < 0), phi = 1 + lam in (0, 1) and
    half_life = -ln(2) / ln(phi). Returns math.inf if non-mean-reverting.
    """
    s = np.asarray(spread, dtype=np.float64)
    if s.size < 10 or float(np.std(s)) < 1e-12:
        return math.inf
    s_lag = s[:-1]
    ds = np.diff(s)
    design = sm.add_constant(s_lag, has_constant="add")
    fit = sm.OLS(ds, design).fit()
    lam = float(fit.params[1])
    phi = 1.0 + lam
    if lam >= 0.0 or phi <= 0.0 or phi >= 1.0:
        return math.inf
    return float(-math.log(2.0) / math.log(phi))


def compute_adf_test(series: np.ndarray) -> tuple[float, float]:
    """Run Augmented Dickey-Fuller unit-root test with constant trend and AIC lag selection."""
    arr = np.asarray(series, dtype=np.float64)
    if arr.size < 15 or float(np.std(arr)) < 1e-12:
        return 0.0, 1.0
    maxlag = max(1, min(int(math.floor(12.0 * (arr.size / 100.0) ** 0.25)), arr.size // 4))
    res = adfuller(arr, maxlag=maxlag, regression="c", autolag="AIC", result_object=False)
    stat = float(res[0])
    pval = float(res[1])
    if not math.isfinite(pval):
        return 0.0, 1.0
    return stat, max(0.0, min(1.0, pval))


def test_engle_granger_pair(
    y: np.ndarray,
    x: np.ndarray,
    *,
    estimator: Literal["ols", "tls"] = "ols",
) -> dict[str, float]:
    """Estimate hedge ratio and test cointegration between two price series y and x."""
    y_arr = np.asarray(y, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)

    if estimator == "ols":
        alpha, beta = estimate_ols_hedge_ratio(y_arr, x_arr)
    elif estimator == "tls":
        alpha, beta = estimate_tls_hedge_ratio(y_arr, x_arr)
    else:
        raise UnsupportedInputError(f"Unsupported hedge_ratio_estimator: '{estimator}'")

    spread = y_arr - alpha - beta * x_arr
    spread_mean = float(np.mean(spread))
    spread_std = float(np.std(spread, ddof=1))

    # Engle-Granger MacKinnon critical-value test when OLS; ADF on residuals when TLS
    if estimator == "ols":
        try:
            t_stat, pvalue, _ = coint(y_arr, x_arr, trend="c", autolag="AIC")
            adf_stat = float(t_stat)
            adf_pval = float(pvalue) if math.isfinite(pvalue) else 1.0
        except Exception:
            adf_stat, adf_pval = compute_adf_test(spread)
    else:
        adf_stat, adf_pval = compute_adf_test(spread)

    half_life = estimate_half_life(spread)
    if float(np.std(y_arr)) < 1e-12 or float(np.std(x_arr)) < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(y_arr, x_arr)[0, 1])

    return {
        "alpha": alpha,
        "beta": beta,
        "spread_mean": spread_mean,
        "spread_std": spread_std,
        "adf_statistic": adf_stat,
        "adf_pvalue": max(0.0, min(1.0, adf_pval)),
        "half_life_days": half_life,
        "correlation": corr,
    }


# Prevent pytest from collecting this domain helper function as a test case
test_engle_granger_pair.__test__ = False  # type: ignore[attr-defined]


def estimate_johansen_basket(
    price_matrix: np.ndarray,
    symbols: tuple[str, ...],
) -> dict[str, Any] | None:
    """Estimate leading cointegrating basket vector using the Johansen trace test."""
    mat = np.asarray(price_matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[1] != len(symbols) or mat.shape[0] < 30:
        return None
    if np.any(np.std(mat, axis=0) < 1e-10):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=np.exceptions.ComplexWarning)
            joh = coint_johansen(mat, det_order=0, k_ar_diff=1)
    except Exception:
        return None

    trace_stat = float(np.real(joh.lr1[0]))
    crit_95 = float(np.real(joh.cvt[0, 1]))
    evec0 = joh.evec[:, 0]
    if np.any(np.abs(np.imag(evec0)) > 1e-8):
        return None
    raw_vec = np.asarray(np.real(evec0), dtype=np.float64)

    # Normalize so the first leg has weight +1.0 (or fall back if near zero)
    if abs(raw_vec[0]) < 1e-8:
        max_idx = int(np.argmax(np.abs(raw_vec)))
        if abs(raw_vec[max_idx]) < 1e-8:
            return None
        weights = raw_vec / raw_vec[max_idx]
    else:
        weights = raw_vec / raw_vec[0]

    # Guard against degenerate extreme leverage ratios
    if float(np.max(np.abs(weights))) > 10.0:
        return None

    raw_spread = mat @ weights
    intercept = float(np.mean(raw_spread))
    spread = raw_spread - intercept
    spread_std = float(np.std(spread, ddof=1))
    if spread_std < 1e-10:
        return None

    adf_stat, adf_pval = compute_adf_test(spread)
    half_life = estimate_half_life(spread)

    # Compute multiple R between leg 0 and remaining legs as correlation diagnostic
    y = mat[:, 0]
    x_rest = mat[:, 1:]
    try:
        fit = sm.OLS(y, sm.add_constant(x_rest, has_constant="add")).fit()
        mult_r = float(math.sqrt(max(0.0, min(1.0, fit.rsquared))))
    except Exception:
        mult_r = 0.0

    return {
        "weights": tuple(float(w) for w in weights),
        "intercept": intercept,
        "spread_mean": 0.0,
        "spread_std": spread_std,
        "adf_statistic": adf_stat,
        "adf_pvalue": adf_pval,
        "half_life_days": half_life,
        "correlation": mult_r,
        "trace_stat": trace_stat,
        "trace_crit_95": crit_95,
        "johansen_significant_95": trace_stat > crit_95,
    }


def adjust_pvalues(
    pvalues: Sequence[float],
    method: Literal["bh", "bonferroni", "none"] = "bh",
) -> list[float]:
    """Adjust raw p-values for multiple hypothesis testing.

    Supports:
    - "bh": Benjamini-Hochberg FDR step-up adjusted p-values (q-values)
    - "bonferroni": FWER Bonferroni correction (m * p, capped at 1.0)
    - "none": pass-through
    """
    m = len(pvalues)
    if m == 0:
        return []
    arr = np.clip(np.asarray(pvalues, dtype=np.float64), 0.0, 1.0)
    if method == "none":
        return [float(p) for p in arr]
    if method == "bonferroni":
        return [float(min(1.0, p * m)) for p in arr]
    if method == "bh":
        order = np.argsort(arr)
        sorted_p = arr[order]
        ranks = np.arange(1, m + 1, dtype=np.float64)
        adjusted_sorted = sorted_p * m / ranks
        # Enforce monotonicity from largest rank down to smallest
        adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
        adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
        out = np.empty(m, dtype=np.float64)
        out[order] = adjusted_sorted
        return [float(v) for v in out]
    raise UnsupportedInputError(f"Unsupported multiple_testing_method: '{method}'")


def _build_training_matrix(
    prices: pl.DataFrame,
    symbols: tuple[str, ...],
    train_start: date,
    train_end: date,
    *,
    use_log_prices: bool,
    max_allowed_date: date | None = None,
) -> tuple[list[date], dict[str, np.ndarray]]:
    if max_allowed_date is not None and train_end > max_allowed_date:
        raise LeakageViolationError(
            f"Selection train_end ({train_end}) exceeds max_allowed_date ({max_allowed_date})"
        )
    window = prices.filter(
        (pl.col("date") >= train_start)
        & (pl.col("date") <= train_end)
        & pl.col("symbol").is_in(list(symbols))
    )
    if window.height == 0:
        return [], {}

    max_seen = window["date"].max()
    if max_seen is not None and max_seen > train_end:
        raise LeakageViolationError(
            f"Training matrix observed date {max_seen} > train_end {train_end}"
        )

    pivoted = window.pivot(on="symbol", index="date", values="close_adj").sort("date").drop_nulls()
    if pivoted.height == 0:
        return [], {}

    dates = pivoted["date"].to_list()
    series_map: dict[str, np.ndarray] = {}
    for sym in symbols:
        if sym in pivoted.columns:
            raw = pivoted[sym].to_numpy().astype(np.float64)
            series_map[sym] = np.log(raw) if use_log_prices else raw
    return dates, series_map


def select_spreads_on_training_window(
    prices: pl.DataFrame,
    eligible_symbols: tuple[str, ...],
    sector_map: dict[str, str],
    train_start: date,
    train_end: date,
    cfg: SelectionConfig,
    *,
    use_log_prices: bool = True,
    same_sector_only: bool = True,
    max_allowed_date: date | None = None,
) -> FoldSelectionResult:
    """Select cointegrated pairs (and optional Johansen baskets) strictly on training data."""
    if max_allowed_date is not None and train_end > max_allowed_date:
        raise LeakageViolationError(
            f"Pair selection train_end ({train_end}) exceeds split boundary ({max_allowed_date})"
        )

    _, series_map = _build_training_matrix(
        prices,
        eligible_symbols,
        train_start,
        train_end,
        use_log_prices=use_log_prices,
        max_allowed_date=max_allowed_date,
    )
    avail_syms = tuple(sorted(s for s in eligible_symbols if s in series_map))
    if len(avail_syms) < 2:
        return FoldSelectionResult(
            train_start=train_start,
            train_end=train_end,
            selected_models=(),
            candidates=(),
            n_pairs_tested=0,
            n_baskets_tested=0,
            n_passed_raw_pvalue=0,
            n_passed_adjusted_pvalue=0,
        )

    # 1. Evaluate all candidate pairs
    pair_records: list[dict[str, Any]] = []
    for sym_a, sym_b in itertools.combinations(avail_syms, 2):
        if same_sector_only and sector_map.get(sym_a) != sector_map.get(sym_b):
            continue
        # Canonical deterministic ordering: test (sym_a on sym_b) and (sym_b on sym_a),
        # keep the direction with lower Engle-Granger p-value (tie-break by symbol order)
        res_ab = test_engle_granger_pair(
            series_map[sym_a], series_map[sym_b], estimator=cfg.hedge_ratio_estimator
        )
        res_ba = test_engle_granger_pair(
            series_map[sym_b], series_map[sym_a], estimator=cfg.hedge_ratio_estimator
        )
        if res_ab["adf_pvalue"] <= res_ba["adf_pvalue"]:
            y_sym, x_sym, best = sym_a, sym_b, res_ab
        else:
            y_sym, x_sym, best = sym_b, sym_a, res_ba

        spread_id = f"PAIR:{y_sym}|{x_sym}"
        pair_records.append(
            {
                "spread_id": spread_id,
                "kind": "pair",
                "symbols": (y_sym, x_sym),
                "weights": (1.0, -float(best["beta"])),
                "intercept": float(best["alpha"]),
                "spread_mean": float(best["spread_mean"]),
                "spread_std": float(best["spread_std"]),
                "estimator": cfg.hedge_ratio_estimator,
                "adf_statistic": float(best["adf_statistic"]),
                "adf_pvalue": float(best["adf_pvalue"]),
                "half_life_days": float(best["half_life_days"]),
                "correlation": float(best["correlation"]),
                "extra_valid": True,
                "extra_reason": None,
            }
        )

    pair_adj_pvals = adjust_pvalues(
        [r["adf_pvalue"] for r in pair_records], method=cfg.multiple_testing_method
    )
    for rec, adj_p in zip(pair_records, pair_adj_pvals, strict=True):
        rec["adjusted_pvalue"] = adj_p

    # 2. Optional Johansen basket candidates grouped by sector (or all symbols)
    basket_records: list[dict[str, Any]] = []
    if cfg.include_baskets and cfg.max_baskets_per_fold > 0 and len(avail_syms) >= cfg.basket_size:
        if same_sector_only:
            groups: dict[str, list[str]] = {}
            for s in avail_syms:
                groups.setdefault(sector_map.get(s, "UNKNOWN"), []).append(s)
            candidate_groups = [sorted(v) for v in groups.values() if len(v) >= cfg.basket_size]
        else:
            candidate_groups = [list(avail_syms)]

        for grp in candidate_groups:
            for combo in itertools.combinations(grp, cfg.basket_size):
                mat = np.column_stack([series_map[s] for s in combo])
                jres = estimate_johansen_basket(mat, combo)
                if jres is None:
                    continue
                spread_id = f"BASKET:{'|'.join(combo)}"
                basket_records.append(
                    {
                        "spread_id": spread_id,
                        "kind": "basket",
                        "symbols": combo,
                        "weights": jres["weights"],
                        "intercept": float(jres["intercept"]),
                        "spread_mean": float(jres["spread_mean"]),
                        "spread_std": float(jres["spread_std"]),
                        "estimator": "johansen",
                        "adf_statistic": float(jres["adf_statistic"]),
                        "adf_pvalue": float(jres["adf_pvalue"]),
                        "half_life_days": float(jres["half_life_days"]),
                        "correlation": float(jres["correlation"]),
                        "extra_valid": bool(jres["johansen_significant_95"]),
                        "extra_reason": None
                        if jres["johansen_significant_95"]
                        else "johansen_trace_below_95pct",
                    }
                )

        basket_adj_pvals = adjust_pvalues(
            [r["adf_pvalue"] for r in basket_records], method=cfg.multiple_testing_method
        )
        for rec, adj_p in zip(basket_records, basket_adj_pvals, strict=True):
            rec["adjusted_pvalue"] = adj_p

    def _evaluate_group(
        records: list[dict[str, Any]],
        max_keep: int,
    ) -> tuple[list[SpreadModel], list[CandidateDiagnostic], int, int]:
        passing: list[dict[str, Any]] = []
        diagnostics: list[CandidateDiagnostic] = []
        raw_pass_count = 0
        adj_pass_count = 0

        for rec in records:
            reasons: list[str] = []
            if not rec["extra_valid"] and rec["extra_reason"]:
                reasons.append(rec["extra_reason"])
            if rec["adf_pvalue"] <= cfg.max_adf_pvalue:
                raw_pass_count += 1
            else:
                reasons.append(f"adf_pvalue({rec['adf_pvalue']:.4f}>{cfg.max_adf_pvalue})")

            if cfg.multiple_testing_method != "none":
                if (
                    rec["adf_pvalue"] <= cfg.max_adf_pvalue
                    and rec["adjusted_pvalue"] <= cfg.fdr_alpha
                ):
                    adj_pass_count += 1
                elif rec["adjusted_pvalue"] > cfg.fdr_alpha:
                    reasons.append(f"adjusted_pvalue({rec['adjusted_pvalue']:.4f}>{cfg.fdr_alpha})")
            else:
                if rec["adf_pvalue"] <= cfg.max_adf_pvalue:
                    adj_pass_count += 1

            hl = rec["half_life_days"]
            if not math.isfinite(hl) or hl < cfg.min_half_life_days or hl > cfg.max_half_life_days:
                reasons.append(
                    f"half_life_out_of_range({hl:.2f} not in "
                    f"[{cfg.min_half_life_days},{cfg.max_half_life_days}])"
                )
            if abs(rec["correlation"]) < cfg.min_correlation:
                reasons.append(f"low_correlation({rec['correlation']:.3f}<{cfg.min_correlation})")
            if rec["spread_std"] <= 1e-8:
                reasons.append("zero_spread_variance")

            rec["reasons"] = tuple(reasons)
            if not reasons:
                passing.append(rec)

        passing.sort(
            key=lambda r: (
                r["adjusted_pvalue"],
                r["adf_pvalue"],
                r["half_life_days"],
                r["spread_id"],
            )
        )
        chosen_ids = {r["spread_id"] for r in passing[:max_keep]}

        models: list[SpreadModel] = []
        for rec in passing[:max_keep]:
            models.append(
                SpreadModel(
                    spread_id=rec["spread_id"],
                    kind=rec["kind"],
                    symbols=rec["symbols"],
                    weights=rec["weights"],
                    intercept=rec["intercept"],
                    spread_mean=rec["spread_mean"],
                    spread_std=rec["spread_std"],
                    use_log_prices=use_log_prices,
                    estimator=rec["estimator"],
                    adf_statistic=rec["adf_statistic"],
                    adf_pvalue=rec["adf_pvalue"],
                    adjusted_pvalue=rec["adjusted_pvalue"],
                    half_life_days=rec["half_life_days"],
                    correlation=rec["correlation"],
                    train_start=train_start,
                    train_end=train_end,
                )
            )

        for rec in records:
            is_sel = rec["spread_id"] in chosen_ids
            reasons = rec["reasons"]
            if not reasons and not is_sel:
                reasons = ("rank_cutoff",)
            diagnostics.append(
                CandidateDiagnostic(
                    spread_id=rec["spread_id"],
                    kind=rec["kind"],
                    symbols=rec["symbols"],
                    adf_statistic=rec["adf_statistic"],
                    adf_pvalue=rec["adf_pvalue"],
                    adjusted_pvalue=rec["adjusted_pvalue"],
                    half_life_days=rec["half_life_days"],
                    correlation=rec["correlation"],
                    selected=is_sel,
                    rejection_reasons=reasons,
                )
            )
        return models, diagnostics, raw_pass_count, adj_pass_count

    pair_models, pair_diags, pair_raw, pair_adj = _evaluate_group(
        pair_records, cfg.max_pairs_per_fold
    )
    basket_models, basket_diags, basket_raw, basket_adj = _evaluate_group(
        basket_records, cfg.max_baskets_per_fold
    )

    return FoldSelectionResult(
        train_start=train_start,
        train_end=train_end,
        selected_models=tuple(pair_models + basket_models),
        candidates=tuple(pair_diags + basket_diags),
        n_pairs_tested=len(pair_records),
        n_baskets_tested=len(basket_records),
        n_passed_raw_pvalue=pair_raw + basket_raw,
        n_passed_adjusted_pvalue=pair_adj + basket_adj,
    )
