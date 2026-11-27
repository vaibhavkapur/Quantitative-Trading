"""Quantitative evaluation metrics: IC, ICIR, bootstrap uncertainty, multiple testing, duplication, stability."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
from scipy import stats

from alpha_engine.contracts.models import FoldMetrics, StabilityReport
from alpha_engine.errors import FactorDuplicationError


def compute_cross_sectional_ic(
    df: pl.DataFrame,
    *,
    signal_col: str = "neutralized_value",
    return_col: str = "fwd_return",
    min_cross_section: int = 5,
) -> pl.DataFrame:
    """Compute per-date Spearman rank IC and Pearson IC."""
    ordered = df.filter(
        pl.col(signal_col).is_finite() & pl.col(return_col).is_finite()
    ).sort(["event_ts_utc", "symbol"])

    rows: list[dict] = []
    for (dt,), grp in ordered.group_by(["event_ts_utc"], maintain_order=True):
        if grp.height < min_cross_section:
            continue
        sig = grp[signal_col].to_numpy()
        ret = grp[return_col].to_numpy()
        if np.std(sig) < 1e-12 or np.std(ret) < 1e-12:
            continue
        spearman_val = float(stats.spearmanr(sig, ret).statistic)
        pearson_val = float(np.corrcoef(sig, ret)[0, 1])
        if np.isfinite(spearman_val) and np.isfinite(pearson_val):
            rows.append(
                {
                    "event_ts_utc": dt,
                    "n_symbols": grp.height,
                    "spearman_ic": spearman_val,
                    "pearson_ic": pearson_val,
                }
            )

    if not rows:
        return pl.DataFrame(
            schema={
                "event_ts_utc": pl.Datetime("us", "UTC"),
                "n_symbols": pl.Int64,
                "spearman_ic": pl.Float64,
                "pearson_ic": pl.Float64,
            }
        )
    return pl.DataFrame(rows).sort("event_ts_utc")


def summarize_ic_series(
    ic_values: Sequence[float],
    *,
    n_bootstrap: int = 400,
    seed: int = 42,
) -> dict[str, float]:
    """Summarize daily IC series with ICIR, t-stat, p-value, hit rate, and bootstrap 95% CI."""
    arr = np.asarray([v for v in ic_values if np.isfinite(v)], dtype=np.float64)
    n = len(arr)
    if n < 2:
        return {
            "mean_ic": 0.0,
            "std_ic": 0.0,
            "icir": 0.0,
            "tstat": 0.0,
            "pvalue": 1.0,
            "hit_rate": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
        }

    mean_ic = float(np.mean(arr))
    std_ic = float(np.std(arr, ddof=1))
    icir = mean_ic / std_ic if std_ic > 1e-12 else 0.0
    se = std_ic / np.sqrt(n) if std_ic > 1e-12 else 0.0
    tstat = mean_ic / se if se > 1e-12 else 0.0
    # One-sided upper-tail test H1: mean_ic > 0 (since signals are oriented positively)
    pvalue = float(stats.t.sf(tstat, df=n - 1)) if se > 1e-12 else 1.0
    hit_rate = float(np.mean(arr > 0.0))

    # Block/IID deterministic bootstrap for uncertainty interval
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = float(np.mean(arr[idx]))
    ci_low = float(np.quantile(boot_means, 0.025))
    ci_high = float(np.quantile(boot_means, 0.975))

    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "icir": icir,
        "tstat": tstat,
        "pvalue": float(np.clip(pvalue, 0.0, 1.0)),
        "hit_rate": hit_rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def benjamini_hochberg_adjust(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg FDR adjusted p-values across multiple research trials."""
    m = len(pvalues)
    if m == 0:
        return []
    arr = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(arr)
    ranked = arr[order]
    adj = np.empty(m, dtype=np.float64)
    running_min = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = min(running_min, ranked[i] * m / rank)
        running_min = val
        adj[i] = min(1.0, max(0.0, val))
    out = np.empty(m, dtype=np.float64)
    out[order] = adj
    return [float(x) for x in out]


def compute_sector_exposure(weights_df: pl.DataFrame) -> float:
    """Return maximum absolute net weight in any single sector across all dates."""
    if weights_df.height == 0 or "sector" not in weights_df.columns:
        return 0.0
    sec_exp = (
        weights_df.group_by(["event_ts_utc", "sector"])
        .agg(pl.col("weight").sum().abs().alias("abs_sector_weight"))
    )
    val = sec_exp.select(pl.col("abs_sector_weight").max()).item()
    return float(val or 0.0)


def compute_factor_correlation_matrix(
    signal_frames: dict[str, pl.DataFrame],
    *,
    value_col: str = "neutralized_value",
) -> dict[str, dict[str, float]]:
    """Compute pairwise mean cross-sectional Spearman rank correlation across signals."""
    ids = sorted(signal_frames.keys())
    matrix: dict[str, dict[str, float]] = {sid: {other: 0.0 for other in ids} for sid in ids}
    for sid in ids:
        matrix[sid][sid] = 1.0

    for i, id_a in enumerate(ids):
        df_a = signal_frames[id_a].select("symbol", "event_ts_utc", pl.col(value_col).alias("val_a"))
        for id_b in ids[i + 1 :]:
            df_b = signal_frames[id_b].select(
                "symbol", "event_ts_utc", pl.col(value_col).alias("val_b")
            )
            joined = df_a.join(df_b, on=["symbol", "event_ts_utc"], how="inner").filter(
                pl.col("val_a").is_finite() & pl.col("val_b").is_finite()
            )
            daily_corrs: list[float] = []
            for _, grp in joined.group_by(["event_ts_utc"]):
                if grp.height < 5:
                    continue
                a = grp["val_a"].to_numpy()
                b = grp["val_b"].to_numpy()
                if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                    continue
                c = float(stats.spearmanr(a, b).statistic)
                if np.isfinite(c):
                    daily_corrs.append(c)
            mean_corr = float(np.mean(daily_corrs)) if daily_corrs else 0.0
            matrix[id_a][id_b] = round(mean_corr, 6)
            matrix[id_b][id_a] = round(mean_corr, 6)
    return matrix


def assert_no_factor_duplication(
    candidate_id: str,
    corr_matrix: dict[str, dict[str, float]],
    *,
    max_abs_correlation: float = 0.85,
) -> tuple[float, str | None]:
    """Return (max_abs_peer_corr, most_correlated_peer) or raise FactorDuplicationError."""
    row = corr_matrix.get(candidate_id, {})
    best_peer: str | None = None
    best_abs = 0.0
    for peer, corr in row.items():
        if peer == candidate_id:
            continue
        if abs(corr) > best_abs:
            best_abs = abs(corr)
            best_peer = peer
    if best_abs > max_abs_correlation and best_peer is not None:
        raise FactorDuplicationError(
            f"Candidate '{candidate_id}' is near-duplicate of '{best_peer}' "
            f"(|corr|={best_abs:.3f} > {max_abs_correlation:.3f}).",
            details={
                "candidate_id": candidate_id,
                "peer_id": best_peer,
                "abs_correlation": round(best_abs, 4),
                "threshold": max_abs_correlation,
            },
        )
    return best_abs, best_peer


def evaluate_stability(
    folds: Sequence[FoldMetrics],
    *,
    max_sign_flip_rate: float = 0.35,
    max_coefficient_cv: float = 1.50,
    min_ic_sign_consistency: float = 0.50,
) -> StabilityReport:
    """Assess fold-to-fold coefficient stability and OOS IC sign consistency."""
    if not folds:
        return StabilityReport(
            coefficient_sign_flip_rate=1.0,
            coefficient_cv=999.0,
            ic_sign_consistency=0.0,
            max_fold_ic_drop=0.0,
            is_stable=False,
            reasons=["no_walk_forward_folds"],
        )

    fold_ics = [f.oos_spearman_ic for f in folds]
    ic_sign_consistency = float(np.mean([ic > 0.0 for ic in fold_ics]))
    max_fold_ic_drop = float(max(fold_ics) - min(fold_ics))

    all_features = sorted({k for f in folds for k in f.coefficients})
    flip_rates: list[float] = []
    cvs: list[float] = []

    for feat in all_features:
        vals = [f.coefficients[feat] for f in folds if feat in f.coefficients]
        if len(vals) < 2:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        # Skip near-zero unused features
        if np.max(np.abs(arr)) < 1e-10:
            continue
        signs = np.sign(arr)
        flips = sum(
            1
            for idx in range(len(signs) - 1)
            if signs[idx] != 0 and signs[idx + 1] != 0 and signs[idx] != signs[idx + 1]
        )
        flip_rates.append(flips / max(1, len(signs) - 1))
        mean_abs = abs(float(np.mean(arr)))
        sd = float(np.std(arr, ddof=1))
        cv = sd / max(mean_abs, 1e-6)
        cvs.append(min(cv, 100.0))

    avg_flip = float(np.mean(flip_rates)) if flip_rates else 0.0
    avg_cv = float(np.mean(cvs)) if cvs else 0.0

    reasons: list[str] = []
    if avg_flip > max_sign_flip_rate:
        reasons.append(
            f"coefficient_sign_flip_rate={avg_flip:.3f} exceeds max {max_sign_flip_rate:.3f}"
        )
    if avg_cv > max_coefficient_cv:
        reasons.append(f"coefficient_cv={avg_cv:.3f} exceeds max {max_coefficient_cv:.3f}")
    if ic_sign_consistency < min_ic_sign_consistency:
        reasons.append(
            f"ic_sign_consistency={ic_sign_consistency:.3f} below min {min_ic_sign_consistency:.3f}"
        )

    return StabilityReport(
        coefficient_sign_flip_rate=round(avg_flip, 6),
        coefficient_cv=round(avg_cv, 6),
        ic_sign_consistency=round(ic_sign_consistency, 6),
        max_fold_ic_drop=round(max_fold_ic_drop, 6),
        is_stable=len(reasons) == 0,
        reasons=reasons,
    )
