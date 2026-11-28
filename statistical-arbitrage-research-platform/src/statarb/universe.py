"""Point-in-time universe filtering strictly bounded to training windows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from statarb.config import UniverseConfig
from statarb.exceptions import LeakageViolationError


@dataclass(frozen=True)
class UniverseFilterResult:
    formation_start: date
    formation_end: date
    eligible_symbols: tuple[str, ...]
    excluded_reasons: dict[str, tuple[str, ...]]
    sector_map: dict[str, str]


def filter_universe(
    prices: pl.DataFrame,
    formation_start: date,
    formation_end: date,
    cfg: UniverseConfig,
    *,
    max_allowed_date: date | None = None,
) -> UniverseFilterResult:
    """Filter tradable universe using strictly point-in-time training data.

    Parameters
    ----------
    prices:
        Validated price DataFrame.
    formation_start, formation_end:
        Inclusive training window boundaries.
    cfg:
        Universe filtering rules.
    max_allowed_date:
        Optional split boundary guard; raises LeakageViolationError if
        formation_end exceeds max_allowed_date.
    """
    if max_allowed_date is not None and formation_end > max_allowed_date:
        raise LeakageViolationError(
            f"Universe formation_end ({formation_end}) exceeds "
            f"max_allowed_date ({max_allowed_date})"
        )
    if formation_start > formation_end:
        raise ValueError(
            f"formation_start ({formation_start}) must be <= formation_end ({formation_end})"
        )

    window = prices.filter(
        (pl.col("date") >= formation_start) & (pl.col("date") <= formation_end)
    ).sort(["symbol", "date"])

    if window.height == 0:
        return UniverseFilterResult(
            formation_start=formation_start,
            formation_end=formation_end,
            eligible_symbols=(),
            excluded_reasons={},
            sector_map={},
        )

    # Latest available bar strictly <= formation_end
    window_end_date = window["date"].max()

    summary = (
        window.group_by("symbol")
        .agg(
            pl.len().alias("history_days"),
            pl.col("date").max().alias("last_date"),
            pl.col("is_active").last().alias("last_active"),
            pl.col("close_raw").last().alias("last_close_raw"),
            pl.col("close_adj").last().alias("last_close_adj"),
            pl.col("adv_usd").median().alias("median_adv_usd"),
            pl.col("sector").last().alias("sector"),
        )
        .sort("symbol")
    )

    eligible: list[str] = []
    excluded: dict[str, tuple[str, ...]] = {}
    sector_map: dict[str, str] = {}

    for row in summary.iter_rows(named=True):
        sym = str(row["symbol"])
        reasons: list[str] = []

        if cfg.require_active and (
            not bool(row["last_active"]) or row["last_date"] != window_end_date
        ):
            reasons.append("inactive_or_delisted_at_formation_end")
        if int(row["history_days"]) < cfg.min_history_days:
            reasons.append(f"insufficient_history({row['history_days']}<{cfg.min_history_days})")
        if float(row["last_close_raw"]) < cfg.min_price:
            reasons.append(f"below_min_price({row['last_close_raw']:.2f}<{cfg.min_price:.2f})")
        if float(row["median_adv_usd"]) < cfg.min_adv_usd:
            reasons.append(f"below_min_adv({row['median_adv_usd']:.0f}<{cfg.min_adv_usd:.0f})")

        if reasons:
            excluded[sym] = tuple(reasons)
        else:
            eligible.append(sym)
            sector_map[sym] = str(row["sector"])

    return UniverseFilterResult(
        formation_start=formation_start,
        formation_end=formation_end,
        eligible_symbols=tuple(sorted(eligible)),
        excluded_reasons=excluded,
        sector_map=sector_map,
    )
