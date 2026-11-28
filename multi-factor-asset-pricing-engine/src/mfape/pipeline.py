"""Point-in-time excess-return and factor-alignment pipeline using Polars."""

from __future__ import annotations

import datetime as dt
from typing import Sequence

import numpy as np
import polars as pl

from mfape.errors import (
    DataContractError,
    DateLeakageError,
    InsufficientObservationsError,
    MissingRiskFreeError,
    PeriodAlignmentError,
    UniverseChangeError,
)
from mfape.schemas import AlignedPanel, DiagnosticCounters, SplitConfig

_ISO_DATE_FMT = "%Y-%m-%d"

# Expected day gaps between consecutive period dates for frequency validation
_FREQ_GAP_BOUNDS: dict[str, tuple[int, int]] = {
    "D": (1, 10),   # daily/business-day series accounting for weekends/holidays
    "W": (5, 10),   # weekly series
    "M": (27, 32),  # monthly series (28-31 days; +/- 1 for weekend shift)
    "Q": (88, 94),  # quarterly series
}


def _parse_iso_dates(date_strings: Sequence[str], table_name: str) -> list[dt.date]:
    parsed: list[dt.date] = []
    for raw in date_strings:
        if not isinstance(raw, str):
            raise DataContractError(
                f"Table '{table_name}' contains non-string date value: {raw!r}.",
                {"table": table_name, "value": repr(raw)},
            )
        try:
            parsed.append(dt.datetime.strptime(raw, _ISO_DATE_FMT).date())
        except ValueError as exc:
            raise PeriodAlignmentError(
                f"Table '{table_name}' contains invalid ISO date '{raw}'. Expected YYYY-MM-DD.",
                {"table": table_name, "invalid_date": raw},
            ) from exc
    return parsed


def _validate_frequency_cadence(
    sorted_dates: list[str],
    frequency: str,
    table_name: str,
) -> None:
    """Verify date monotonicity, period uniqueness, and consistent period cadence."""
    parsed = _parse_iso_dates(sorted_dates, table_name)
    if len(parsed) < 2:
        return

    # For monthly and quarterly data, ensure at most one observation per calendar (year, period)
    if frequency == "M":
        ym = [(d.year, d.month) for d in parsed]
        if len(set(ym)) != len(ym):
            raise PeriodAlignmentError(
                f"Table '{table_name}' contains multiple observations in the same calendar month "
                f"under frequency='M'.",
                {"table": table_name, "frequency": frequency},
            )
    elif frequency == "Q":
        yq = [(d.year, (d.month - 1) // 3 + 1) for d in parsed]
        if len(set(yq)) != len(yq):
            raise PeriodAlignmentError(
                f"Table '{table_name}' contains multiple observations in the same calendar quarter "
                f"under frequency='Q'.",
                {"table": table_name, "frequency": frequency},
            )

    min_gap, max_gap = _FREQ_GAP_BOUNDS[frequency]
    bad_gaps: list[dict[str, object]] = []
    for prev_s, curr_s, prev_d, curr_d in zip(sorted_dates, sorted_dates[1:], parsed, parsed[1:]):
        gap = (curr_d - prev_d).days
        if gap < min_gap or gap > max_gap:
            bad_gaps.append({"from": prev_s, "to": curr_s, "gap_days": gap})
    if bad_gaps:
        raise PeriodAlignmentError(
            f"Table '{table_name}' violates expected period cadence for frequency='{frequency}' "
            f"(expected {min_gap}-{max_gap} days between consecutive periods); "
            f"found {len(bad_gaps)} misaligned transition(s) starting with {bad_gaps[0]}.",
            {
                "table": table_name,
                "frequency": frequency,
                "expected_gap_days": [min_gap, max_gap],
                "violations": bad_gaps[:5],
            },
        )


def _require_columns(df: pl.DataFrame, required: Sequence[str], table_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataContractError(
            f"Table '{table_name}' is missing required column(s): {missing}.",
            {"table": table_name, "missing_columns": missing, "available_columns": df.columns},
        )


def _check_finite_numeric(df: pl.DataFrame, cols: Sequence[str], table_name: str) -> pl.DataFrame:
    exprs = [pl.col(c).cast(pl.Float64) for c in cols]
    try:
        casted = df.with_columns(exprs)
    except Exception as exc:
        raise DataContractError(
            f"Table '{table_name}' failed numeric float64 casting for columns {list(cols)}: {exc}",
            {"table": table_name, "columns": list(cols)},
        ) from exc

    for col in cols:
        bad_nulls = casted.filter(pl.col(col).is_null() | pl.col(col).is_nan() | pl.col(col).is_infinite())
        if bad_nulls.height > 0:
            sample = bad_nulls.select(["date", col] if "date" in casted.columns else [col]).head(5).to_dicts()
            raise DataContractError(
                f"Table '{table_name}' has {bad_nulls.height} null/NaN/infinite value(s) in column '{col}'.",
                {"table": table_name, "column": col, "bad_count": bad_nulls.height, "sample": sample},
            )
    return casted


def _check_unique_keys(df: pl.DataFrame, keys: Sequence[str], table_name: str) -> None:
    dup_count = df.height - df.unique(subset=list(keys)).height
    if dup_count > 0:
        raise DataContractError(
            f"Table '{table_name}' contains {dup_count} duplicate key row(s) on {list(keys)}.",
            {"table": table_name, "keys": list(keys), "duplicate_count": dup_count},
        )


def build_aligned_panel(
    asset_returns_df: pl.DataFrame,
    factor_returns_df: pl.DataFrame,
    risk_free_df: pl.DataFrame,
    memberships_df: pl.DataFrame,
    required_factors: Sequence[str],
    frequency: str = "M",
    universe_policy: str = "pit_active",
    rf_missing_policy: str = "error",
    period_alignment_policy: str = "strict",
    diagnostics: DiagnosticCounters | None = None,
) -> AlignedPanel:
    """Validate contracts and align asset returns, memberships, risk-free rates, and factors.

    Computes point-in-time excess returns (total_return - rf) with zero lookahead.
    """
    diag = diagnostics if diagnostics is not None else DiagnosticCounters()
    diag.raw_asset_rows = asset_returns_df.height
    diag.raw_factor_rows = factor_returns_df.height
    diag.raw_rf_rows = risk_free_df.height
    diag.raw_membership_rows = memberships_df.height

    # 1. Validate schemas and key uniqueness
    _require_columns(asset_returns_df, ["date", "asset_id", "total_return"], "asset_returns")
    _require_columns(factor_returns_df, ["date", *required_factors], "factor_returns")
    _require_columns(risk_free_df, ["date", "rf"], "risk_free")
    _require_columns(memberships_df, ["date", "asset_id", "in_universe"], "portfolio_memberships")

    assets_clean = asset_returns_df.select(
        pl.col("date").cast(pl.String),
        pl.col("asset_id").cast(pl.String),
        pl.col("total_return"),
    )
    if assets_clean.filter(pl.col("date").is_null() | pl.col("asset_id").is_null()).height > 0:
        raise DataContractError("Table 'asset_returns' contains null 'date' or 'asset_id'.")
    assets_clean = _check_finite_numeric(assets_clean, ["total_return"], "asset_returns")
    _check_unique_keys(assets_clean, ["date", "asset_id"], "asset_returns")

    factors_clean = factor_returns_df.select(
        [pl.col("date").cast(pl.String), *[pl.col(f) for f in required_factors]]
    )
    if factors_clean.filter(pl.col("date").is_null()).height > 0:
        raise DataContractError("Table 'factor_returns' contains null 'date'.")
    factors_clean = _check_finite_numeric(factors_clean, required_factors, "factor_returns")
    _check_unique_keys(factors_clean, ["date"], "factor_returns")

    rf_clean = risk_free_df.select(pl.col("date").cast(pl.String), pl.col("rf").cast(pl.Float64))
    if rf_clean.filter(pl.col("date").is_null()).height > 0:
        raise DataContractError("Table 'risk_free' contains null 'date'.")
    _check_unique_keys(rf_clean, ["date"], "risk_free")

    # Normalize in_universe to boolean (supports bool or 0/1 ints)
    try:
        mem_clean = memberships_df.select(
            pl.col("date").cast(pl.String),
            pl.col("asset_id").cast(pl.String),
            pl.col("in_universe").cast(pl.Boolean),
        )
    except Exception as exc:
        raise DataContractError(
            f"Table 'portfolio_memberships' failed boolean cast for 'in_universe': {exc}"
        ) from exc
    if mem_clean.filter(
        pl.col("date").is_null() | pl.col("asset_id").is_null() | pl.col("in_universe").is_null()
    ).height > 0:
        raise DataContractError(
            "Table 'portfolio_memberships' contains null 'date', 'asset_id', or 'in_universe'."
        )
    _check_unique_keys(mem_clean, ["date", "asset_id"], "portfolio_memberships")

    # 2. Point-in-time membership join: keep only active (date, asset_id) pairs
    joined_pit = assets_clean.join(mem_clean, on=["date", "asset_id"], how="left")
    active_assets_df = joined_pit.filter(pl.col("in_universe").fill_null(False))
    diag.dropped_inactive_membership_rows = assets_clean.height - active_assets_df.height
    if diag.dropped_inactive_membership_rows > 0:
        diag.record(
            "INFO",
            "membership_filter",
            f"Excluded {diag.dropped_inactive_membership_rows} non-active asset-period observations via point-in-time membership.",
            dropped_rows=diag.dropped_inactive_membership_rows,
        )

    if active_assets_df.height == 0:
        raise InsufficientObservationsError(
            "Zero active asset observations remain after point-in-time membership filtering."
        )

    # 3. Validate date parsing & frequency cadence on asset dates and factor dates
    asset_dates = sorted(active_assets_df["date"].unique().to_list())
    factor_dates = sorted(factors_clean["date"].unique().to_list())
    _parse_iso_dates(rf_clean["date"].to_list(), "risk_free")
    _validate_frequency_cadence(asset_dates, frequency, "asset_returns")
    _validate_frequency_cadence(factor_dates, frequency, "factor_returns")

    # 4. Period alignment between active asset returns and factor returns
    asset_date_set = set(asset_dates)
    factor_date_set = set(factor_dates)
    missing_in_factors = sorted(asset_date_set - factor_date_set)
    missing_in_assets = sorted(factor_date_set - asset_date_set)

    if missing_in_factors or missing_in_assets:
        if period_alignment_policy == "strict":
            raise PeriodAlignmentError(
                "Period alignment mismatch between active asset returns and factor returns under "
                "period_alignment_policy='strict'.",
                {
                    "dates_in_assets_not_factors": missing_in_factors[:10],
                    "dates_in_factors_not_assets": missing_in_assets[:10],
                    "count_missing_in_factors": len(missing_in_factors),
                    "count_missing_in_assets": len(missing_in_assets),
                },
            )
        aligned_dates = sorted(asset_date_set & factor_date_set)
        diag.dropped_unaligned_periods = len(missing_in_factors) + len(missing_in_assets)
        diag.record(
            "WARNING",
            "period_alignment",
            f"Dropped {diag.dropped_unaligned_periods} unaligned period(s) under policy='inner'.",
            dates_missing_in_factors=missing_in_factors[:10],
            dates_missing_in_assets=missing_in_assets[:10],
        )
    else:
        aligned_dates = asset_dates

    if not aligned_dates:
        raise PeriodAlignmentError("No overlapping dates found between asset returns and factor returns.")

    # 5. Risk-free alignment and missing-observation handling
    rf_valid_df = rf_clean.filter(
        pl.col("rf").is_not_null() & pl.col("rf").is_not_nan() & pl.col("rf").is_finite()
    )
    rf_valid_dates = set(rf_valid_df["date"].to_list())
    missing_rf_dates = [d for d in aligned_dates if d not in rf_valid_dates]

    if missing_rf_dates:
        if rf_missing_policy == "error":
            raise MissingRiskFreeError(
                f"Missing or non-finite risk-free observations for {len(missing_rf_dates)} "
                f"aligned period(s): {missing_rf_dates[:10]}.",
                {
                    "missing_rf_dates": missing_rf_dates,
                    "missing_count": len(missing_rf_dates),
                },
            )
        aligned_dates = [d for d in aligned_dates if d in rf_valid_dates]
        diag.dropped_missing_rf_periods = len(missing_rf_dates)
        diag.record(
            "WARNING",
            "risk_free_alignment",
            f"Dropped {len(missing_rf_dates)} period(s) with missing risk-free rate under rf_missing_policy='drop_period'.",
            dropped_dates=missing_rf_dates[:10],
        )

    if not aligned_dates:
        raise MissingRiskFreeError("No periods remain after removing missing risk-free observations.")

    # 6. Compute excess returns = total_return - rf in Polars
    aligned_date_df = pl.DataFrame({"date": aligned_dates})
    panel_long = (
        active_assets_df.join(aligned_date_df, on="date", how="inner")
        .join(rf_valid_df, on="date", how="inner")
        .with_columns((pl.col("total_return") - pl.col("rf")).alias("excess_return"))
        .sort(["date", "asset_id"])
    )

    # 7. Track point-in-time universe changes across consecutive aligned periods
    assets_by_date: dict[str, set[str]] = {}
    for row in panel_long.select(["date", "asset_id"]).iter_rows():
        assets_by_date.setdefault(row[0], set()).add(row[1])

    universe_changes: list[dict[str, object]] = []
    prev_universe: set[str] | None = None
    for d in aligned_dates:
        curr_universe = assets_by_date.get(d, set())
        if prev_universe is not None and curr_universe != prev_universe:
            entered = sorted(curr_universe - prev_universe)
            exited = sorted(prev_universe - curr_universe)
            diag.universe_entries += len(entered)
            diag.universe_exits += len(exited)
            diag.universe_change_periods += 1
            universe_changes.append(
                {
                    "date": d,
                    "entered": entered,
                    "exited": exited,
                    "active_count": len(curr_universe),
                }
            )
        prev_universe = curr_universe

    is_balanced = len(universe_changes) == 0 and all(
        len(assets_by_date.get(d, set())) == len(assets_by_date.get(aligned_dates[0], set()))
        for d in aligned_dates
    )

    if not is_balanced:
        if universe_policy == "strict_balanced":
            raise UniverseChangeError(
                f"Universe membership changed across {len(universe_changes)} period(s) under "
                f"universe_policy='strict_balanced'.",
                {
                    "universe_change_periods": len(universe_changes),
                    "changes": universe_changes[:10],
                },
            )
        if universe_policy == "min_obs_balanced":
            common_assets = set.intersection(*(assets_by_date[d] for d in aligned_dates))
            all_seen_assets = set.union(*(assets_by_date[d] for d in aligned_dates))
            excluded = sorted(all_seen_assets - common_assets)
            diag.assets_excluded_min_obs = len(excluded)
            diag.record(
                "WARNING",
                "universe_policy",
                f"Excluded {len(excluded)} asset(s) not present in all {len(aligned_dates)} periods under policy='min_obs_balanced'.",
                excluded_assets=excluded[:20],
            )
            if not common_assets:
                raise UniverseChangeError(
                    "No assets remain present across all periods under universe_policy='min_obs_balanced'."
                )
            panel_long = panel_long.filter(pl.col("asset_id").is_in(sorted(common_assets)))
            is_balanced = True
        else:
            diag.record(
                "INFO",
                "universe_policy",
                f"Retained unbalanced point-in-time panel with {len(universe_changes)} universe change period(s) "
                f"({diag.universe_entries} entries, {diag.universe_exits} exits).",
                universe_changes=universe_changes[:10],
            )

    # 8. Materialize deterministic sorted NumPy matrices (T, N), (T, K), (T,)
    sorted_assets = sorted(panel_long["asset_id"].unique().to_list())
    date_to_idx = {d: i for i, d in enumerate(aligned_dates)}
    asset_to_idx = {a: j for j, a in enumerate(sorted_assets)}

    n_periods = len(aligned_dates)
    n_assets = len(sorted_assets)
    excess_matrix = np.full((n_periods, n_assets), np.nan, dtype=np.float64)
    active_mask = np.zeros((n_periods, n_assets), dtype=bool)

    for d, asset_id, exc_ret in panel_long.select(["date", "asset_id", "excess_return"]).iter_rows():
        ti = date_to_idx[d]
        aj = asset_to_idx[asset_id]
        excess_matrix[ti, aj] = float(exc_ret)
        active_mask[ti, aj] = True

    factors_aligned = (
        aligned_date_df.join(factors_clean, on="date", how="inner").sort("date")
    )
    factor_matrix = factors_aligned.select(list(required_factors)).to_numpy().astype(np.float64)

    rf_aligned = aligned_date_df.join(rf_valid_df, on="date", how="inner").sort("date")
    rf_vector = rf_aligned["rf"].to_numpy().astype(np.float64)

    diag.aligned_rows = int(active_mask.sum())
    diag.aligned_periods = n_periods
    diag.aligned_assets = n_assets

    return AlignedPanel(
        dates=tuple(aligned_dates),
        assets=tuple(sorted_assets),
        factor_names=tuple(required_factors),
        excess_returns=excess_matrix,
        active_mask=active_mask,
        factor_returns=factor_matrix,
        risk_free_rates=rf_vector,
        frequency=frequency,
        is_balanced=is_balanced,
        universe_changes=tuple(universe_changes),
    )


def slice_panel(panel: AlignedPanel, start_date: str, end_date: str) -> AlignedPanel:
    """Slice an AlignedPanel inclusively between [start_date, end_date] with zero leakage."""
    indices = [i for i, d in enumerate(panel.dates) if start_date <= d <= end_date]
    if not indices:
        raise InsufficientObservationsError(
            f"No aligned periods fall within [{start_date}, {end_date}].",
            {
                "start_date": start_date,
                "end_date": end_date,
                "panel_start": panel.dates[0] if panel.dates else None,
                "panel_end": panel.dates[-1] if panel.dates else None,
            },
        )
    idx_arr = np.array(indices, dtype=np.int64)
    sub_dates = tuple(panel.dates[i] for i in indices)
    sub_excess = panel.excess_returns[idx_arr, :]
    sub_mask = panel.active_mask[idx_arr, :]
    sub_factors = panel.factor_returns[idx_arr, :]
    sub_rf = panel.risk_free_rates[idx_arr]

    # Retain only assets that have at least one active observation in this slice
    asset_has_obs = sub_mask.any(axis=0)
    kept_asset_indices = np.where(asset_has_obs)[0]
    if kept_asset_indices.size == 0:
        raise InsufficientObservationsError(
            f"No active assets in sliced window [{start_date}, {end_date}]."
        )

    sub_assets = tuple(panel.assets[j] for j in kept_asset_indices)
    sub_excess = sub_excess[:, kept_asset_indices]
    sub_mask = sub_mask[:, kept_asset_indices]
    sub_changes = tuple(
        c for c in panel.universe_changes if start_date <= str(c["date"]) <= end_date
    )
    is_balanced = bool(sub_mask.all())

    return AlignedPanel(
        dates=sub_dates,
        assets=sub_assets,
        factor_names=panel.factor_names,
        excess_returns=sub_excess,
        active_mask=sub_mask,
        factor_returns=sub_factors,
        risk_free_rates=sub_rf,
        frequency=panel.frequency,
        is_balanced=is_balanced,
        universe_changes=sub_changes,
    )


def split_panel(
    panel: AlignedPanel,
    split: SplitConfig | None,
) -> tuple[AlignedPanel, AlignedPanel | None]:
    """Split an AlignedPanel into chronological (train_panel, test_panel) with leakage guards."""
    if split is None:
        return panel, None
    split.validate()

    train_panel = slice_panel(panel, split.train_start, split.train_end)
    if split.test_start is None or split.test_end is None:
        return train_panel, None

    test_panel = slice_panel(panel, split.test_start, split.test_end)

    # Hard runtime invariant check: max(train_dates) < min(test_dates)
    if train_panel.dates[-1] >= test_panel.dates[0]:
        raise DateLeakageError(
            f"Date leakage detected between sliced train ({train_panel.dates[-1]}) "
            f"and test ({test_panel.dates[0]}) windows.",
            {"train_last_date": train_panel.dates[-1], "test_first_date": test_panel.dates[0]},
        )
    return train_panel, test_panel
