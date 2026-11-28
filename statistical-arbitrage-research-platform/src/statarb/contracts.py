"""Canonical data contracts, schema validation, and deterministic input hashing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from statarb.exceptions import (
    ContractValidationError,
    MissingDataError,
    UnsupportedInputError,
)

CONTRACT_VERSION = "1.0.0"

PRICE_REQUIRED_COLUMNS: dict[str, pl.DataType] = {
    "date": pl.Date,
    "symbol": pl.String,
    "close_adj": pl.Float64,
    "close_raw": pl.Float64,
    "volume": pl.Float64,
    "adv_usd": pl.Float64,
    "market_cap_usd": pl.Float64,
    "sector": pl.String,
    "is_active": pl.Boolean,
    "split_factor": pl.Float64,
    "cash_dividend": pl.Float64,
}

BORROW_REQUIRED_COLUMNS: dict[str, pl.DataType] = {
    "date": pl.Date,
    "symbol": pl.String,
    "borrow_available": pl.Boolean,
    "borrow_fee_bps_annual": pl.Float64,
    "max_short_notional": pl.Float64,
}


@dataclass(frozen=True)
class ValidationReport:
    contract_version: str
    prices_rows: int
    borrow_rows: int
    symbols: tuple[str, ...]
    dropped_symbols: tuple[str, ...]
    date_start: str
    date_end: str
    corporate_action_events: int
    prices_hash: str
    borrow_hash: str


def read_table(path: str | Path) -> pl.DataFrame:
    """Read a Parquet or CSV table into a Polars DataFrame."""
    file_path = Path(path)
    if not file_path.exists():
        raise MissingDataError(f"Input dataset not found: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(file_path)
    if suffix == ".csv":
        return pl.read_csv(file_path, try_parse_dates=True)
    raise UnsupportedInputError(
        f"Unsupported dataset format '{suffix}' for '{file_path}'. Supported: .parquet, .csv"
    )


def _cast_required_columns(
    df: pl.DataFrame,
    required: dict[str, pl.DataType],
    table_name: str,
) -> pl.DataFrame:
    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        raise ContractValidationError(f"{table_name} missing required columns: {missing_cols}")
    if df.height == 0:
        raise ContractValidationError(f"{table_name} is empty (0 rows)")

    cast_exprs = []
    for col, dtype in required.items():
        if dtype == pl.Date and df.schema[col] == pl.String:
            cast_exprs.append(pl.col(col).str.to_date(strict=True).alias(col))
        else:
            cast_exprs.append(pl.col(col).cast(dtype, strict=True).alias(col))
    try:
        out = df.select(cast_exprs)
    except Exception as exc:
        raise ContractValidationError(f"{table_name} failed schema type casting: {exc}") from exc
    return out


def compute_frame_hash(df: pl.DataFrame, sort_cols: list[str]) -> str:
    """Compute a deterministic SHA-256 hash (first 16 hex chars) of a Polars DataFrame."""
    ordered = df.sort(sort_cols)
    csv_bytes = ordered.write_csv().encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()[:16]


def validate_prices(
    df: pl.DataFrame,
    *,
    max_missing_fraction: float = 0.02,
    missing_price_policy: str = "raise",
    check_corporate_actions: bool = True,
) -> tuple[pl.DataFrame, list[str], int]:
    """Validate canonical price bars and corporate action consistency.

    Returns (validated_sorted_df, dropped_symbols, corporate_action_event_count).
    """
    typed = _cast_required_columns(df, PRICE_REQUIRED_COLUMNS, "prices")

    # Key columns must never be null
    for key_col in ("date", "symbol", "is_active", "sector"):
        if typed[key_col].null_count() > 0:
            raise ContractValidationError(
                f"prices column '{key_col}' contains {typed[key_col].null_count()} null values"
            )

    # Check duplicate (date, symbol)
    dup_count = typed.height - typed.select(["date", "symbol"]).unique().height
    if dup_count > 0:
        raise ContractValidationError(f"prices contains {dup_count} duplicate (date, symbol) rows")

    # Check null or non-finite prices
    null_or_nan = typed.filter(
        pl.col("close_adj").is_null()
        | pl.col("close_adj").is_nan()
        | pl.col("close_adj").is_infinite()
        | pl.col("close_raw").is_null()
        | pl.col("close_raw").is_nan()
        | pl.col("close_raw").is_infinite()
    )
    dropped_symbols: list[str] = []
    if null_or_nan.height > 0:
        bad_syms = sorted(null_or_nan["symbol"].unique().to_list())
        if missing_price_policy == "raise":
            raise MissingDataError(
                f"prices contains {null_or_nan.height} null/non-finite price bars "
                f"for symbols: {bad_syms}"
            )
        if missing_price_policy == "drop_symbol":
            dropped_symbols.extend(bad_syms)
            typed = typed.filter(~pl.col("symbol").is_in(bad_syms))
        else:
            raise UnsupportedInputError(
                f"Unsupported missing_price_policy: '{missing_price_policy}'"
            )

    # Check other numeric columns for nulls / domain bounds
    for col in ("volume", "adv_usd", "market_cap_usd", "split_factor", "cash_dividend"):
        if typed[col].null_count() > 0 or typed.filter(pl.col(col).is_nan()).height > 0:
            raise MissingDataError(f"prices column '{col}' contains null or NaN values")

    if typed.filter((pl.col("close_adj") <= 0.0) | (pl.col("close_raw") <= 0.0)).height > 0:
        raise ContractValidationError("prices close_adj and close_raw must be strictly positive")

    if typed.filter(pl.col("split_factor") <= 0.0).height > 0:
        raise ContractValidationError("prices split_factor must be strictly positive")

    for nonneg_col in ("volume", "adv_usd", "market_cap_usd", "cash_dividend"):
        if typed.filter(pl.col(nonneg_col) < 0.0).height > 0:
            raise ContractValidationError(f"prices column '{nonneg_col}' must be >= 0")

    # Check active-symbol date coverage against expected trading calendar
    all_dates = typed.select("date").unique().sort("date")
    total_dates = all_dates.height
    sym_stats = (
        typed.filter(pl.col("is_active"))
        .group_by("symbol")
        .agg(
            pl.col("date").min().alias("first_date"),
            pl.col("date").max().alias("last_date"),
            pl.len().alias("observed_days"),
        )
    )
    date_list = all_dates["date"].to_list()
    date_index = {d: i for i, d in enumerate(date_list)}

    gap_symbols: list[str] = []
    for row in sym_stats.iter_rows(named=True):
        expected_days = date_index[row["last_date"]] - date_index[row["first_date"]] + 1
        if expected_days > 0:
            missing_frac = 1.0 - (row["observed_days"] / expected_days)
            if missing_frac > max_missing_fraction + 1e-12:
                gap_symbols.append(row["symbol"])

    if gap_symbols:
        gap_symbols = sorted(gap_symbols)
        if missing_price_policy == "raise":
            raise MissingDataError(
                f"Active symbols exceed max_missing_fraction ({max_missing_fraction}): "
                f"{gap_symbols} (calendar days={total_dates})"
            )
        dropped_symbols.extend(gap_symbols)
        typed = typed.filter(~pl.col("symbol").is_in(gap_symbols))

    if typed.height == 0:
        raise MissingDataError("No valid price rows remain after applying missing_price_policy")

    typed = typed.sort(["symbol", "date"])

    # Corporate action consistency check: when split_factor != 1.0 on day t,
    # raw_ratio = close_raw[t] / close_raw[t-1] should reflect split_factor
    # while adj_ratio = close_adj[t] / close_adj[t-1] should NOT contain the split jump.
    # Specifically, (close_raw[t] * split_factor[t] / close_raw[t-1]) should approximately
    # track close_adj[t] / close_adj[t-1] (up to dividend adjustment).
    ca_events_df = typed.filter((pl.col("split_factor") != 1.0) | (pl.col("cash_dividend") > 0.0))
    ca_event_count = ca_events_df.height

    if check_corporate_actions and ca_event_count > 0:
        with_prev = typed.with_columns(
            pl.col("close_adj").shift(1).over("symbol").alias("prev_adj"),
            pl.col("close_raw").shift(1).over("symbol").alias("prev_raw"),
        )
        split_rows = with_prev.filter(
            (pl.col("split_factor") != 1.0)
            & pl.col("prev_adj").is_not_null()
            & pl.col("prev_raw").is_not_null()
        )
        for row in split_rows.iter_rows(named=True):
            raw_split_adj_ret = (
                row["close_raw"] * row["split_factor"] + row["cash_dividend"]
            ) / row["prev_raw"]
            adj_ret = row["close_adj"] / row["prev_adj"]
            # Allow modest tolerance for dividend timing conventions, but catch unadjusted splits
            if abs(adj_ret - raw_split_adj_ret) / max(abs(raw_split_adj_ret), 1e-9) > 0.05:
                raise ContractValidationError(
                    f"Corporate action inconsistency for {row['symbol']} on {row['date']}: "
                    f"split_factor={row['split_factor']} implies split-adjusted return "
                    f"{raw_split_adj_ret:.4f}, but close_adj ratio is {adj_ret:.4f}. "
                    "Ensure close_adj is backward split-adjusted."
                )

    return typed.sort(["date", "symbol"]), sorted(set(dropped_symbols)), ca_event_count


def validate_borrow(df: pl.DataFrame) -> pl.DataFrame:
    """Validate canonical borrow availability and fee schedule."""
    typed = _cast_required_columns(df, BORROW_REQUIRED_COLUMNS, "borrow")

    for col in BORROW_REQUIRED_COLUMNS:
        if typed[col].null_count() > 0:
            raise ContractValidationError(
                f"borrow column '{col}' contains {typed[col].null_count()} null values"
            )

    dup_count = typed.height - typed.select(["date", "symbol"]).unique().height
    if dup_count > 0:
        raise ContractValidationError(f"borrow contains {dup_count} duplicate (date, symbol) rows")

    if (
        typed.filter(
            pl.col("borrow_fee_bps_annual").is_nan() | (pl.col("borrow_fee_bps_annual") < 0.0)
        ).height
        > 0
    ):
        raise ContractValidationError("borrow_fee_bps_annual must be finite and >= 0")

    if (
        typed.filter(
            pl.col("max_short_notional").is_nan() | (pl.col("max_short_notional") < 0.0)
        ).height
        > 0
    ):
        raise ContractValidationError("max_short_notional must be finite and >= 0")

    return typed.sort(["date", "symbol"])


def load_and_validate_inputs(
    prices_path: str | Path,
    borrow_path: str | Path,
    *,
    max_missing_fraction: float = 0.02,
    missing_price_policy: str = "raise",
) -> tuple[pl.DataFrame, pl.DataFrame, ValidationReport]:
    """Load and validate both canonical input datasets and return a ValidationReport."""
    raw_prices = read_table(prices_path)
    raw_borrow = read_table(borrow_path)

    prices, dropped_symbols, ca_count = validate_prices(
        raw_prices,
        max_missing_fraction=max_missing_fraction,
        missing_price_policy=missing_price_policy,
    )
    borrow = validate_borrow(raw_borrow)

    symbols = tuple(sorted(prices["symbol"].unique().to_list()))
    dates = prices["date"].unique().sort()
    report = ValidationReport(
        contract_version=CONTRACT_VERSION,
        prices_rows=prices.height,
        borrow_rows=borrow.height,
        symbols=symbols,
        dropped_symbols=tuple(dropped_symbols),
        date_start=str(dates[0]),
        date_end=str(dates[-1]),
        corporate_action_events=ca_count,
        prices_hash=compute_frame_hash(prices, ["date", "symbol"]),
        borrow_hash=compute_frame_hash(borrow, ["date", "symbol"]),
    )
    return prices, borrow, report
