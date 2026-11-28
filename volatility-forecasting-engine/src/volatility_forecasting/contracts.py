"""Canonical data contracts, Polars readers, and strict schema validation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl

from volatility_forecasting.errors import DataContractError, NegativeForecastError

CONTRACT_VERSION = "1.0.0"

REQUIRED_RETURNS_COLUMNS = ("asset_id", "timestamp", "log_return")
OPTIONAL_RETURNS_COLUMNS = ("overnight_return", "session_date", "close_price")

REQUIRED_INTRADAY_COLUMNS = (
    "asset_id",
    "timestamp",
    "session_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
OPTIONAL_INTRADAY_COLUMNS = ("is_missing",)

REQUIRED_LABELS_COLUMNS = (
    "asset_id",
    "timestamp",
    "horizon_days",
    "realized_variance",
    "realized_vol",
)


def compute_file_sha256(path: str | Path) -> str:
    """Compute SHA-256 hex digest for a file to support input versioning."""
    p = Path(path)
    if not p.exists():
        raise DataContractError(f"Input file does not exist: {p}")
    hasher = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _check_required_columns(df: pl.DataFrame, required: Iterable[str], table_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataContractError(
            f"Table '{table_name}' is missing required columns: {missing}. "
            f"Found columns: {df.columns}."
        )


def _parse_timestamp_col(df: pl.DataFrame, col: str, table_name: str) -> pl.DataFrame:
    dtype = df.schema[col]
    if dtype == pl.Utf8 or dtype == pl.String:
        try:
            df = df.with_columns(
                pl.col(col).str.to_datetime(strict=True, time_zone=None).alias(col)
            )
        except Exception:
            try:
                df = df.with_columns(
                    pl.col(col)
                    .str.to_date(strict=True)
                    .cast(pl.Datetime("us"))
                    .alias(col)
                )
            except Exception as exc:
                raise DataContractError(
                    f"Table '{table_name}' column '{col}' could not be parsed as ISO-8601 "
                    f"timestamp or date: {exc}"
                ) from exc
    elif dtype == pl.Date:
        df = df.with_columns(pl.col(col).cast(pl.Datetime("us")).alias(col))
    elif isinstance(dtype, pl.Datetime):
        # Normalize to timezone-naive UTC-equivalent microseconds if timezone-aware
        if dtype.time_zone is not None:
            df = df.with_columns(
                pl.col(col).dt.convert_time_zone("UTC").dt.replace_time_zone(None).alias(col)
            )
        df = df.with_columns(pl.col(col).cast(pl.Datetime("us")).alias(col))
    else:
        raise DataContractError(
            f"Table '{table_name}' column '{col}' has unsupported timestamp dtype: {dtype}."
        )
    return df


def _parse_date_col(df: pl.DataFrame, col: str, table_name: str) -> pl.DataFrame:
    dtype = df.schema[col]
    if dtype == pl.Utf8 or dtype == pl.String:
        try:
            return df.with_columns(pl.col(col).str.to_date(strict=True).alias(col))
        except Exception as exc:
            raise DataContractError(
                f"Table '{table_name}' column '{col}' could not be parsed as ISO-8601 date: {exc}"
            ) from exc
    if dtype == pl.Date:
        return df
    if isinstance(dtype, pl.Datetime):
        return df.with_columns(pl.col(col).dt.date().alias(col))
    raise DataContractError(
        f"Table '{table_name}' column '{col}' has unsupported date dtype: {dtype}."
    )


def validate_returns_df(df: pl.DataFrame) -> pl.DataFrame:
    """Validate and normalize canonical timestamped returns table."""
    if df.height == 0:
        raise DataContractError("Returns table must not be empty.")
    _check_required_columns(df, REQUIRED_RETURNS_COLUMNS, "returns")

    out = _parse_timestamp_col(df, "timestamp", "returns")
    out = out.with_columns(
        pl.col("asset_id").cast(pl.String),
        pl.col("log_return").cast(pl.Float64),
    )

    if "session_date" in out.columns:
        out = _parse_date_col(out, "session_date", "returns")
    else:
        out = out.with_columns(pl.col("timestamp").dt.date().alias("session_date"))

    if "overnight_return" in out.columns:
        out = out.with_columns(pl.col("overnight_return").cast(pl.Float64))
    else:
        out = out.with_columns(pl.lit(0.0, dtype=pl.Float64).alias("overnight_return"))

    if out.select(pl.col("asset_id").is_null().any()).item():
        raise DataContractError("Returns table contains null asset_id values.")
    if out.select(pl.col("timestamp").is_null().any()).item():
        raise DataContractError("Returns table contains null timestamp values.")
    if out.select(pl.col("log_return").is_null().any()).item():
        raise DataContractError("Returns table contains null log_return values.")

    log_ret_np = out["log_return"].to_numpy()
    if not np.all(np.isfinite(log_ret_np)):
        raise DataContractError("Returns table contains NaN or infinite log_return values.")

    overnight_np = out["overnight_return"].fill_null(0.0).to_numpy()
    if not np.all(np.isfinite(overnight_np)):
        raise DataContractError("Returns table contains NaN or infinite overnight_return values.")
    out = out.with_columns(pl.col("overnight_return").fill_null(0.0))

    # Check uniqueness per (asset_id, timestamp) and (asset_id, session_date)
    if out.select(["asset_id", "timestamp"]).n_unique() != out.height:
        raise DataContractError("Returns table contains duplicate (asset_id, timestamp) rows.")
    if out.select(["asset_id", "session_date"]).n_unique() != out.height:
        raise DataContractError("Returns table contains duplicate (asset_id, session_date) rows.")

    # Sort chronologically per asset
    out = out.sort(["asset_id", "timestamp"])
    return out


def validate_intraday_bars_df(df: pl.DataFrame) -> pl.DataFrame:
    """Validate and normalize canonical intraday OHLCV bars table."""
    if df.height == 0:
        raise DataContractError("Intraday bars table must not be empty.")
    _check_required_columns(df, REQUIRED_INTRADAY_COLUMNS, "intraday_bars")

    out = _parse_timestamp_col(df, "timestamp", "intraday_bars")
    out = _parse_date_col(out, "session_date", "intraday_bars")
    out = out.with_columns(
        pl.col("asset_id").cast(pl.String),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
    )

    if "is_missing" in out.columns:
        out = out.with_columns(pl.col("is_missing").cast(pl.Boolean).fill_null(False))
    else:
        out = out.with_columns(pl.lit(False).alias("is_missing"))

    if out.select(pl.col("asset_id").is_null().any()).item():
        raise DataContractError("Intraday bars table contains null asset_id values.")
    if out.select(pl.col("timestamp").is_null().any()).item():
        raise DataContractError("Intraday bars table contains null timestamp values.")

    if out.select(["asset_id", "timestamp"]).n_unique() != out.height:
        raise DataContractError(
            "Intraday bars table contains duplicate (asset_id, timestamp) rows."
        )

    # Validate non-missing rows have strictly positive finite OHLC and high >= max(open, close) >= low
    present = out.filter(~pl.col("is_missing"))
    if present.height > 0:
        for col in ("open", "high", "low", "close"):
            vals = present[col].to_numpy()
            if np.any(np.isnan(vals)) or np.any(~np.isfinite(vals)) or np.any(vals <= 0.0):
                raise DataContractError(
                    f"Intraday bars column '{col}' contains non-positive, NaN, or infinite values "
                    "on non-missing rows."
                )
        bad_hl = present.filter(
            (pl.col("high") < pl.col("low"))
            | (pl.col("high") < pl.col("open"))
            | (pl.col("high") < pl.col("close"))
            | (pl.col("low") > pl.col("open"))
            | (pl.col("low") > pl.col("close"))
        )
        if bad_hl.height > 0:
            raise DataContractError(
                f"Intraday bars contain {bad_hl.height} OHLC ordering violations "
                "(high < low or open/close outside [low, high])."
            )
        vol_vals = present["volume"].to_numpy()
        if np.any(np.isnan(vol_vals)) or np.any(~np.isfinite(vol_vals)) or np.any(vol_vals < 0.0):
            raise DataContractError(
                "Intraday bars column 'volume' contains negative, NaN, or infinite values."
            )

    return out.sort(["asset_id", "session_date", "timestamp"])


def validate_labels_df(df: pl.DataFrame) -> pl.DataFrame:
    """Validate optional precomputed realized volatility labels table."""
    if df.height == 0:
        raise DataContractError("Realized volatility labels table must not be empty.")
    _check_required_columns(df, REQUIRED_LABELS_COLUMNS, "realized_vol_labels")

    out = _parse_timestamp_col(df, "timestamp", "realized_vol_labels")
    out = out.with_columns(
        pl.col("asset_id").cast(pl.String),
        pl.col("horizon_days").cast(pl.Int64),
        pl.col("realized_variance").cast(pl.Float64),
        pl.col("realized_vol").cast(pl.Float64),
    )

    for col in ("realized_variance", "realized_vol"):
        vals = out[col].to_numpy()
        if np.any(np.isnan(vals)) or np.any(~np.isfinite(vals)) or np.any(vals < 0.0):
            raise DataContractError(
                f"Labels table column '{col}' must be finite and nonnegative."
            )
    if out.filter(pl.col("horizon_days") <= 0).height > 0:
        raise DataContractError("Labels table 'horizon_days' must be strictly positive.")

    if out.select(["asset_id", "timestamp", "horizon_days"]).n_unique() != out.height:
        raise DataContractError(
            "Labels table contains duplicate (asset_id, timestamp, horizon_days) rows."
        )

    return out.sort(["asset_id", "horizon_days", "timestamp"])


def validate_nonnegative_forecasts(
    values: np.ndarray,
    model_name: str,
    quantity: str = "variance",
) -> np.ndarray:
    """Enforce finite, strictly nonnegative forecast output contract."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise DataContractError(
            f"Forecast array for '{model_name}' must be 1D, got shape {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        bad_idx = np.where(~np.isfinite(arr))[0][:5].tolist()
        raise NegativeForecastError(
            f"Model '{model_name}' emitted non-finite {quantity} forecasts at indices {bad_idx}."
        )
    if np.any(arr < 0.0):
        neg_idx = np.where(arr < 0.0)[0]
        min_val = float(np.min(arr[neg_idx]))
        raise NegativeForecastError(
            f"Model '{model_name}' emitted {len(neg_idx)} negative {quantity} forecasts "
            f"(minimum={min_val:.6e} at indices {neg_idx[:5].tolist()}). "
            "Negative volatility/variance forecasts are explicitly rejected."
        )
    return arr


def read_table(path: str | Path) -> pl.DataFrame:
    """Read CSV or Parquet input table with explicit error propagation."""
    p = Path(path)
    if not p.exists():
        raise DataContractError(f"Data file not found: {p}")
    suffix = p.suffix.lower()
    try:
        if suffix == ".csv":
            return pl.read_csv(p)
        if suffix == ".parquet":
            return pl.read_parquet(p)
    except Exception as exc:
        raise DataContractError(f"Failed to read {p}: {exc}") from exc
    raise DataContractError(
        f"Unsupported input file extension '{suffix}' for {p}. Supported: .csv, .parquet."
    )
