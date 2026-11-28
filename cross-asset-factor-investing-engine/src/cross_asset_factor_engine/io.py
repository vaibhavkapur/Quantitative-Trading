"""Isolated I/O adapters for loading canonical datasets and persisting versioned outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from cross_asset_factor_engine.contracts import (
    CONTRACT_METADATA_SCHEMA,
    FACTOR_INPUTS_SCHEMA,
    FX_RATES_SCHEMA,
    PRICES_SCHEMA,
    MarketDataset,
    validate_dataset,
)
from cross_asset_factor_engine.errors import SchemaValidationError


def _read_csv_with_schema(
    path: Path,
    schema: dict[str, pl.DataType],
    table_name: str,
) -> pl.DataFrame:
    if not path.exists():
        raise SchemaValidationError(
            f"Required input file not found for '{table_name}': {path}",
            details={"table": table_name, "path": str(path)},
        )

    # Read raw first to validate required columns with clean error messages before casting
    raw_df = pl.read_csv(path, try_parse_dates=False, infer_schema_length=0)
    missing = [col for col in schema if col not in raw_df.columns]
    if missing:
        raise SchemaValidationError(
            f"Table '{table_name}' ({path.name}) is missing required columns: {missing}",
            details={"table": table_name, "path": str(path), "missing_columns": missing},
        )

    exprs: list[pl.Expr] = []
    for col_name, dtype in schema.items():
        col_expr = pl.col(col_name).replace("", None)
        if dtype == pl.Date:
            exprs.append(col_expr.str.to_date("%Y-%m-%d", strict=True).alias(col_name))
        elif dtype == pl.Datetime("us", "UTC"):
            exprs.append(
                col_expr.str.to_datetime(time_zone="UTC", time_unit="us", strict=True).alias(col_name)
            )
        elif dtype == pl.Boolean:
            exprs.append(
                pl.when(col_expr.str.to_lowercase() == "true")
                .then(pl.lit(True))
                .when(col_expr.str.to_lowercase() == "false")
                .then(pl.lit(False))
                .otherwise(None)
                .cast(pl.Boolean)
                .alias(col_name)
            )
        elif dtype == pl.Float64:
            exprs.append(col_expr.cast(pl.Float64, strict=True).alias(col_name))
        elif dtype == pl.Int64:
            exprs.append(col_expr.cast(pl.Int64, strict=True).alias(col_name))
        else:
            exprs.append(col_expr.cast(pl.String).alias(col_name))

    try:
        return raw_df.select(exprs)
    except pl.exceptions.ComputeError as exc:
        raise SchemaValidationError(
            f"Type parsing error in table '{table_name}' ({path.name}): {exc}",
            details={"table": table_name, "path": str(path)},
        ) from exc


def load_dataset_from_dir(
    dataset_dir: str | Path,
    *,
    base_currency: str = "USD",
) -> MarketDataset:
    """Load and validate all canonical tables from a directory."""
    root = Path(dataset_dir)
    if not root.exists() or not root.is_dir():
        raise SchemaValidationError(
            f"Dataset directory does not exist: {root}",
            details={"dataset_dir": str(root)},
        )

    meta = _read_csv_with_schema(root / "contract_metadata.csv", CONTRACT_METADATA_SCHEMA, "contract_metadata")
    prices = _read_csv_with_schema(root / "prices.csv", PRICES_SCHEMA, "prices")
    fx_rates = _read_csv_with_schema(root / "fx_rates.csv", FX_RATES_SCHEMA, "fx_rates")
    factor_inputs = _read_csv_with_schema(root / "factor_inputs.csv", FACTOR_INPUTS_SCHEMA, "factor_inputs")

    provenance = "synthetic_fixture"
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        provenance = str(manifest.get("provenance", provenance))

    return validate_dataset(
        meta,
        prices,
        fx_rates,
        factor_inputs,
        base_currency=base_currency,
        source_provenance=provenance,
    )


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def write_csv(path: str | Path, df: pl.DataFrame) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out)
    return out
