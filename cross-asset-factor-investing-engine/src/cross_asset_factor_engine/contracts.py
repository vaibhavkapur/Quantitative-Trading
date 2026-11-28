"""Canonical data contracts, Polars schemas, and strict input/output validators."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Final

import polars as pl

from cross_asset_factor_engine import SCHEMA_VERSION
from cross_asset_factor_engine.config import SUPPORTED_ASSET_CLASSES
from cross_asset_factor_engine.errors import (
    CurrencyConversionError,
    RollNormalizationError,
    SchemaValidationError,
    TimestampGateError,
    UnsupportedAssetError,
)

SUPPORTED_INSTRUMENT_TYPES: Final[tuple[str, ...]] = (
    "equity_index_future",
    "equity_spot",
    "fx_forward",
    "commodity_future",
)

SUPPORTED_CARRY_CONVENTIONS: Final[tuple[str, ...]] = (
    "dividend_minus_financing",
    "interest_rate_differential",
    "futures_term_structure_roll",
)

EXPECTED_CARRY_BY_CLASS: Final[dict[str, str]] = {
    "equity": "dividend_minus_financing",
    "fx": "interest_rate_differential",
    "commodity": "futures_term_structure_roll",
}

CONTRACT_METADATA_SCHEMA: Final[dict[str, pl.DataType]] = {
    "asset_id": pl.String,
    "asset_class": pl.String,
    "quote_currency": pl.String,
    "base_currency": pl.String,
    "instrument_type": pl.String,
    "carry_convention": pl.String,
    "contract_multiplier": pl.Float64,
    "tick_size": pl.Float64,
    "tc_bps": pl.Float64,
    "financing_spread_bps": pl.Float64,
    "roll_cost_bps": pl.Float64,
    "active_from": pl.Date,
    "active_to": pl.Date,
}

PRICES_SCHEMA: Final[dict[str, pl.DataType]] = {
    "date": pl.Date,
    "asset_id": pl.String,
    "contract_code": pl.String,
    "front_price": pl.Float64,
    "next_price": pl.Float64,
    "front_expiry_days": pl.Int64,
    "next_expiry_days": pl.Int64,
    "is_roll_date": pl.Boolean,
    "prev_front_price_new_contract": pl.Float64,
    "as_of_ts": pl.Datetime("us", "UTC"),
    "publication_ts": pl.Datetime("us", "UTC"),
}

FX_RATES_SCHEMA: Final[dict[str, pl.DataType]] = {
    "date": pl.Date,
    "currency": pl.String,
    "usd_per_unit": pl.Float64,
    "short_rate": pl.Float64,
    "as_of_ts": pl.Datetime("us", "UTC"),
    "publication_ts": pl.Datetime("us", "UTC"),
}

FACTOR_INPUTS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "date": pl.Date,
    "asset_id": pl.String,
    "dividend_yield": pl.Float64,
    "earnings_quality": pl.Float64,
    "foreign_short_rate": pl.Float64,
    "domestic_short_rate": pl.Float64,
    "convenience_yield_proxy": pl.Float64,
    "as_of_ts": pl.Datetime("us", "UTC"),
    "publication_ts": pl.Datetime("us", "UTC"),
}

NORMALIZED_RETURNS_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "asset_id",
    "asset_class",
    "quote_currency",
    "base_currency",
    "instrument_type",
    "carry_convention",
    "contract_code",
    "front_price",
    "next_price",
    "front_expiry_days",
    "next_expiry_days",
    "is_roll_date",
    "local_return",
    "fx_return_factor",
    "carry_return_component",
    "gross_excess_return_base",
    "roll_cost_drag",
    "financing_cost_drag",
    "net_return_before_tc",
    "implied_roll_yield",
    "as_of_ts",
    "publication_ts",
)

FACTOR_SIGNALS_COLUMNS: Final[tuple[str, ...]] = (
    "signal_date",
    "as_of_ts",
    "source_max_publication_ts",
    "asset_id",
    "asset_class",
    "factor_name",
    "raw_value",
    "z_score",
    "signal_status",
    "missing_reason",
    "provenance_hash",
)


@dataclass(frozen=True)
class MarketDataset:
    """Validated canonical market dataset passed into domain engines."""

    contract_metadata: pl.DataFrame
    prices: pl.DataFrame
    fx_rates: pl.DataFrame
    factor_inputs: pl.DataFrame
    base_currency: str = "USD"
    schema_version: str = SCHEMA_VERSION
    dataset_hash: str = ""
    source_provenance: str = "synthetic_fixture"


def _require_columns(df: pl.DataFrame, schema: dict[str, pl.DataType], table_name: str) -> None:
    missing = [col for col in schema if col not in df.columns]
    if missing:
        raise SchemaValidationError(
            f"Table '{table_name}' is missing required columns: {missing}",
            details={"table": table_name, "missing_columns": missing},
        )


def compute_dataset_hash(
    contract_metadata: pl.DataFrame,
    prices: pl.DataFrame,
    fx_rates: pl.DataFrame,
    factor_inputs: pl.DataFrame,
) -> str:
    """Deterministic SHA-256 fingerprint across all canonical input tables."""
    h = hashlib.sha256()
    tables = [
        ("contract_metadata", contract_metadata.sort("asset_id")),
        ("prices", prices.sort(["date", "asset_id"])),
        ("fx_rates", fx_rates.sort(["date", "currency"])),
        ("factor_inputs", factor_inputs.sort(["date", "asset_id", "publication_ts"])),
    ]
    for name, df in tables:
        h.update(name.encode("utf-8"))
        h.update(b"\n")
        h.update(df.write_csv().encode("utf-8"))
        h.update(b"\n---\n")
    return h.hexdigest()[:16]


def validate_contract_metadata(df: pl.DataFrame) -> pl.DataFrame:
    """Validate contract_metadata schema, enumerations, and numerical invariants."""
    _require_columns(df, CONTRACT_METADATA_SCHEMA, "contract_metadata")
    if df.height == 0:
        raise SchemaValidationError(
            "Table 'contract_metadata' must contain at least one asset.",
            details={"table": "contract_metadata"},
        )

    null_ids = df.filter(pl.col("asset_id").is_null() | (pl.col("asset_id") == "")).height
    if null_ids > 0:
        raise SchemaValidationError("contract_metadata contains null or empty asset_id.")

    if df["asset_id"].n_unique() != df.height:
        dups = df.group_by("asset_id").len().filter(pl.col("len") > 1)["asset_id"].to_list()
        raise SchemaValidationError(
            f"contract_metadata contains duplicate asset_id values: {dups}",
            details={"duplicate_asset_ids": dups},
        )

    bad_classes = (
        df.filter(~pl.col("asset_class").is_in(list(SUPPORTED_ASSET_CLASSES)))["asset_class"]
        .unique()
        .to_list()
    )
    if bad_classes:
        raise UnsupportedAssetError(
            f"Unsupported asset_class values: {bad_classes}. Supported: {list(SUPPORTED_ASSET_CLASSES)}",
            details={"unsupported_asset_classes": bad_classes},
        )

    bad_types = (
        df.filter(~pl.col("instrument_type").is_in(list(SUPPORTED_INSTRUMENT_TYPES)))["instrument_type"]
        .unique()
        .to_list()
    )
    if bad_types:
        raise UnsupportedAssetError(
            f"Unsupported instrument_type values: {bad_types}.",
            details={"unsupported_instrument_types": bad_types},
        )

    bad_carry = (
        df.filter(~pl.col("carry_convention").is_in(list(SUPPORTED_CARRY_CONVENTIONS)))["carry_convention"]
        .unique()
        .to_list()
    )
    if bad_carry:
        raise UnsupportedAssetError(
            f"Unsupported carry_convention values: {bad_carry}.",
            details={"unsupported_carry_conventions": bad_carry},
        )

    for row in df.iter_rows(named=True):
        expected_conv = EXPECTED_CARRY_BY_CLASS[row["asset_class"]]
        if row["carry_convention"] != expected_conv:
            raise UnsupportedAssetError(
                f"Asset '{row['asset_id']}' ({row['asset_class']}) has carry_convention "
                f"'{row['carry_convention']}', expected '{expected_conv}'.",
                details={"asset_id": row["asset_id"], "asset_class": row["asset_class"]},
            )
        qccy = row["quote_currency"]
        if not qccy or len(qccy) != 3 or not qccy.isupper():
            raise SchemaValidationError(
                f"Asset '{row['asset_id']}' has invalid quote_currency '{qccy}'."
            )
        if row["asset_class"] == "fx":
            bccy = row["base_currency"]
            if not bccy or len(bccy) != 3 or not bccy.isupper():
                raise SchemaValidationError(
                    f"FX asset '{row['asset_id']}' must specify a 3-letter uppercase base_currency."
                )

    bad_pos = df.filter(
        pl.col("contract_multiplier").is_null()
        | (pl.col("contract_multiplier") <= 0)
        | pl.col("tick_size").is_null()
        | (pl.col("tick_size") <= 0)
    )
    if bad_pos.height > 0:
        raise SchemaValidationError(
            "contract_multiplier and tick_size must be strictly positive.",
            details={"invalid_assets": bad_pos["asset_id"].to_list()},
        )

    bad_costs = df.filter(
        pl.col("tc_bps").is_null()
        | (pl.col("tc_bps") < 0)
        | pl.col("financing_spread_bps").is_null()
        | (pl.col("financing_spread_bps") < 0)
        | pl.col("roll_cost_bps").is_null()
        | (pl.col("roll_cost_bps") < 0)
    )
    if bad_costs.height > 0:
        raise SchemaValidationError(
            "Cost parameters (tc_bps, financing_spread_bps, roll_cost_bps) must be non-negative.",
            details={"invalid_assets": bad_costs["asset_id"].to_list()},
        )

    return df.select(list(CONTRACT_METADATA_SCHEMA.keys())).sort("asset_id")


def validate_prices(df: pl.DataFrame, metadata_df: pl.DataFrame) -> pl.DataFrame:
    """Validate point-in-time prices, positive quotes, timestamp ordering, and roll continuity."""
    _require_columns(df, PRICES_SCHEMA, "prices")
    if df.height == 0:
        raise SchemaValidationError("Table 'prices' must not be empty.", details={"table": "prices"})

    known_assets = set(metadata_df["asset_id"].to_list())
    unknown_assets = sorted(set(df["asset_id"].to_list()) - known_assets)
    if unknown_assets:
        raise SchemaValidationError(
            f"prices contains asset_id values not present in contract_metadata: {unknown_assets}",
            details={"unknown_assets": unknown_assets},
        )

    dup_count = df.group_by(["date", "asset_id"]).len().filter(pl.col("len") > 1).height
    if dup_count > 0:
        raise SchemaValidationError(
            f"prices contains {dup_count} duplicate (date, asset_id) keys.",
            details={"duplicate_key_count": dup_count},
        )

    bad_front = df.filter(
        pl.col("front_price").is_null()
        | pl.col("front_price").is_nan()
        | (pl.col("front_price") <= 0)
    )
    if bad_front.height > 0:
        raise SchemaValidationError(
            "prices contains null, NaN, or non-positive front_price values.",
            details={"bad_row_count": bad_front.height, "sample_assets": bad_front["asset_id"].head(5).to_list()},
        )

    bad_next = df.filter(
        pl.col("next_price").is_not_null() & (pl.col("next_price").is_nan() | (pl.col("next_price") <= 0))
    )
    if bad_next.height > 0:
        raise SchemaValidationError(
            "prices contains NaN or non-positive next_price values.",
            details={"bad_row_count": bad_next.height},
        )

    bad_expiry = df.filter(
        pl.col("front_expiry_days").is_null()
        | (pl.col("front_expiry_days") < 0)
        | (
            pl.col("next_expiry_days").is_not_null()
            & (pl.col("next_expiry_days") <= pl.col("front_expiry_days"))
        )
    )
    if bad_expiry.height > 0:
        raise SchemaValidationError(
            "Invalid expiry days: require front_expiry_days >= 0 and next_expiry_days > front_expiry_days.",
            details={"bad_row_count": bad_expiry.height},
        )

    bad_ts = df.filter(
        pl.col("as_of_ts").is_null()
        | pl.col("publication_ts").is_null()
        | (pl.col("publication_ts") < pl.col("as_of_ts"))
    )
    if bad_ts.height > 0:
        raise TimestampGateError(
            "prices contains rows with null timestamps or publication_ts < as_of_ts.",
            details={"bad_row_count": bad_ts.height},
        )

    ordered = df.select(list(PRICES_SCHEMA.keys())).sort(["asset_id", "date"])
    checked = ordered.with_columns(
        pl.col("contract_code").shift(1).over("asset_id").alias("_prev_contract_code")
    )
    bad_rolls = checked.filter(
        pl.col("_prev_contract_code").is_not_null()
        & (pl.col("contract_code") != pl.col("_prev_contract_code"))
        & (
            pl.col("prev_front_price_new_contract").is_null()
            | pl.col("prev_front_price_new_contract").is_nan()
            | (pl.col("prev_front_price_new_contract") <= 0)
        )
    )
    if bad_rolls.height > 0:
        sample = bad_rolls.select(["date", "asset_id", "_prev_contract_code", "contract_code"]).head(5).to_dicts()
        raise RollNormalizationError(
            "Contract code changed without a valid positive prev_front_price_new_contract.",
            details={"bad_roll_count": bad_rolls.height, "samples": sample},
        )

    return ordered


def validate_fx_rates(
    df: pl.DataFrame,
    metadata_df: pl.DataFrame,
    prices_df: pl.DataFrame,
    base_currency: str = "USD",
) -> pl.DataFrame:
    """Validate FX rates and guarantee full currency coverage across all price dates."""
    _require_columns(df, FX_RATES_SCHEMA, "fx_rates")
    if df.height == 0:
        raise SchemaValidationError("Table 'fx_rates' must not be empty.", details={"table": "fx_rates"})

    dup_count = df.group_by(["date", "currency"]).len().filter(pl.col("len") > 1).height
    if dup_count > 0:
        raise SchemaValidationError(
            f"fx_rates contains {dup_count} duplicate (date, currency) keys.",
            details={"duplicate_key_count": dup_count},
        )

    bad_fx = df.filter(
        pl.col("usd_per_unit").is_null()
        | pl.col("usd_per_unit").is_nan()
        | (pl.col("usd_per_unit") <= 0)
        | pl.col("short_rate").is_null()
        | pl.col("short_rate").is_nan()
    )
    if bad_fx.height > 0:
        raise CurrencyConversionError(
            "fx_rates contains invalid usd_per_unit or short_rate values.",
            details={"bad_row_count": bad_fx.height},
        )

    bad_ts = df.filter(
        pl.col("as_of_ts").is_null()
        | pl.col("publication_ts").is_null()
        | (pl.col("publication_ts") < pl.col("as_of_ts"))
    )
    if bad_ts.height > 0:
        raise TimestampGateError(
            "fx_rates contains rows with null timestamps or publication_ts < as_of_ts.",
            details={"bad_row_count": bad_ts.height},
        )

    required_ccys = set(metadata_df["quote_currency"].to_list())
    base_ccys = [c for c in metadata_df["base_currency"].to_list() if c is not None and c != ""]
    required_ccys.update(base_ccys)
    required_ccys.add(base_currency)

    available_ccys = set(df["currency"].unique().to_list())
    missing_ccys = sorted(required_ccys - available_ccys)
    if missing_ccys:
        raise CurrencyConversionError(
            f"fx_rates is missing required currencies: {missing_ccys}",
            details={"missing_currencies": missing_ccys},
        )

    price_dates = set(prices_df["date"].unique().to_list())
    for ccy in sorted(required_ccys):
        ccy_dates = set(df.filter(pl.col("currency") == ccy)["date"].to_list())
        missing_dates = sorted(price_dates - ccy_dates)
        if missing_dates:
            raise CurrencyConversionError(
                f"fx_rates for currency '{ccy}' is missing {len(missing_dates)} price dates "
                f"(first missing: {missing_dates[0]}).",
                details={"currency": ccy, "missing_date_count": len(missing_dates)},
            )

    if base_currency == "USD":
        bad_usd = df.filter((pl.col("currency") == "USD") & ((pl.col("usd_per_unit") - 1.0).abs() > 1e-12))
        if bad_usd.height > 0:
            raise CurrencyConversionError(
                "USD usd_per_unit must equal 1.0 when base_currency is USD.",
                details={"bad_row_count": bad_usd.height},
            )

    return df.select(list(FX_RATES_SCHEMA.keys())).sort(["currency", "date"])


def validate_factor_inputs(df: pl.DataFrame, metadata_df: pl.DataFrame) -> pl.DataFrame:
    """Validate point-in-time factor inputs schema and timestamp ordering."""
    _require_columns(df, FACTOR_INPUTS_SCHEMA, "factor_inputs")
    if df.height == 0:
        return df.select(list(FACTOR_INPUTS_SCHEMA.keys()))

    known_assets = set(metadata_df["asset_id"].to_list())
    unknown_assets = sorted(set(df["asset_id"].to_list()) - known_assets)
    if unknown_assets:
        raise SchemaValidationError(
            f"factor_inputs contains unknown asset_id values: {unknown_assets}",
            details={"unknown_assets": unknown_assets},
        )

    bad_ts = df.filter(
        pl.col("as_of_ts").is_null()
        | pl.col("publication_ts").is_null()
        | (pl.col("publication_ts") < pl.col("as_of_ts"))
    )
    if bad_ts.height > 0:
        raise TimestampGateError(
            "factor_inputs contains rows with null timestamps or publication_ts < as_of_ts.",
            details={"bad_row_count": bad_ts.height},
        )

    dup_count = (
        df.group_by(["date", "asset_id", "publication_ts"]).len().filter(pl.col("len") > 1).height
    )
    if dup_count > 0:
        raise SchemaValidationError(
            f"factor_inputs contains {dup_count} duplicate (date, asset_id, publication_ts) rows.",
            details={"duplicate_key_count": dup_count},
        )

    return df.select(list(FACTOR_INPUTS_SCHEMA.keys())).sort(["asset_id", "publication_ts", "date"])


def validate_dataset(
    contract_metadata: pl.DataFrame,
    prices: pl.DataFrame,
    fx_rates: pl.DataFrame,
    factor_inputs: pl.DataFrame,
    *,
    base_currency: str = "USD",
    source_provenance: str = "synthetic_fixture",
) -> MarketDataset:
    """Validate all four canonical tables and return an immutable MarketDataset."""
    meta = validate_contract_metadata(contract_metadata)
    px = validate_prices(prices, meta)
    fx = validate_fx_rates(fx_rates, meta, px, base_currency=base_currency)
    fi = validate_factor_inputs(factor_inputs, meta)
    ds_hash = compute_dataset_hash(meta, px, fx, fi)
    return MarketDataset(
        contract_metadata=meta,
        prices=px,
        fx_rates=fx,
        factor_inputs=fi,
        base_currency=base_currency,
        schema_version=SCHEMA_VERSION,
        dataset_hash=ds_hash,
        source_provenance=source_provenance,
    )
