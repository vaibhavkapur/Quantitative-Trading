"""Cross-asset return, futures roll, FX conversion, and financing/roll cost normalization."""

from __future__ import annotations

import math

import polars as pl

from cross_asset_factor_engine.contracts import NORMALIZED_RETURNS_COLUMNS, MarketDataset
from cross_asset_factor_engine.diagnostics import DiagnosticTracker
from cross_asset_factor_engine.errors import CurrencyConversionError, RollNormalizationError


def normalize_market_data(
    dataset: MarketDataset,
    *,
    annualization_days: int = 252,
    diagnostics: DiagnosticTracker | None = None,
) -> pl.DataFrame:
    """Normalize heterogeneous cross-asset prices into base-currency excess returns.

    Conventions documented in docs/contracts.md and docs/architecture.md:
      - Futures (`equity_index_future`, `commodity_future`):
          Unfunded variation-margin P&L in `quote_currency` converted to `base_currency`
          via `local_return * (FX_{quote,t} / FX_{quote,t-1})`.
          Roll continuity on contract transition uses `prev_front_price_new_contract` so
          spliced contract price jumps never contaminate economic returns.
      - FX Forwards (`fx_forward`):
          Spot P&L in `base_currency` (`local_return * fx_return_factor`) plus lagged
          daily interest-rate differential `(r_base_{t-1} - r_quote_{t-1}) / annualization_days`.
      - Funded Spot (`equity_spot`):
          Full-principal currency-converted total return minus base risk-free rate.
      - Explicit cost drags:
          `roll_cost_drag` = `roll_cost_bps / 10_000` on roll dates (0 otherwise).
          `financing_cost_drag` = `(financing_spread_bps / 10_000) / annualization_days`.
    """
    meta = dataset.contract_metadata
    prices = dataset.prices.sort(["asset_id", "date"])
    fx = dataset.fx_rates.sort(["currency", "date"])
    base_ccy = dataset.base_currency

    if diagnostics is not None:
        diagnostics.inc("prices_ingested", prices.height)
        diagnostics.log(
            "INFO",
            "normalize_start",
            n_assets=meta.height,
            n_price_rows=prices.height,
            base_currency=base_ccy,
        )

    # Prepare base currency FX frame (port_usd_per_unit, port_short_rate)
    port_fx = (
        fx.filter(pl.col("currency") == base_ccy)
        .select(
            pl.col("date"),
            pl.col("usd_per_unit").alias("_port_usd_per_unit"),
            pl.col("short_rate").alias("_port_short_rate"),
            pl.col("short_rate").shift(1).alias("_port_short_rate_prev"),
            pl.col("publication_ts").alias("_port_fx_pub_ts"),
        )
    )
    if port_fx.height == 0:
        raise CurrencyConversionError(
            f"Base currency '{base_ccy}' not found in fx_rates.",
            details={"base_currency": base_ccy},
        )

    # Compute currency-level usd_per_unit relative to portfolio base currency + prior day values
    fx_enriched = (
        fx.join(port_fx, on="date", how="inner")
        .sort(["currency", "date"])
        .with_columns(
            (pl.col("usd_per_unit") / pl.col("_port_usd_per_unit")).alias("_fx_in_base"),
        )
        .with_columns(
            pl.col("_fx_in_base").shift(1).over("currency").alias("_fx_in_base_prev"),
            pl.col("short_rate").shift(1).over("currency").alias("_short_rate_prev"),
        )
        .with_columns(
            pl.when(pl.col("_fx_in_base_prev").is_not_null())
            .then(pl.col("_fx_in_base") / pl.col("_fx_in_base_prev"))
            .otherwise(pl.lit(1.0))
            .alias("_fx_return_factor_ccy")
        )
    )

    quote_fx = fx_enriched.select(
        pl.col("date"),
        pl.col("currency").alias("quote_currency"),
        pl.col("_fx_return_factor_ccy").alias("fx_return_factor"),
        pl.col("short_rate").alias("_quote_short_rate"),
        pl.col("_short_rate_prev").alias("_quote_short_rate_prev"),
        pl.col("_port_short_rate_prev").alias("_port_short_rate_prev"),
        pl.col("publication_ts").alias("_quote_fx_pub_ts"),
    )

    base_fx = fx_enriched.select(
        pl.col("date"),
        pl.col("currency").alias("base_currency"),
        pl.col("short_rate").alias("_base_short_rate"),
        pl.col("_short_rate_prev").alias("_base_short_rate_prev"),
        pl.col("publication_ts").alias("_base_fx_pub_ts"),
    )

    # Join metadata and compute roll-safe local returns
    joined = (
        prices.join(
            meta.select(
                [
                    "asset_id",
                    "asset_class",
                    "quote_currency",
                    "base_currency",
                    "instrument_type",
                    "carry_convention",
                    "financing_spread_bps",
                    "roll_cost_bps",
                ]
            ),
            on="asset_id",
            how="inner",
        )
        .sort(["asset_id", "date"])
        .with_columns(
            pl.col("front_price").shift(1).over("asset_id").alias("_prev_front_price"),
            pl.col("contract_code").shift(1).over("asset_id").alias("_prev_contract_code"),
        )
    )

    # Defensive check for roll continuity even if called directly without validate_prices
    bad_roll = joined.filter(
        pl.col("_prev_contract_code").is_not_null()
        & (pl.col("contract_code") != pl.col("_prev_contract_code"))
        & (
            pl.col("prev_front_price_new_contract").is_null()
            | (pl.col("prev_front_price_new_contract") <= 0)
        )
    )
    if bad_roll.height > 0:
        raise RollNormalizationError(
            "Encountered contract roll transition without valid prev_front_price_new_contract.",
            details={"bad_roll_count": bad_roll.height},
        )

    joined = (
        joined.join(quote_fx, on=["date", "quote_currency"], how="left")
        .join(base_fx, on=["date", "base_currency"], how="left")
    )

    if joined.filter(pl.col("fx_return_factor").is_null()).height > 0:
        raise CurrencyConversionError(
            "Missing FX conversion factor for one or more (date, quote_currency) rows."
        )

    # Local return (roll-adjusted)
    local_ret_expr = (
        pl.when(pl.col("_prev_front_price").is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(pl.col("contract_code") != pl.col("_prev_contract_code"))
        .then(pl.col("front_price") / pl.col("prev_front_price_new_contract") - 1.0)
        .otherwise(pl.col("front_price") / pl.col("_prev_front_price") - 1.0)
    )

    # Carry return component embedded in instrument excess return (strictly using t-1 rates)
    carry_ret_expr = (
        pl.when(pl.col("local_return").is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(pl.col("instrument_type") == "fx_forward")
        .then(
            (pl.col("_base_short_rate_prev") - pl.col("_quote_short_rate_prev"))
            / float(annualization_days)
        )
        .when(pl.col("instrument_type") == "equity_spot")
        .then(-pl.col("_port_short_rate_prev") / float(annualization_days))
        .otherwise(pl.lit(0.0))
    )

    # Gross excess return in portfolio base currency
    gross_ret_expr = (
        pl.when(pl.col("local_return").is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(pl.col("instrument_type") == "equity_spot")
        .then(
            (1.0 + pl.col("local_return")) * pl.col("fx_return_factor")
            - 1.0
            + pl.col("carry_return_component")
        )
        .otherwise(pl.col("local_return") * pl.col("fx_return_factor") + pl.col("carry_return_component"))
    )

    # Explicit roll & financing drags
    roll_drag_expr = (
        pl.when(pl.col("local_return").is_null())
        .then(pl.lit(0.0))
        .when(pl.col("is_roll_date"))
        .then(pl.col("roll_cost_bps") / 10_000.0)
        .otherwise(pl.lit(0.0))
    )
    fin_drag_expr = (
        pl.when(pl.col("local_return").is_null())
        .then(pl.lit(0.0))
        .otherwise((pl.col("financing_spread_bps") / 10_000.0) / float(annualization_days))
    )

    # Implied annualized futures term-structure roll yield (backwardation > 0, contango < 0)
    implied_roll_expr = (
        pl.when(
            pl.col("next_price").is_not_null()
            & pl.col("next_expiry_days").is_not_null()
            & (pl.col("next_expiry_days") > pl.col("front_expiry_days"))
        )
        .then(
            (pl.col("front_price") / pl.col("next_price")).log()
            * (365.0 / (pl.col("next_expiry_days") - pl.col("front_expiry_days")).cast(pl.Float64))
        )
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )

    # Effective publication_ts of normalized row = max(price pub_ts, quote fx pub_ts, base fx pub_ts)
    effective_pub_ts = pl.max_horizontal(
        [
            pl.col("publication_ts"),
            pl.col("_quote_fx_pub_ts"),
            pl.col("_base_fx_pub_ts"),
        ]
    ).alias("publication_ts")

    result = (
        joined.with_columns(local_ret_expr.alias("local_return"))
        .with_columns(carry_ret_expr.alias("carry_return_component"))
        .with_columns(
            gross_ret_expr.alias("gross_excess_return_base"),
            roll_drag_expr.alias("roll_cost_drag"),
            fin_drag_expr.alias("financing_cost_drag"),
            implied_roll_expr.alias("implied_roll_yield"),
            effective_pub_ts,
        )
        .with_columns(
            pl.when(pl.col("gross_excess_return_base").is_null())
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(
                pl.col("gross_excess_return_base")
                - pl.col("roll_cost_drag")
                - pl.col("financing_cost_drag")
            )
            .alias("net_return_before_tc")
        )
        .select(list(NORMALIZED_RETURNS_COLUMNS))
        .sort(["date", "asset_id"])
    )

    if diagnostics is not None:
        n_rolls = int(result.filter(pl.col("is_roll_date")).height)
        n_non_base_fx = int(result.filter(pl.col("quote_currency") != base_ccy).height)
        diagnostics.inc("roll_events_processed", n_rolls)
        diagnostics.inc("fx_conversions_applied", n_non_base_fx)
        diagnostics.log(
            "INFO",
            "normalize_complete",
            n_rows=result.height,
            n_roll_events=n_rolls,
            n_non_base_fx_rows=n_non_base_fx,
        )

    # Sanity guard against NaN/Inf in non-null returns
    bad_num = result.filter(
        pl.col("net_return_before_tc").is_not_null()
        & (pl.col("net_return_before_tc").is_nan() | pl.col("net_return_before_tc").is_infinite())
    )
    if bad_num.height > 0:
        raise RollNormalizationError(
            "Normalization produced NaN or infinite returns.",
            details={"bad_row_count": bad_num.height},
        )

    return result


def annualized_log_roll_yield(
    front_price: float,
    next_price: float,
    front_expiry_days: int,
    next_expiry_days: int,
    annualization_days: float = 365.0,
) -> float:
    """Pure scalar helper for implied annualized futures roll yield."""
    if front_price <= 0 or next_price <= 0 or next_expiry_days <= front_expiry_days:
        raise RollNormalizationError("Invalid inputs for annualized_log_roll_yield.")
    return math.log(front_price / next_price) * (annualization_days / float(next_expiry_days - front_expiry_days))
