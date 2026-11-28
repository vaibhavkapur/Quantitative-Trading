# Data Contracts & Schema Specification

**Schema version:** `1.0.0`  
**Base currency (default):** `USD`  
**Timestamp standard:** All timestamps are ISO-8601 UTC (`YYYY-MM-DDTHH:MM:SSZ`), stored as Polars `Datetime("us", "UTC")`. Trading dates are ISO `YYYY-MM-DD` (`pl.Date`).

---

## 1. Timestamp Semantics & Look-Ahead Invariant

Every time-varying input carries two distinct timestamps:
- **`as_of_ts`**: When the economic observation or close quote was formed.
- **`publication_ts`**: When the record was actually published and became observable to a trading system.

**Hard invariant:**
1. On ingestion, every row must satisfy `as_of_ts <= publication_ts`. Otherwise `TimestampGateError` is raised.
2. For every decision date $d$ with configured UTC cutoff `decision_cutoff_utc_time` (default `21:00:00Z`), the decision timestamp is:
   `decision_as_of_ts(d) = d + decision_cutoff_utc_time`.
3. Only input rows with `publication_ts <= decision_as_of_ts(d)` may be read when computing signals on date $d$.
4. Every emitted signal records `source_max_publication_ts` (maximum `publication_ts` across all consumed rows) and `provenance_hash`. The validator asserts `source_max_publication_ts <= as_of_ts` across 100% of rows.

---

## 2. Canonical Input Schemas

### 2.1 `contract_metadata.csv`
Defines the investable universe, asset class, currency conventions, and explicit cost parameters.

| Column | Polars Type | Nullable | Units / Domain | Validation Rules |
|---|---|---|---|---|
| `asset_id` | `String` | No | Canonical ID (e.g. `EQ_SPX`) | Non-empty, unique |
| `asset_class` | `String` | No | `equity`, `fx`, `commodity` | Must be in supported set |
| `quote_currency` | `String` | No | ISO-4217 3-letter uppercase | e.g. `USD`, `EUR`, `JPY`, `GBP` |
| `base_currency` | `String` | Yes | ISO-4217 3-letter uppercase | Required for `asset_class == "fx"` |
| `instrument_type` | `String` | No | `equity_index_future`, `equity_spot`, `fx_forward`, `commodity_future` | Must be in supported set |
| `carry_convention` | `String` | No | `dividend_minus_financing`, `interest_rate_differential`, `futures_term_structure_roll` | Must match `asset_class` |
| `contract_multiplier` | `Float64` | No | Contract units per index point | Strictly `> 0` |
| `tick_size` | `Float64` | No | Quote currency | Strictly `> 0` |
| `tc_bps` | `Float64` | No | One-way transaction cost (bps) | `>= 0` |
| `financing_spread_bps` | `Float64` | No | Annualized financing/borrow spread (bps) | `>= 0` |
| `roll_cost_bps` | `Float64` | No | One-way roll slippage per roll event (bps) | `>= 0` |
| `active_from` | `Date` | No | ISO date | Valid date |
| `active_to` | `Date` | Yes | ISO date | Optional delisting date |

### 2.2 `prices.csv`
Point-in-time front and next-maturity contract quotes with explicit roll transition pricing.

| Column | Polars Type | Nullable | Units / Domain | Validation Rules |
|---|---|---|---|---|
| `date` | `Date` | No | Trading date | Unique `(date, asset_id)` |
| `asset_id` | `String` | No | FK -> `contract_metadata.asset_id` | Must exist in metadata |
| `contract_code` | `String` | No | Contract symbol (e.g. `CLG24`) | Non-empty |
| `front_price` | `Float64` | No | `quote_currency` | Finite, strictly `> 0` |
| `next_price` | `Float64` | Yes | `quote_currency` | When non-null, strictly `> 0` |
| `front_expiry_days` | `Int64` | No | Calendar days to expiry | `>= 0` |
| `next_expiry_days` | `Int64` | Yes | Calendar days to next expiry | When non-null, `> front_expiry_days` |
| `is_roll_date` | `Boolean` | No | `true` if rolled at close of `date` | Boolean |
| `prev_front_price_new_contract` | `Float64` | Yes | `quote_currency` | Required & `> 0` whenever `contract_code` changes from prior trading day |
| `as_of_ts` | `Datetime(us, UTC)` | No | UTC timestamp | Non-null |
| `publication_ts` | `Datetime(us, UTC)` | No | UTC timestamp | `>= as_of_ts` |

### 2.3 `fx_rates.csv`
Point-in-time FX spot conversion rates and money-market short rates.

| Column | Polars Type | Nullable | Units / Domain | Validation Rules |
|---|---|---|---|---|
| `date` | `Date` | No | Trading date | Unique `(date, currency)`; must cover all price dates for all required currencies |
| `currency` | `String` | No | ISO-4217 3-letter uppercase | Must cover `base_currency` and all asset quote/base currencies |
| `usd_per_unit` | `Float64` | No | USD per 1 unit of `currency` | Strictly `> 0`; equals `1.0` for `USD` |
| `short_rate` | `Float64` | No | Annualized decimal (e.g. `0.05` = 5%) | Finite |
| `as_of_ts` | `Datetime(us, UTC)` | No | UTC timestamp | Non-null |
| `publication_ts` | `Datetime(us, UTC)` | No | UTC timestamp | `>= as_of_ts` |

### 2.4 `factor_inputs.csv`
Point-in-time fundamental and interest-rate factor inputs (supports realistic publication lags).

| Column | Polars Type | Nullable | Units / Domain | Validation Rules |
|---|---|---|---|---|
| `date` | `Date` | No | Reference observation date | Unique `(date, asset_id, publication_ts)` |
| `asset_id` | `String` | No | FK -> `contract_metadata.asset_id` | Must exist in metadata |
| `dividend_yield` | `Float64` | Yes | Annualized decimal | Used for equities carry |
| `earnings_quality` | `Float64` | Yes | Standardized quality score | Used for equities defensive |
| `foreign_short_rate` | `Float64` | Yes | Annualized decimal | Base-currency short rate for FX carry |
| `domestic_short_rate` | `Float64` | Yes | Annualized decimal | Quote-currency short rate for FX carry |
| `convenience_yield_proxy` | `Float64` | Yes | Annualized decimal | Optional commodity metadata |
| `as_of_ts` | `Datetime(us, UTC)` | No | UTC timestamp | Non-null |
| `publication_ts` | `Datetime(us, UTC)` | No | UTC timestamp | `>= as_of_ts` |

---

## 3. Cross-Asset Carry & Return Normalization Definitions

Because carry is structurally different across asset classes, every class uses an explicit, audited formula:

1. **Equities (`dividend_minus_financing`):**
   - Return: unfunded futures P&L converted to base currency via $r^{\text{base}}_{i,t} = r^{\text{local}}_{i,t} \cdot \frac{FX_{q,t}}{FX_{q,t-1}}$.
   - Carry signal: $\text{Carry}^{\text{EQ}}_{i,t} = y^{\text{div}}_{i,t^*} - r^{\text{local}}_{f,t^*} - \text{financing\_spread}_i$, where $t^*$ is the latest record with `publication_ts <= as_of_ts`.
2. **FX Forwards (`interest_rate_differential`):**
   - Return: base-currency spot P&L $r^{\text{local}}_{i,t} \cdot \frac{FX_{q,t}}{FX_{q,t-1}}$ plus lagged daily carry $(r_{\text{base},t-1} - r_{\text{quote},t-1}) / 252$.
   - For reverse-quoted pairs such as `USDJPY` (`base=USD, quote=JPY`), $FX_{\text{JPY},t} = 1/S_t$, so $r^{\text{local}}_t \cdot (FX_t / FX_{t-1}) = (S_t - S_{t-1})/S_t$, exact USD P&L.
   - Carry signal: $\text{Carry}^{\text{FX}}_{i,t} = r^{\text{base}}_{i,t^*} - r^{\text{quote}}_{i,t^*} - \text{financing\_spread}_i$.
3. **Commodities (`futures_term_structure_roll`):**
   - Return: roll-adjusted futures return using `prev_front_price_new_contract` on contract transitions, minus explicit `roll_cost_bps / 10000` on roll dates and daily `financing_spread_bps / 10000 / 252`.
   - Carry signal: annualized log roll yield minus annualized roll cost:
     $$\text{Carry}^{\text{CMD}}_{i,t} = \ln\!\left(\frac{P^{\text{front}}_{i,t}}{P^{\text{next}}_{i,t}}\right)\frac{365}{T_{\text{next}} - T_{\text{front}}} - \frac{\text{roll\_cost\_bps}_i}{10000}\frac{365}{T_{\text{next}} - T_{\text{front}}}$$

---

## 4. Canonical Output Contracts

- **Normalized Returns (`NORMALIZED_RETURNS_COLUMNS`)**: 23 columns including `local_return`, `fx_return_factor`, `carry_return_component`, `gross_excess_return_base`, `roll_cost_drag`, `financing_cost_drag`, `net_return_before_tc`, `implied_roll_yield`, `as_of_ts`, `publication_ts`.
- **Factor Signals (`FACTOR_SIGNALS_COLUMNS`)**: `signal_date`, `as_of_ts`, `source_max_publication_ts`, `asset_id`, `asset_class`, `factor_name`, `raw_value`, `z_score`, `signal_status`, `missing_reason`, `provenance_hash`.
- **Run Report Directory (`reports/runs/<run_id>/`)**:
  - `summary.json`: Run ID, schema version, dataset/config hashes, runtime environment, walk-forward fold boundaries + fitted scaler params, strategy metrics, per-class attribution, constraint audit, provenance audit, correlations, diagnostics counters.
  - `strategy_comparison.csv`, `class_attribution.csv`, `daily_returns.csv`, `portfolio_weights.csv`, `signal_provenance_audit.csv`, `diagnostics.jsonl`.

---

## 5. Error Contract

| Exception Class | Error Code | Trigger Condition |
|---|---|---|
| `ConfigurationError` | `CONFIGURATION_ERROR` | Missing/malformed YAML, unknown keys, `target_volatility > volatility_cap`, `execution_lag_days < 1`, invalid walk-forward windows |
| `SchemaValidationError` | `SCHEMA_VALIDATION_ERROR` | Missing files/columns, type cast error, duplicate keys, non-positive prices, unknown `asset_id` |
| `UnsupportedAssetError` | `UNSUPPORTED_ASSET_ERROR` | Unsupported `asset_class`, `instrument_type`, or mismatched `carry_convention` |
| `TimestampGateError` | `TIMESTAMP_GATE_ERROR` | `publication_ts < as_of_ts` in input data, or `source_max_publication_ts > as_of_ts` in output signals |
| `CurrencyConversionError` | `CURRENCY_CONVERSION_ERROR` | Missing FX currency or trading date coverage, non-positive FX rate, `USD != 1.0` |
| `RollNormalizationError` | `ROLL_NORMALIZATION_ERROR` | Contract code changes without positive `prev_front_price_new_contract`, or NaN/Inf returns |
| `MissingFactorError` | `MISSING_FACTOR_ERROR` | Missing/stale factor input when `data.strict_missing_factors: true` |
| `ConstraintViolationError` | `CONSTRAINT_VIOLATION_ERROR` | Infeasible constraint set or post-optimization violation beyond `constraint_tolerance` |

---

## 6. Fixture Inventory

| Path | Category | Purpose |
|---|---|---|
| `data/fixtures/valid_small/` | Valid + Oracle | 12-day, 4-asset deterministic dataset + hand-calculated pure-Python `expected_reference.json` (`tol = 1e-10`) |
| `data/fixtures/synthetic_multi_year/` | Valid Multi-Year | 1,008-day (4-year), 9-asset seeded (`seed=42`) cross-asset dataset with periodic rolls and 15-day equity fundamental lags |
| `data/fixtures/invalid/` | Invalid | `missing_columns_prices.csv`, `negative_price.csv`, `invalid_roll_transition.csv` |
| `data/fixtures/boundary/` | Boundary | `missing_factor_inputs.csv`, `publication_lag_scenario.json` |
