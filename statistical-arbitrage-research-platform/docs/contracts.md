# Data Contracts & Schemas

**Contract Version:** `1.0.0`  
**Timestamp Semantics:** Daily end-of-day UTC calendar trading dates (`YYYY-MM-DD`, Polars `pl.Date`). Signals computed at close of date `t` execute no earlier than date `t + execution_delay_days` (default `t + 1`).

## 1. Input Schemas

### 1.1 Prices Table (`prices_path`: `.parquet` or `.csv`)

Primary key: `(date, symbol)` — must be unique and non-null.

| Column | Type | Units / Semantics | Validation Rules |
|---|---|---|---|
| `date` | `Date` | UTC trading date (`YYYY-MM-DD`) | Non-null |
| `symbol` | `String` | Canonical uppercase ticker identifier | Non-null |
| `close_adj` | `Float64` | Backward split- & dividend-adjusted close (USD) | Finite, `> 0` |
| `close_raw` | `Float64` | Unadjusted trading close (USD) | Finite, `> 0` |
| `volume` | `Float64` | Daily share volume (shares) | Finite, `>= 0` |
| `adv_usd` | `Float64` | Point-in-time trailing average daily dollar volume (USD) | Finite, `>= 0` |
| `market_cap_usd` | `Float64` | Point-in-time market capitalization (USD) | Finite, `>= 0` |
| `sector` | `String` | Point-in-time sector classification | Non-null |
| `is_active` | `Boolean` | Whether security is actively listed/tradable at close | Non-null |
| `split_factor` | `Float64` | Forward split multiplier on ex-date (`2.0` for 2-for-1; `1.0` otherwise) | Finite, `> 0` |
| `cash_dividend` | `Float64` | Cash dividend per raw share on ex-date (USD) | Finite, `>= 0` |

**Corporate Action Consistency Rule:** Whenever `split_factor != 1.0` on date `t` for a symbol with prior bar `t-1`, the validator checks that `close_adj[t] / close_adj[t-1]` agrees within 5% relative tolerance with `(close_raw[t] * split_factor[t] + cash_dividend[t]) / close_raw[t-1]`. Unadjusted jumps passed as `close_adj` raise `ContractValidationError`.

**Missing Data Policy (`data.missing_price_policy`):**
- `"raise"` (default): Any null/NaN/infinite price or active-symbol date gap exceeding `max_missing_fraction` raises `MissingDataError`.
- `"drop_symbol"`: Affected symbols are dropped deterministically and recorded in `input_validation.dropped_symbols`.

### 1.2 Borrow Schedule (`borrow_path`: `.parquet` or `.csv`)

Primary key: `(date, symbol)` — must be unique and non-null.

| Column | Type | Units / Semantics | Validation Rules |
|---|---|---|---|
| `date` | `Date` | UTC trading date (`YYYY-MM-DD`) | Non-null |
| `symbol` | `String` | Canonical ticker matching prices table | Non-null |
| `borrow_available` | `Boolean` | Whether short locate is available on date `t` | Non-null |
| `borrow_fee_bps_annual` | `Float64` | Annualized short borrow fee in basis points (`100 bps = 1.0%`) | Finite, `>= 0` |
| `max_short_notional` | `Float64` | Maximum allowed short dollar notional (USD) | Finite, `>= 0` |

## 2. Configuration Contract (`config/*.toml` or `.json`)

Validated via `statarb.config.PlatformConfig` with `extra="forbid"`. Key invariants:
- `0 <= signals.z_exit < signals.z_entry < signals.z_stop`
- `0 < selection.min_half_life_days < selection.max_half_life_days`
- `walk_forward.train_window_days >= universe.min_history_days`
- `execution.execution_delay_days >= 1` (prevents same-bar execution lookahead)
- Every `evaluation.sweep_z_entries` value satisfies `z_exit < z < z_stop`.

## 3. Output Contracts (`reports/`)

Every pipeline run produces stable deterministic identifiers:
- `config_hash`: first 16 hex chars of SHA-256 over canonical sorted JSON configuration.
- `prices_hash`, `borrow_hash`: first 16 hex chars of SHA-256 over sorted canonical CSV serialization of validated input frames.
- `run_id`: `<run_name>_<sha256(run_name|config_hash|prices_hash|borrow_hash|seed)[:12]>`.

Emitted files per run:
1. `<run_id>_summary.json` (and `<run_name>_latest.json`): top-level fields `schema_version`, `run_id`, `run_name`, `config_hash`, `data_provenance`, `runtime_environment`, `input_validation`, `config`, `performance`, `uncertainty`, `multiple_testing`, `stability`, `benchmarks`, `reconciliation`, `diagnostics`, `folds`.
2. `<run_id>_equity_curve.csv`: daily OOS ledger (`date, fold_id, equity, cash, holdings_value, gross_notional, net_notional, daily_return_net, daily_return_gross, commission, slippage, borrow_cost, turnover, active_spreads`).
3. `<run_id>_trades.csv`: immutable trade ledger (`trade_id, date, fold_id, symbol, side, quantity, price_adj, notional, commission, slippage, reason`).

## 4. Error Hierarchy

| Exception | Trigger | CLI Exit Code |
|---|---|---|
| `ConfigurationError` | Missing config file, parse failure, unknown key, invalid parameter bounds | `2` |
| `ContractValidationError` | Missing required columns, duplicate keys, non-positive prices, corporate action mismatch | `2` |
| `MissingDataError` | Missing input file, null/NaN prices under `raise`, insufficient calendar dates for walk-forward | `2` |
| `BorrowUnavailableError` | Short trade attempted without locate when `strict_borrow_errors = true` | `2` |
| `LeakageViolationError` | Training/selection invoked with timestamps exceeding split boundary | `2` |
| `ReconciliationError` | Replay of trades/cashflows disagrees with daily positions or equity ledger | `2` |
| `UnsupportedInputError` | Unsupported file format (e.g. `.xlsx`, `.yaml`), unknown estimator or policy | `2` |

## 5. Fixture Inventory (`data/fixtures/`)

All fixtures are 100% synthetic and deterministically generated by `scripts/generate_fixtures.py` (documented in `data/fixtures/manifest.json`):

- **Valid Reference Fixture:** `reference_pair_prices.csv`, `reference_pair_borrow.csv`, `reference_pair_ground_truth.json` (seed 7, 260 days, 4 symbols: planted cointegrated pair `REF_Y`/`REF_X` with $\alpha=5.0, \beta=1.5, \phi=0.85, \text{HL}\approx 4.265\text{d}$ plus independent random walks `REF_IND_A`/`REF_IND_B`).
- **Valid Synthetic Universe Fixture:** `synthetic_universe_prices.parquet`, `synthetic_universe_borrow.parquet`, `synthetic_universe_metadata.json` (seed 42, 600 days, 14 symbols across `TECH`, `FIN`, `IND`, including cointegrated pairs, a 3-asset Johansen basket, a regime-breaking pair, independent controls, a penny/illiquid stock, a mid-sample delisting, a 2-for-1 split + \$0.50 cash dividend, and a borrow outage window).
- **Invalid & Boundary Fixtures (`data/fixtures/invalid/`):**
  - `missing_prices.csv` — null `close_adj` bar
  - `duplicate_keys.csv` — duplicate `(date, symbol)` row
  - `negative_price.csv` — negative `close_adj` value
  - `unavailable_borrow.csv` — `borrow_available = false` and `max_short_notional = 0.0` across all dates
  - `unadjusted_split_prices.csv` — 2-for-1 split in `close_raw` copied unadjusted into `close_adj`
