# Data Contracts — Multi-Factor Asset Pricing Engine

**Contract / Schema Version:** `1.0.0`

## 1. General Conventions

- **Return Units:** All returns (`total_return`, factor returns, `rf`, `excess_return`, estimated alphas, premia, and pricing errors) are expressed in **decimal return per period** (e.g., `0.0125` = `+1.25%` per period), NOT percentage points.
- **Timestamp Semantics:** `date` is an ISO-8601 calendar date string (`YYYY-MM-DD`) representing the **period-end observation date** (e.g., `2020-01-31` for January 2020 monthly returns). All four input tables must use the same period-end convention and frequency (`M`, `D`, `W`, or `Q`).
- **Identifiers:**
  - `asset_id`: non-null string identifying an asset or test portfolio (e.g., `S1V1`, `P1`).
  - Factor names: non-null column headers in `factor_returns` matching the `factors` list in `EngineConfig.models`.
  - `run_id`: deterministic content-addressed identifier `mfape-<16-hex-chars>` derived from schema version, engine version, canonical config (excluding `output_dir`), and SHA-256 hashes of all input files.
- **Supported Tabular Formats:** `.csv` and `.parquet`.

## 2. Canonical Input Schemas

### 2.1 `asset_returns`
| Column | Type | Required | Constraints |
|---|---|---|---|
| `date` | `string` (`YYYY-MM-DD`) | Yes | Non-null, valid ISO date, conforms to configured `frequency` cadence |
| `asset_id` | `string` | Yes | Non-null; `(date, asset_id)` must be unique |
| `total_return` | `float64` | Yes | Finite (no null, NaN, or ±Inf); decimal total return for the period |

### 2.2 `factor_returns`
| Column | Type | Required | Constraints |
|---|---|---|---|
| `date` | `string` (`YYYY-MM-DD`) | Yes | Non-null, unique primary key, valid ISO date, conforms to `frequency` cadence |
| `<factor_name>` | `float64` | Yes (all factors referenced in config) | Finite decimal factor returns (market factor `MKT_RF` is already excess of `rf`; long-short factors such as `SMB`, `HML`, `MOM` are zero-investment spreads) |

### 2.3 `risk_free`
| Column | Type | Required | Constraints |
|---|---|---|---|
| `date` | `string` (`YYYY-MM-DD`) | Yes | Non-null, unique primary key, valid ISO date |
| `rf` | `float64` | Yes | Decimal risk-free return over the period. Missing dates or null/non-finite values trigger `MissingRiskFreeError` under `rf_missing_policy="error"`, or drop the period under `"drop_period"`. |

### 2.4 `portfolio_memberships`
| Column | Type | Required | Constraints |
|---|---|---|---|
| `date` | `string` (`YYYY-MM-DD`) | Yes | Non-null, valid ISO date |
| `asset_id` | `string` | Yes | Non-null; `(date, asset_id)` must be unique |
| `in_universe` | `bool` (or `0`/`1`) | Yes | Non-null point-in-time active indicator. Only rows where `in_universe == true` enter the excess-return panel. |

### 2.5 Excess Return Definition
For every aligned period $t$ and active asset $i$:
$$R^e_{i,t} = R^{\text{total}}_{i,t} - R_{f,t}$$

## 3. Output Contracts

Every engine run returns a typed `RunOutput` object and optionally writes four versioned files into `output_dir`:

| File | Description |
|---|---|
| `run_result.json` | Complete versioned JSON document (`schema_version`, `engine_version`, `run_id`, `config`, `input_fingerprints`, `runtime_environment`, `split_boundaries`, `diagnostics`, `models` with time-series, Fama-MacBeth, and OOS metrics, and `comparison`). |
| `replay_manifest.json` | Minimal deterministic manifest (`schema_version`, `engine_version`, `run_id`, `config`, `input_fingerprints`, `runtime_environment`) consumed by `mfape replay`. |
| `model_comparison.csv` | Flat tabular comparison of all configured models vs. `baseline_model_name` (CAPM). |
| `summary.md` | Human-readable Markdown report summarizing model comparisons and factor premia with FM, Shanken, and HAC standard errors. |

## 4. Validation Errors and Policies

| Error Class | Error Code | Trigger Condition | Policy / Recovery |
|---|---|---|---|
| `ConfigurationError` | `CONFIGURATION_ERROR` | Missing config file, malformed JSON, missing required fields, duplicate model names, missing baseline model | Fix config JSON and re-run `mfape validate` |
| `UnsupportedInputError` | `UNSUPPORTED_INPUT` | Unsupported file extension, `frequency`, `cov_type`, or policy name | Use documented supported values |
| `DataContractError` | `DATA_CONTRACT_ERROR` | Missing columns, duplicate keys, null/NaN/Inf values, replay SHA-256 fingerprint mismatch | Repair upstream input table or verify file provenance |
| `MissingRiskFreeError` | `MISSING_RISK_FREE` | Risk-free observation missing/non-finite for an aligned period | Default `rf_missing_policy="error"`; set `"drop_period"` to drop with warning counter |
| `PeriodAlignmentError` | `PERIOD_ALIGNMENT_ERROR` | Invalid ISO date, broken frequency cadence, duplicate month/quarter, asset/factor date mismatch | Default `period_alignment_policy="strict"`; set `"inner"` to intersect dates with warning counter |
| `UniverseChangeError` | `UNIVERSE_CHANGE_ERROR` | Mid-sample asset entry/exit under `universe_policy="strict_balanced"` | Switch to `"pit_active"` (unbalanced PIT) or `"min_obs_balanced"` (common-core balanced) |
| `DateLeakageError` | `DATE_LEAKAGE_ERROR` | `test_start <= train_end` or overlapping sliced panels | Enforce strictly disjoint `train_end < test_start` |
| `RankDeficiencyError` | `RANK_DEFICIENCY_ERROR` | Collinear factor matrix or collinear cross-sectional beta matrix (rank $< p$ or $\kappa > \text{max\_condition\_number}$) | Remove redundant factor or increase cross-sectional beta dispersion |
| `InsufficientObservationsError` | `INSUFFICIENT_OBSERVATIONS` | Too few active periods ($< \max(\text{min\_ts\_observations}, p+1)$) or assets ($< \max(\text{min\_cs\_assets}, p+1)$) | Expand date window or adjust asset filters |

## 5. Fixture Catalog (Valid, Boundary, and Invalid)

All fixtures are synthetic, deterministic, and regenerated via `scripts/generate_fixtures.py`:

| Fixture Directory | Type | Purpose & Provenance |
|---|---|---|
| `data/fixtures/reference_hand_calculated/` | Valid (Analytical Oracle) | 6 months, 4 portfolios, 2 exactly orthogonal factors, zero idiosyncratic noise; paired with `expected_oracle.json` for exact closed-form verification. |
| `data/fixtures/synthetic_ff3/` | Valid (Research Demo) | 180 months (2010–2024), 25 Size×Value portfolios generated from a 3-factor DGP (`seed=42`) plus unloaded `MOM` factor; paired with `ground_truth.json`. |
| `data/fixtures/universe_change/` | Boundary (Universe Churn) | 48 months, 5 core assets + 1 exiting asset (`EXIT_M24`) + 1 entering asset (`ENTRY_M25`) (`seed=101`). |
| `data/fixtures/invalid/missing_rf/` | Invalid | Omits `rf` for `2020-03-31` and `2020-05-31`. |
| `data/fixtures/invalid/rank_deficient_factors/` | Invalid | Adds `FAC_COLLINEAR = 2 * MKT_RF + 0.005` causing exact rank deficiency. |
| `data/fixtures/invalid/misaligned_periods/` | Invalid | Shifts `factor_returns` dates by one month relative to `asset_returns`. |
