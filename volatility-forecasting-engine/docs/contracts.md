# Data Contracts — Volatility Forecasting Engine

**Contract Version:** `1.0.0`

## 1. Units and Timestamp Semantics

- **Timestamps:** ISO-8601 strings, Polars `Date`, or Polars `Datetime`. Timezone-aware timestamps are converted to UTC and stored as timezone-naive microsecond timestamps (`Datetime("us")`).
- **Returns (`log_return`, `overnight_return`):** Dimensionless natural-log returns (e.g., `0.01` = +1% log return).
- **Daily Realized Variance (`daily_rv`, `target_rv_h{h}`, `pred_var`):** Daily variance units (squared log returns per trading day, dimensionless, strictly $\ge 0$).
- **Annualized Realized Volatility (`vol_*`, `target_vol_h{h}`, `pred_vol`):** $\sigma_{\text{ann}} = \sqrt{A \cdot \text{RV}_{\text{daily}}}$, where $A$ is `annualization_factor` (default `252` trading days/year; e.g., `0.20` = 20% annualized volatility).
- **Horizon (`horizon_days`, $h$):** Forward averaging window in trading days $(t+1, \dots, t+h)$ strictly after forecast origin $t$.

## 2. Canonical Input Schemas

### 2.1 Timestamped Returns (`returns`, required)

| Column | Type | Required | Constraints |
|---|---|---|---|
| `asset_id` | `String` | Yes | Non-null asset identifier |
| `timestamp` | `Datetime` / ISO-8601 | Yes | Non-null; unique per `(asset_id, timestamp)` |
| `log_return` | `Float64` | Yes | Finite, non-null close-to-close log return |
| `overnight_return` | `Float64` | No | Finite; defaults to `0.0` when omitted |
| `session_date` | `Date` / ISO-8601 | No | Derived from `timestamp` date if omitted; unique per `(asset_id, session_date)` |
| `close_price` | `Float64` | No | Informational close price |

### 2.2 Intraday OHLCV Bars (`intraday_bars`, optional)

| Column | Type | Required | Constraints |
|---|---|---|---|
| `asset_id` | `String` | Yes | Non-null asset identifier |
| `timestamp` | `Datetime` / ISO-8601 | Yes | Non-null; unique per `(asset_id, timestamp)` |
| `session_date` | `Date` / ISO-8601 | Yes | Non-null trading session date |
| `open`, `high`, `low`, `close` | `Float64` | Yes | Strictly $> 0$ and finite on non-missing rows; `low <= min(open, close) <= max(open, close) <= high` |
| `volume` | `Float64` | Yes | Finite and $\ge 0$ on non-missing rows |
| `is_missing` | `Boolean` | No | Defaults to `false`; rows marked `true` are excluded from RV and lower `intraday_bar_coverage` |

### 2.3 Precomputed Realized Volatility Labels (`realized_vol_labels`, optional)

| Column | Type | Required | Constraints |
|---|---|---|---|
| `asset_id` | `String` | Yes | Non-null |
| `timestamp` | `Datetime` / ISO-8601 | Yes | Forecast origin timestamp $t$; unique per `(asset_id, timestamp, horizon_days)` |
| `horizon_days` | `Int64` | Yes | Strictly $> 0$ |
| `realized_variance` | `Float64` | Yes | Finite and $\ge 0$ |
| `realized_vol` | `Float64` | Yes | Finite and $\ge 0$ |

## 3. Canonical Output Contracts

### 3.1 `reports/model_comparison.csv`
One row per evaluated model sorted ascending by OOS `qlike`:
- Identifiers & counts: `model`, `model_type`, `horizon_days`, `n_oos`, `n_splits`, `baseline_model`
- Nonnegativity proof: `min_pred_var` (strictly $\ge 0$)
- Error & skill metrics: `qlike`, `qlike_split_std`, `qlike_vs_baseline_pct`, `mae_vol`, `mae_vol_vs_baseline_pct`, `rmse_vol`, `mae_var`, `bias_vol`
- Mincer-Zarnowitz calibration: `mz_intercept`, `mz_slope`, `mz_r2`
- Interval calibration for each configured nominal level $p \in \{50, 80, 90\}$: `coverage_{p}`, `width_vol_{p}`, `winkler_{p}`

### 3.2 `reports/forecast_intervals.csv`
Row-level OOS predictions and conformal prediction intervals:
- `split_id`, `horizon_days`, `model`, `asset_id`, `timestamp`, `regime`
- `actual_var`, `pred_var`, `actual_vol`, `pred_vol`
- `lower_vol_{p}`, `upper_vol_{p}` for each configured level (guaranteed `0 <= lower_vol_{p} <= upper_vol_{p}`)

### 3.3 `reports/regime_diagnostics.csv`
Per-model, per-regime (`low`, `normal`, `high`, `shock`) metrics:
- `model`, `regime`, `n_obs`, `mean_true_vol`, `mean_pred_vol`, `bias_vol`, `qlike`, `mae_vol`, `rmse_vol`, `coverage_80`

### 3.4 `reports/run_manifest.json` & `reports/splits.json`
Reproducibility metadata: `run_id`, `config_fingerprint`, SHA-256 `input_fingerprints`, `runtime_environment`, `feature_diagnostics`, exact split boundaries, `attempted_configurations`, and `counters`.

## 4. Validation Error Catalog

| Exception | CLI Exit Code | Trigger Condition |
|---|---|---|
| `ConfigurationError` | `2` | Missing/unreadable YAML, malformed fields, invalid quantiles or horizon settings |
| `DataContractError` | `2` | Missing columns, empty table, duplicate timestamps, `NaN`/Inf, OHLC ordering violations, unsupported file extension |
| `FutureLeakageError` | `2` | `embargo_steps < horizon_days`, overlapping train/test indices, or training label window extending to/past `test_start_ts` |
| `NegativeForecastError` | `2` | Any model emitting variance/volatility $< 0$, `NaN`, or `Inf` |
| `ModelFitError` | `2` | Model called before `fit()`, empty training slice, or invalid hyperparameters |
| `UnsupportedOptionError` | `2` | Unsupported model type, split scheme, or `backend="arch"` when optional `arch` is absent |

## 5. Shipped Fixtures (`data/fixtures/`)

- **Valid multi-regime fixture:** `synthetic_multi_regime_returns.csv`, `synthetic_multi_regime_intraday.csv` (420 trading days, 13 bars/day, calm/elevated/shock regimes, overnight gaps, partial & full missing-bar sessions)
- **Hand-calculated reference fixture:** `reference_hand_calc_returns.csv`, `reference_hand_calc_intraday.csv` (6 trading days with analytically verified RV, overnight gap, and multi-horizon targets)
- **Boundary fixture:** `boundary_constant_returns.csv` (80 days of constant zero returns)
- **Invalid fixtures:** `invalid_missing_column_returns.csv` (missing `log_return`), `invalid_ohlc_intraday.csv` (`high < low`)
