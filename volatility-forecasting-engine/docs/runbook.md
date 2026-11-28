# Operator & Developer Runbook — Volatility Forecasting Engine

## 1. Clean Local Setup

From `volatility-forecasting-engine/`:

```bash
python3 -m uv venv --python 3.12 .venv
python3 -m uv pip install --python .venv/bin/python -e ".[dev]"
```

(Or with standard `python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"`.)

Optional `arch` GARCH backend:
```bash
.venv/bin/pip install -e ".[arch]"
```

## 2. Reproducible Demo Walkthrough

### Step 1: Generate deterministic fixtures
```bash
.venv/bin/vol-forecast generate-fixtures --out-dir data/fixtures
```

### Step 2: Validate input tables against data contracts
```bash
.venv/bin/vol-forecast validate-data \
  --returns data/fixtures/synthetic_multi_regime_returns.csv \
  --intraday data/fixtures/synthetic_multi_regime_intraday.csv
```

### Step 3: Run the full chronological benchmark
```bash
.venv/bin/vol-forecast run --config config/default.yaml
```
Alternatively, run both fixture generation and evaluation in one command:
```bash
.venv/bin/python scripts/run_demo.py
```

### Step 4: Inspect generated reports
- `reports/model_comparison.csv` — OOS QLIKE, MAE, RMSE, Mincer-Zarnowitz calibration, interval coverage, and % improvement vs `naive_persistence`.
- `reports/forecast_intervals.csv` — Row-level predictions and 50%/80%/90% conformal forecast intervals.
- `reports/regime_diagnostics.csv` — Per-regime (`low`, `normal`, `high`, `shock`) metrics.
- `reports/splits.json` — Exact chronological split train/calibration/test boundaries and embargo lengths.
- `reports/run_manifest.json` & `reports/run_summary.json` — Stable `run_id`, SHA-256 input & config fingerprints, counters, and runtime package versions.

## 3. Running Validation & Test Suite

```bash
.venv/bin/pytest -v
```

Targeted scenario verification:
```bash
# Domain correctness (hand-calculated reference, constant returns, missing bars, overnight gaps)
.venv/bin/pytest -v tests/test_contracts_and_features.py tests/test_models_and_edge_cases.py

# Future label leakage & reproducibility
.venv/bin/pytest -v tests/test_leakage_and_pipeline.py

# Volatility shock boundary & operational error handling
.venv/bin/pytest -v tests/test_shock_and_operational.py
```

## 4. Diagnostics, Replay, and Failure Recovery

- **Deterministic Replay / Rerun:** Re-running `vol-forecast run --config config/default.yaml` against unchanged inputs produces the exact same `run_id` (`<run_name>-<hash12>`) and identical CSV/JSON metrics.
- **Structured Error Diagnostics:** Any contract, configuration, leakage, or negative-forecast violation exits with status code `2` and emits a JSON error payload to `stderr` containing `error_type` and `message`.
- **Common Recovery Actions:**
  - `ConfigurationError` (missing/malformed YAML): Verify `--config` path and validate keys against `config/default.yaml`.
  - `DataContractError` (missing columns, NaN, OHLC ordering): Run `vol-forecast validate-data` to pinpoint the failing file and constraint before re-running.
  - `FutureLeakageError`: Ensure `splits.embargo_steps` is either `null` (auto-defaults to `horizon_days`) or explicitly `>= features.primary_horizon`.
  - `NegativeForecastError`: Inspect custom model outputs; negative or non-finite variance predictions are never silently clipped.
  - `UnsupportedOptionError` for `backend="arch"`: Either install `.[arch]` or switch `params.backend` to `"scipy"`.
