# Volatility Forecasting Engine

An offline, deterministic research prototype for **causal realized-volatility forecasting** comparing statistical and machine-learning baselines across chronological walk-forward splits with QLIKE/MAE/calibration metrics, conformal forecast intervals, and regime-specific diagnostics.

> **Scope & Disclaimer:** Independently developed educational/research prototype using deterministic synthetic fixtures. Forecast quality depends on sampling frequency and volatility regime; there is no guarantee of options trading profitability. Live capital, real customer records, market-vendor credentials, and proprietary employer/SAAT artifacts are excluded.

## Core Capabilities

- **Causal Realized-Volatility Features & Horizons (`src/volatility_forecasting/features.py`):**
  - Intraday squared-return realized variance, Parkinson range estimator, overnight gap inclusion/inference, and missing-bar coverage diagnostics with clean fallback to daily close-to-close squared returns.
  - Strictly causal features at origin $t$ ($\le t$ only): `rv_1d`, `rv_5d`, `rv_22d`, annualized vols, log-RVs, negative-return leverage terms, overnight gap squared, bar coverage, and short/long volatility ratios.
  - Strictly forward horizon-$h$ targets over $(t+1, \dots, t+h)$ with recorded `label_start_ts_h{h}` / `label_end_ts_h{h}` boundaries.
- **Baselines Isolated from I/O (`src/volatility_forecasting/models.py`):**
  - `NaivePersistenceModel` (lagged RV benchmark)
  - `RollingHistoricalModel` (trailing $W$-day RV average)
  - `EWMAModel` (RiskMetrics recursive filter, $\lambda = 0.94$)
  - `GARCH11Model` (self-contained stationary Gaussian QML via SciPy L-BFGS-B + exact multi-step horizon averaging; optional `arch` library adapter via `backend="arch"`)
  - `RidgeHARModel` (`StandardScaler` + `Ridge` in a scikit-learn `Pipeline`, log-variance target with training-only Duan smearing)
  - `HGBRVolModel` (`HistGradientBoostingRegressor` with fixed seed, log-variance target + smearing)
- **Leakage-Guarded Chronological Evaluation (`src/volatility_forecasting/splits.py`, `evaluation.py`):**
  - Expanding or rolling chronological splits with mandatory purge/embargo (`embargo_steps >= horizon_days`) and automated `FutureLeakageError` checks.
  - Explicit `NegativeForecastError` rejection of any negative, `NaN`, or infinite predictions.
  - Multiplicative conformal prediction intervals (50% / 80% / 90%) fit strictly on an inner calibration tail within each training fold.
  - Regime-specific diagnostics (`low`, `normal`, `high`, `shock`) using thresholds fit strictly on each training fold.

## Quickstart

From `volatility-forecasting-engine/`:

```bash
# 1. Create environment and install with dev dependencies
python3 -m uv venv --python 3.12 .venv
python3 -m uv pip install --python .venv/bin/python -e ".[dev]"

# 2. Generate deterministic fixtures and run the full benchmark demo
.venv/bin/python scripts/run_demo.py

# Or via the installed CLI entry point:
.venv/bin/vol-forecast generate-fixtures --out-dir data/fixtures
.venv/bin/vol-forecast validate-data \
  --returns data/fixtures/synthetic_multi_regime_returns.csv \
  --intraday data/fixtures/synthetic_multi_regime_intraday.csv
.venv/bin/vol-forecast run --config config/default.yaml
```

## Example Observed Output (Synthetic Multi-Regime Fixture)

The table below reports **actual observed measurements** on the deterministic synthetic fixture (`SYN_EQ_01`, 420 trading days, 13 intraday 30-min bars/day, calm + elevated + shock regimes, overnight gaps, and missing-bar sessions; `config/default.yaml`, `seed=42`, `horizon=5` days, 4 expanding splits with `embargo_steps=5`, `n_oos=240`, Run ID `default_benchmark-6adebe115efb`). These are **synthetic fixture measurements**, not live market performance claims.

| Model | OOS QLIKE | QLIKE vs Naive | OOS MAE (Ann. Vol) | MAE vs Naive | MZ Slope | MZ $R^2$ | 80% Coverage | Min Pred Var |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `garch_11` | **0.1087** | **+50.93%** | **0.1033** | **+10.77%** | 0.679 | 0.712 | 56.3% | `1.10e-4` |
| `naive_persistence` | 0.2216 | 0.00% | 0.1158 | 0.00% | 0.624 | 0.536 | 81.7% | `2.75e-5` |
| `ml_ridge_har` | 0.2226 | -0.46% | 0.1308 | -13.03% | 0.628 | 0.561 | 69.6% | `9.30e-5` |
| `rolling_historical_22d` | 0.3061 | -38.12% | 0.1446 | -24.92% | 0.802 | 0.526 | 55.8% | `1.16e-4` |
| `ml_hgbr` | 0.3464 | -56.31% | 0.1535 | -32.64% | 1.039 | 0.539 | 36.3% | `1.12e-4` |
| `ewma_094` | 0.3492 | -57.60% | 0.1662 | -43.54% | 0.856 | 0.466 | 51.3% | `1.37e-4` |

**Key diagnostic takeaways from `reports/regime_diagnostics.csv`:**
- `garch_11` leads overall and in the `high` and `shock` regimes (`QLIKE 0.0700` in shock) because the synthetic generator follows a GARCH-with-leverage data-generating process.
- Tree-based `ml_hgbr` under-predicts during out-of-sample volatility shocks (`bias_vol = -0.2463` in `shock`), demonstrating the well-known tree extrapolation limitation under unseen extreme regimes.
- Conformal 80% interval coverage is close to nominal (`81.7%`) for `naive_persistence`, while under-covering during abrupt non-stationary regime transitions for slower-adapting models.

## Validation & Tests

```bash
.venv/bin/pytest -v
```

All 17 automated unit, integration, leakage, boundary, and operational tests pass (`tests/`). See [reports/acceptance.md](reports/acceptance.md) for the complete mapping from acceptance criteria to test and artifact evidence.

## Documentation & Artifacts

- [docs/architecture.md](docs/architecture.md) — Component diagram, data flow, strict causality invariants, and ADR-001.
- [docs/contracts.md](docs/contracts.md) — Input/output schemas, units, versions, error catalog, and shipped fixtures.
- [docs/integration.md](docs/integration.md) — Adapter boundaries, optional `arch` integration, dependency licenses, and fixture provenance.
- [docs/runbook.md](docs/runbook.md) — Setup, reproducible demo walkthrough, validation, diagnostics, and failure recovery.
- [reports/acceptance.md](reports/acceptance.md) — Acceptance criteria evidence table and explicitly marked untested capabilities.
- `reports/` — Raw machine-readable outputs (`model_comparison.csv`, `forecast_intervals.csv`, `regime_diagnostics.csv`, `splits.json`, `run_manifest.json`, `run_summary.json`).

## Assumptions & Known Limitations

- **Synthetic data only:** All bundled measurements come from deterministic synthetic fixtures (`src/volatility_forecasting/fixtures.py`), not live equity/options market feeds.
- **Sampling & microstructure:** Intraday realized variance does not apply microstructure noise corrections (e.g., two-scale RV or realized kernels) that may be required on ultra-high-frequency tick data.
- **Regime non-stationarity:** Multiplicative conformal intervals assume approximate residual exchangeability between the training calibration tail and the test window; sudden regime shifts cause temporary under-coverage.
- **Optional `arch` backend not tested in core env:** `GARCH11Model(backend="arch")` is implemented and verified to fail cleanly with `UnsupportedOptionError` when `arch` is absent; live execution of `arch` is marked not run in environments without `.[arch]` installed.
- **No trading/options P&L modeling:** This prototype forecasts realized variance/volatility only; no options pricing, transaction costs, or live trading execution is modeled.
