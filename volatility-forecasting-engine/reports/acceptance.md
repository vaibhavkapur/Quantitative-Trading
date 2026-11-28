# Acceptance Criteria Evidence — Volatility Forecasting Engine

**Run ID:** `default_benchmark-6adebe115efb`  
**Config:** `config/default.yaml` (fingerprint `5f4cbbec95da8350`, seed `42`)  
**Runtime Environment:** Python `3.12.15`, NumPy `2.5.5`, Polars `1.44.2`, SciPy `1.18.1`, scikit-learn `1.9.1`, pytest `8.4.2`  
**Fixture Provenance:** Synthetic multi-regime dataset (`data/fixtures/synthetic_multi_regime_returns.csv`, SHA-256 `2fd9a71c...fadd5`; `synthetic_multi_regime_intraday.csv`, SHA-256 `76a49281...62c`). All figures below are **measured on synthetic fixtures**, not real-market observations.

## 1. Project-Specific Acceptance Criteria

| Acceptance Criterion | Status | Automated Test / Artifact Evidence | Observed Measurement |
|---|---|---|---|
| **All forecasts nonnegative** | **PASS** | `tests/test_models_and_edge_cases.py::test_all_baseline_models_fit_predict_nonnegative`, `test_negative_and_nan_forecast_explicit_rejection`; `reports/run_summary.json` | `all_forecasts_nonnegative = true`; minimum predicted variance across 1,440 OOS forecasts = `2.7504e-5` (`>= 0`); synthetic negative/NaN test doubles explicitly raise `NegativeForecastError`. |
| **Report OOS QLIKE and MAE against naive baseline** | **PASS** | `reports/model_comparison.csv`, `reports/run_summary.json`, `tests/test_leakage_and_pipeline.py::test_fixed_splits_exact_reproducibility_and_baseline_comparison` | `garch_11` OOS QLIKE = `0.1087` (**+50.93%** vs `naive_persistence` `0.2216`); OOS MAE (vol) = `0.1033` (**+10.77%** vs `0.1158`). All 6 models reported with `qlike_vs_baseline_pct` and `mae_vol_vs_baseline_pct`. |
| **Fixed splits reproduce metrics** | **PASS** | `tests/test_leakage_and_pipeline.py::test_fixed_splits_exact_reproducibility_and_baseline_comparison`; `reports/splits.json`, `reports/run_manifest.json` | Two independent pipeline runs with identical config/inputs/seed produce identical `run_id`, bit-equal `model_comparison`, `forecast_intervals`, and `regime_diagnostics` DataFrames. |
| **All required test scenarios pass (or marked not run)** | **PASS** | `pytest -v` (17/17 passed in 2.87s) across all 5 test matrix areas below | 17 passed, 0 failed. Optional `arch` package backend execution explicitly marked **NOT RUN** below. |
| **README documents setup, demo, validation, assumptions, and limitations** | **PASS** | `README.md`, `docs/runbook.md`, `docs/architecture.md`, `docs/contracts.md`, `docs/integration.md` | Distinguishes synthetic fixture observations from live market performance; documents setup, CLI/demo commands, assumptions, and known limitations. |
| **No real transactions, customer data, credentials, or proprietary artifacts** | **PASS** | Codebase & `data/fixtures/` inspection | 100% original code and synthetically generated fixtures (`src/volatility_forecasting/fixtures.py`); zero credentials, customer records, or external proprietary code. |

## 2. Test and Validation Matrix Mapping

| Test Area | Required Scenario | Test Cases | Result |
|---|---|---|---|
| **Domain correctness** | Hand-calculated reference fixture | `tests/test_contracts_and_features.py::test_hand_calculated_reference_fixture_exact_match`, `tests/test_models_and_edge_cases.py::test_ewma_exact_analytical_recursion`, `test_garch11_analytical_horizon_averaging` | **PASS** (rtol `1e-9` to `1e-12`) |
| **Domain correctness** | Constant returns | `tests/test_contracts_and_features.py::test_constant_returns_zero_variance_clean`, `tests/test_models_and_edge_cases.py::test_all_models_handle_constant_returns_cleanly` | **PASS** (exact `0.0` variance, QLIKE `0.0`, no NaN/Inf) |
| **Domain correctness** | Missing bars | `tests/test_contracts_and_features.py::test_missing_bars_partial_and_full_session_fallback` | **PASS** (partial coverage `0.5`, full missing fallback to daily squared return without double counting) |
| **Domain correctness** | Overnight gaps | `tests/test_contracts_and_features.py::test_overnight_gap_inclusion_vs_exclusion` | **PASS** (explicit and inferred overnight gap squared added accurately) |
| **Integration and resilience** | Future label leakage | `tests/test_leakage_and_pipeline.py::test_future_label_leakage_detected_and_blocked`, `test_causal_invariance_under_future_perturbation` | **PASS** (`FutureLeakageError` raised on unpurged splits; 20x future perturbation produces bit-identical prior features & forecasts) |
| **Boundary and failure behavior** | Volatility shock | `tests/test_shock_and_operational.py::test_volatility_shock_stability_and_regime_classification` | **PASS** (finite nonnegative forecasts under extreme shock; `shock` regime tagged; EWMA shock forecast > normal forecast) |
| **Boundary and failure behavior** | Negative forecast rejection | `tests/test_models_and_edge_cases.py::test_negative_and_nan_forecast_explicit_rejection` | **PASS** (`NegativeForecastError` raised directly and in pipeline) |
| **Reproducibility** | Same versioned inputs/config & fixed seed | `tests/test_leakage_and_pipeline.py::test_fixed_splits_exact_reproducibility_and_baseline_comparison` | **PASS** (exact DataFrame equality) |
| **Operational review** | Missing config, unsupported inputs, CLI diagnostics | `tests/test_contracts_and_features.py::test_data_contract_validation_errors`, `tests/test_shock_and_operational.py::test_operational_missing_and_malformed_configs`, `test_cli_end_to_end_and_error_reporting` | **PASS** (CLI exit code `2` + structured JSON error payload on stderr; zero silent failures) |

## 3. Explicitly Untested / Out-of-Scope Capabilities

- **Optional `arch` GARCH backend live fitting:** **NOT RUN** in the default offline core environment because the optional `arch` dependency (`pip install -e '.[arch]'`) is not installed. The adapter's missing-dependency guard (`UnsupportedOptionError`) **is** tested and passing (`test_optional_arch_backend_clear_error_when_missing`).
- **Live market / tick-level real-world datasets:** **NOT RUN** (out of scope per plan; offline synthetic fixtures only).
- **Options pricing, hedging, and live execution:** **OUT OF SCOPE / DISABLED**.
