# Acceptance Verification Report

**Project:** Statistical Arbitrage Research Platform (`statistical-arbitrage-research-platform`)  
**Verification Environment:** macOS Darwin 25.5.0 (arm64), CPython `3.12.15`, `numpy 2.5.5`, `polars 1.44.2`, `statsmodels 0.15.0`, `scipy 1.18.1`, `pytest 8.4.2`, `ruff 0.17.6`  
**Data Provenance:** 100% deterministic synthetic fixtures (`data/fixtures/manifest.json`). **Live / permitted historical third-party market feeds were NOT run in this offline environment** and are explicitly marked below.

## 1. Project-Specific Acceptance Criteria Mapping

| # | Acceptance Criterion (`plan.md` §5) | Status | Verification Evidence |
|---|---|---|---|
| AC-1 | **No test-window pair selection** | **PASS** | `tests/test_universe_and_leakage.py::test_no_test_window_pair_selection_invariance` scrambles all prices for `date > fold.train_end` and asserts bit-for-bit identical fold-1 universe and selected `SpreadModel` parameters; `test_explicit_leakage_guards_raise` confirms `LeakageViolationError` on boundary violations. |
| AC-2 | **Report OOS net returns and uncertainty** | **PASS** | `reports/default_walkforward_demo_659102efff8c_summary.json` and `reports/reference_deterministic_fixture_caf5c9d29032_summary.json` report OOS net/gross returns, Sharpe, analytical Lo/Mertens Sharpe SE, and 95% circular block-bootstrap CIs; asserted by `tests/test_evaluation_and_reproducibility.py`. |
| AC-3 | **All trades reconcile to positions** | **PASS** | Mandatory `verify_trade_position_reconciliation` passed on both runs: Demo (`261` trades, `2,144` position checks, `max_position_diff = 0.0`, `max_equity_diff = 0.0`); Reference (`187` trades, `313` position checks, `max_position_diff = 0.0`, `max_equity_diff = 0.0`). Tamper detection verified by `tests/test_backtest_reconciliation.py::test_all_trades_reconcile_and_tampering_raises`. |
| AC-4 | **All required test scenarios pass (untested deps marked)** | **PASS** | `ruff check src tests scripts`: 0 errors; `pytest`: **30 passed, 0 failed, 0 warnings** (9.73s). External/historical vendor equity datasets: **Not run** (offline prototype scope). |
| AC-5 | **README documents setup, demo, validation, assumptions, limitations, synthetic vs actual distinction** | **PASS** | Documented in `README.md` with explicit synthetic provenance labels and links to all `docs/` and `reports/` artifacts. |
| AC-6 | **No real transactions, customer data, credentials, or proprietary employer/SAAT artifacts** | **PASS** | All source code is original; all data fixtures are synthetic RNG outputs; no live brokers, credentials, or external endpoints exist. |

## 2. Test & Validation Matrix Coverage (`plan.md` §4)

| Test Area | Required Scenario | Automated Tests | Result |
|---|---|---|---|
| Domain correctness | Known cointegrated vs independent series (plus hand-calculated OLS/TLS/AR(1)/BH oracles & 3-asset Johansen basket) | `tests/test_cointegration_reference.py` (7 tests), `tests/test_signals.py` (6 tests) | 13 / 13 PASSED |
| Integration & resilience | Train/test leakage invariance, point-in-time delisting filter, and 4-for-1 corporate action split invariance | `tests/test_universe_and_leakage.py` (4 tests) | 4 / 4 PASSED |
| Boundary & failure behavior | Unavailable borrow (graceful block & strict error), mid-series borrow outage (`TECH_D`), missing prices (`raise` & `drop_symbol`), unadjusted split, duplicate keys, negative price | `tests/test_backtest_reconciliation.py` (4 tests), `tests/test_contracts.py` (6 tests) | 10 / 10 PASSED |
| Reproducibility | Same versioned inputs, configuration, and fixed seed yield identical logical outputs | `tests/test_evaluation_and_reproducibility.py::test_end_to_end_reproducibility_and_required_report_fields` | PASSED |
| Operational review | Missing configuration, unknown config fields, invalid threshold ordering, unsupported file format, CLI non-zero exit diagnostics | `tests/test_contracts.py::test_operational_config_errors_and_cli_nonzero_exit`, `tests/test_evaluation_and_reproducibility.py::test_cli_validate_and_run_smoke` | 2 / 2 PASSED |

## 3. Measured Synthetic Results Summary

> **Important:** All metrics below are measured on **deterministic synthetic fixtures** (`data_provenance = "synthetic_fixture"`), NOT real market observations, and do not imply live trading edge.

### 3.1 Primary Walk-Forward Demo (`default_walkforward_demo_659102efff8c`)

- **Config hash:** `caf52c43cbd5e7db` | **Prices hash:** `4da7f63daaf4b269` | **Borrow hash:** `6f033c71104669ec`
- **Schedule:** 6 rolling folds (180d train, 1d embargo, 60d test -> 360 OOS trading days, `2022-09-13` to `2024-01-29`)
- **OOS Net Return:** `-0.19%` (`-0.001882`) vs **OOS Gross Return:** `+0.12%` (`+0.001156`)
- **Cost Breakdown:** Commission `\$901.79`, Slippage `\$1,803.59`, Borrow `\$345.99` -> total cost drag `30.39 bps` (`annualized_turnover = 4.19x`, `261` trades)
- **OOS Net Sharpe:** `-0.0653` (analytical SE `0.8379`, 95% block-bootstrap CI `[-1.6875, +1.5005]`; 95% annualized return CI `[-3.05%, +2.77%]`)
- **Max Drawdown (Net):** `-2.42%` | **Positive OOS Folds:** `3 / 6` (`50.0%`)
- **Multiple-Testing Controls:** BH-FDR (`alpha = 0.10`) across `118` pair + `87` basket fold hypotheses (`78` passed raw `p <= 0.05`, `62` passed BH-FDR). Across `4` attempted `z_entry` configurations (`1.5, 1.75, 2.0, 2.25`), Deflated Sharpe Probability `= 0.2786` (expected null max Sharpe `0.4267`).
- **Benchmarks (exact same 360 OOS days):**
  - Zero-cost frictionless strategy: `+0.12%` total return, Sharpe `+0.0545` (cost impact `-0.30%`, `-0.1198` Sharpe)
  - Equal-weight active universe: `+21.00%` total return, Sharpe `+1.3984`, Max DD `-13.02%`
- **Delisting & Corporate Actions:** `TECH_DELIST` liquidated cleanly on delisting (`delist_liquidations = 1`, `missing_price_skips = 1`); `TECH_A` 2-for-1 split and \$0.50 cash dividend validated without distortion.

### 3.2 Reference Deterministic Fixture (`reference_deterministic_fixture_caf5c9d29032`)

- **Config hash:** `f4bf1c35af088e36` | **Prices hash:** `e376f058ce14198e` | **Borrow hash:** `9ac9a70d804974e1`
- **Schedule:** 3 rolling folds (120d train, 1d embargo, 40d test -> 120 OOS trading days)
- **OOS Net Return:** `+11.02%` vs **Gross Return:** `+11.45%` (cost drag `42.49 bps`, `187` trades, `annualized_turnover = 24.33x`)
- **OOS Net Sharpe:** `2.8762` (analytical SE `1.3576`, 95% bootstrap CI `[-0.5725, +5.8211]`; DSR prob across 2 trials `0.9777`)
- **Hedge-Ratio Stability (`PAIR:REF_Y|REF_X`, true $\beta = 1.50$):** Selected in all 3 folds with mean $\hat\beta = 1.5131$, std `0.0251`, max drift `0.0488`.
- **Benchmarks:** Zero-cost strategy `+11.45%` (Sharpe `2.9763`); Equal-weight universe `-10.26%` (Sharpe `-0.8425`).

## 4. Explicitly Untested / Deferred Capabilities

- **Live / historical vendor equity datasets (e.g., CRSP, Compustat, Polygon, IBKR):** Not run — offline synthetic fixtures only.
- **Intraday / tick-level execution, partial fills, and non-linear market impact:** Outside scope of daily close-to-next-close prototype.
- **Tax lots, margin calls, financing spreads on cash collateral, and integer share-lot rounding:** Outside scope.
- **CI workflow execution on GitHub Actions runners:** Workflow defined in `../.github/workflows/statistical-arbitrage-research-platform.yml` at the repository root with all underlying commands verified locally.
