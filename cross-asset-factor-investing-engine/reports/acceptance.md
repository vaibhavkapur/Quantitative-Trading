# Acceptance Evidence Report

**Engine:** `cross-asset-factor-investing-engine` v0.1.0  
**Schema Version:** `1.0.0`  
**Verified Run ID (Multi-Year Synthetic):** `default_cross_asset_walkforward_a2c4def8_c420020e`  
**Verified Run ID (Small Analytical Reference):** `reference_fixture_small_c989fd60_b941662a`  
**Runtime Environment:** Python `3.12.15`, Polars `1.44.2`, NumPy `2.5.5`, SciPy `1.18.1`, Pytest `8.4.2`, `macOS-26.5.1-arm64-arm-64bit`  
**Data Provenance:** 100% deterministic synthetic fixtures (`synthetic_seeded_dgp`, `seed=42`, and `synthetic_hand_calculated_reference`). No live market data, real transactions, customer records, credentials, or proprietary employer/SAAT artifacts exist anywhere in this project.

---

## 1. Test & Validation Matrix Mapping

| Test Area | Required Scenario | Automated Evidence | Status |
|---|---|---|---|
| **Domain correctness** | Futures rolls, FX conversions, and missing factors | `tests/test_normalization_rolls_fx.py` (exact match to pure-Python oracle `data/fixtures/valid_small/expected_reference.json` within `1e-10` tol, plus `USDJPY` reverse-quote identity) & `tests/test_factors_and_missing_inputs.py` (missing-factor propagation, low-vol fallback, partial blend, strict mode `MissingFactorError`) | **PASS** (6 tests) |
| **Integration & resilience** | No look-ahead from publication lag | `tests/test_publication_lag_no_lookahead.py` (explicit lagged fundamental gating before/after `publication_ts`, plus end-to-end future perturbation invariance proving bitwise-identical signals, scaler params, weights, and OOS returns before cutoff; `4,185` future rows blocked in multi-year run) | **PASS** (2 tests) |
| **Boundary & failure behavior** | Exposure and volatility cap violations | `tests/test_portfolio_constraints_and_boundaries.py` (tight-cap optimization, unit tests catching all 6 individual cap violations, infeasible gross exposure / volatility cap / net-band regression tests raising `ConstraintViolationError`) | **PASS** (5 tests) |
| **Reproducibility** | Same versioned inputs/configuration and fixed seed | `tests/test_walkforward_and_reproducibility.py` & `cafe verify-replay --config config/default.yaml` (`equivalent=true`, `max_return_diff=0.0`, `max_weight_diff=0.0`, identical `run_id`) | **PASS** (1 test + CLI) |
| **Operational review** | Missing config, unsupported inputs, restart/rerun | `tests/test_contracts_and_validation.py` & `tests/test_cli_and_operational_resilience.py` (invalid fixtures, missing FX currency, unsupported asset class, `publication_ts < as_of_ts`, missing/malformed YAML returning exit code `1` + structured JSON error) | **PASS** (11 tests) |

**Total automated test run:** `25 passed in 24.37s` (`pytest tests -v`).

---

## 2. Project-Specific Acceptance Criteria

| # | Criterion | Measured Evidence | Status |
|---|---|---|---|
| 1 | **Every signal has as-of provenance** | `provenance_audit` in `reports/runs/default_cross_asset_walkforward_a2c4def8_c420020e/summary.json`: `27,180` OOS signals audited; `all_have_as_of_provenance: true`; `lookahead_violations: 0`. Raw audit table in `signal_provenance_audit.csv`. | **PASS** |
| 2 | **Report net OOS performance and turnover by class** | Full net OOS strategy metrics and per-class (`equity`, `fx`, `commodity`) return contribution, cost drag, exposure, and annualized turnover archived in `summary.json`, `strategy_comparison.csv`, and `class_attribution.csv` (summarized in Section 3 below). | **PASS** |
| 3 | **Weights respect constraints** | `constraint_audit` in `summary.json`: `151` rebalances per constrained strategy (`604` total rebalances across `momentum`, `carry`, `defensive`, `blended`), `violations: 0` across every asset weight, gross, net, class gross, volatility, and turnover cap (Section 4 below). | **PASS** |
| 4 | **Required test scenarios pass / untested dependencies marked** | All 25 automated tests + CLI validation + replay checks pass locally on Python 3.12. Untested / out-of-scope production items are explicitly documented in Section 5. | **PASS** |
| 5 | **README documents setup, demo, validation, assumptions, limitations** | Documented in `README.md`, `docs/architecture.md`, `docs/contracts.md`, `docs/integration.md`, `docs/runbook.md`, and `docs/adr/0001-cross-asset-normalization-and-asof-gating.md`; synthetic fixture provenance is explicitly distinguished from live observations. | **PASS** |
| 6 | **No real transactions, customer data, credentials, or proprietary artifacts** | Verified across all source, config, fixture, test, and report files. | **PASS** |

---

## 3. Measured Walk-Forward Results (Synthetic Multi-Year Fixture)

> **Provenance Notice:** The figures below are measured observations on the deterministic synthetic 4-year fixture (`data/fixtures/synthetic_multi_year/`, 9 assets across equities/FX/commodities, `seed=42`, 6 rolling walk-forward folds, 755 OOS trading days from `2021-12-23` to `2024-11-13`, 1-day embargo, 1-day execution lag). They validate engine behavior and cost accounting — not live market alpha.

### 3.1 Portfolio-Level Net OOS Comparison (`strategy_comparison.csv`)

| Strategy | Gross Ret | Net Ret | Ann Vol | Gross SR | Net SR | Sortino | Max DD | Ann Turnover | TC Drag | Roll Drag | Financing Drag | Total Cost Drag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `momentum` | 7.70% | 6.96% | 8.57% | 0.91 | 0.83 | 1.21 | -7.65% | 6.98x | 0.33% | 0.12% | 0.23% | 0.69% |
| `carry` | 5.21% | 4.61% | 7.52% | 0.71 | 0.64 | 0.93 | -12.36% | 3.35x | 0.21% | 0.12% | 0.25% | 0.58% |
| `defensive` | 5.52% | 4.82% | 7.61% | 0.74 | 0.66 | 0.95 | -8.42% | 6.75x | 0.27% | 0.13% | 0.27% | 0.67% |
| **`blended`** | **9.94%** | **9.23%** | **7.86%** | **1.24** | **1.16** | **1.75** | **-7.67%** | **6.35x** | **0.30%** | **0.11%** | **0.23%** | **0.64%** |
| `benchmark_equal_weight` | 8.93% | 8.57% | 9.70% | 0.93 | 0.90 | 1.32 | -11.93% | 0.61x | 0.03% | 0.11% | 0.20% | 0.34% |
| `benchmark_inv_vol` | 9.53% | 9.04% | 10.87% | 0.89 | 0.85 | 1.25 | -13.54% | 1.70x | 0.07% | 0.13% | 0.25% | 0.45% |

**Factor diversification (OOS net return correlations):** `carry` vs `defensive` = `-0.4607`, `carry` vs `momentum` = `+0.0659`, `defensive` vs `momentum` = `+0.1120`.

### 3.2 Net OOS Performance & Turnover by Asset Class (`class_attribution.csv`)

| Strategy | Asset Class | Net Return Contrib | Cost Drag | Ann Turnover | Avg Gross Exposure | Max Gross Exposure |
|---|---|---:|---:|---:|---:|---:|
| `momentum` | `equity` | +1.24% | 0.26% | 2.53x | 0.386 | 0.821 |
| `momentum` | `fx` | +4.10% | 0.16% | 2.53x | 0.567 | 0.824 |
| `momentum` | `commodity` | +1.76% | 0.27% | 1.92x | 0.335 | 0.668 |
| `carry` | `equity` | +1.54% | 0.17% | 0.63x | 0.428 | 0.613 |
| `carry` | `fx` | +5.47% | 0.12% | 0.47x | 0.674 | 0.777 |
| `carry` | `commodity` | -2.23% | 0.29% | 2.24x | 0.349 | 0.570 |
| `defensive` | `equity` | +0.90% | 0.24% | 1.61x | 0.481 | 0.675 |
| `defensive` | `fx` | -2.42% | 0.21% | 3.90x | 0.661 | 0.814 |
| `defensive` | `commodity` | +6.51% | 0.22% | 1.24x | 0.360 | 0.512 |
| **`blended`** | **`equity`** | **+2.97%** | **0.23%** | **1.95x** | **0.366** | **0.807** |
| **`blended`** | **`fx`** | **+3.58%** | **0.17%** | **2.49x** | **0.664** | **0.821** |
| **`blended`** | **`commodity`** | **+2.59%** | **0.25%** | **1.92x** | **0.285** | **0.556** |

*(Note: `Max Gross Exposure` in daily attribution reflects intra-week drifted weights between 5-day rebalances; rebalance-time caps are strictly verified at `<= 0.80` below.)*

---

## 4. Constraint Compliance Audit (`summary.json`)

| Strategy | Rebalances | Violations | Max `|w_i|` (cap 0.30) | Max Gross (cap 1.60) | Net Band (cap [-0.50, 0.50]) | Max Class Gross (cap 0.80) | Max Ex-Ante Vol (cap 0.12) | Max Rebal TO (cap 0.35) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `momentum` | 151 | **0** | 0.3000 | 1.6000 | [-0.5000, +0.5000] | 0.8000 | 0.1042 | 0.3500 |
| `carry` | 151 | **0** | 0.3000 | 1.6000 | [-0.5000, +0.0416] | 0.7729 | 0.0929 | 0.3500 |
| `defensive` | 151 | **0** | 0.3000 | 1.6000 | [-0.5000, +0.5000] | 0.8000 | 0.1003 | 0.3500 |
| `blended` | 151 | **0** | 0.3000 | 1.6000 | [-0.5000, +0.5000] | 0.8000 | 0.0942 | 0.3500 |

---

## 5. Untested / Explicitly Out-of-Scope Capabilities

| Capability | Status | Rationale |
|---|---|---|
| Live vendor point-in-time market data adapters (Bloomberg, Refinitiv, Databento) | **Not Run / Out of Scope** | Offline research prototype; no licensed external data feed or network dependency in tests. |
| Nonlinear market impact & order-book execution simulation | **Not Run / Out of Scope** | Proportional `tc_bps` + explicit roll/financing costs are modeled; microstructure impact belongs to `optimal-execution-research-engine`. |
| Live OMS / brokerage execution & real capital deployment | **Excluded** | Strictly outside scope per `plan.md`; requires separate security, operational, legal, and risk review. |
| Remote GitHub Actions CI execution | **Defined, Local Equivalent Verified** | `../.github/workflows/cross-asset-factor-investing-engine.yml` is configured at the repository root and all its commands (`cafe validate`, `pytest`, `run_demo.py`, `cafe verify-replay`) were verified locally on Python 3.12. |

---

## 6. Archived Raw Outputs

- Multi-year walk-forward run: `reports/runs/default_cross_asset_walkforward_a2c4def8_c420020e/`
  - `summary.json`, `strategy_comparison.csv`, `class_attribution.csv`, `daily_returns.csv`, `portfolio_weights.csv`, `signal_provenance_audit.csv`, `diagnostics.jsonl`
- Small analytical reference baseline: `reports/runs/reference_fixture_small_c989fd60_b941662a/`
