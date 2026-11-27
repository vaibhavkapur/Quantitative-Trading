# Acceptance Criteria & Milestone Evidence Traceability Matrix

This document attaches concrete code, test, and report evidence to every milestone and acceptance criterion in [plan.md](../plan.md).

## 1. Acceptance Criteria Checklist (`plan.md` Section 6)

- [x] **Attribution sums to total within documented tolerance or shows explicit residual.**
  - *Implementation:* `src/pae/engine/contribution.py`, `src/pae/engine/reconciliation.py` (`return_tolerance = 1e-6` / `0.01 bps`, Carino linking identity verified to `< 1e-10`).
  - *Evidence:* `tests/unit/test_hand_worked_portfolios.py`, `tests/scenarios/test_required_scenarios.py::test_scenario_7_rounding_residual`, `reports/validation_report.md` (`+0.000000 bps` residual on v1, v2, and hand-worked portfolios).
- [x] **Return methodology is versioned.**
  - *Implementation:* `config/default_config.json` (`methodology_version: "pae-v1.0.0"`, `config_version: "1.0.0"`, deterministic `config_hash`), embedded in every `AttributionReport` and API `/api/v1/health` response.
  - *Evidence:* `tests/unit/test_domain_and_returns.py::test_config_hash_deterministic`, `docs/assumptions_and_methodology.md`.
- [x] **Benchmarks use aligned dates/currency.**
  - *Implementation:* `src/pae/engine/returns.py` converts benchmark currency via point-in-time FX (`(1+r_loc)*(1+r_fx)-1`) and withholds cumulative active returns with `MISSING_BENCHMARK` / `BENCHMARK_CURRENCY_MISMATCH` exceptions on gaps.
  - *Evidence:* `tests/unit/test_domain_and_returns.py::test_benchmark_currency_alignment_with_fx_conversion`, `tests/scenarios/test_required_scenarios.py::test_scenario_5_missing_benchmark`.
- [x] **Every contribution links to positions/trades.**
  - *Implementation:* `DailyPositionContribution` and `AggregatedContribution` record `position_ids` and `trade_ids`; `ReconciliationReport` enforces `every_contribution_links_to_positions`; CSV export writes `position_id` and `trade_ids` per row.
  - *Evidence:* `tests/scenarios/test_required_scenarios.py::test_scenario_6_rebalance`, `reports/demo_exports/*.csv`.
- [x] **Corrections create revised report versions.**
  - *Implementation:* `SQLiteAttributionStore` + `AttributionService.run_attribution` create immutable `v(N+1)` reports referencing `supersedes_version_id`, compute `revision_diff`, mark prior status `SUPERSEDED` without mutating prior payload/hash, and log to the SHA-256 hash-chained audit trail.
  - *Evidence:* `tests/integration/test_pipeline_storage_and_api.py`, `tests/scenarios/test_required_scenarios.py::test_scenario_3_fee_corrections`, `reports/validation_report.md` (`v2` supersedes `v1`).
- [x] **A clean checkout can run the documented setup, tests and demo with permitted fixtures.**
  - *Implementation:* Zero third-party dependencies required (`Python 3.9+` stdlib only); reproducible scripts `scripts/setup.sh`, `scripts/validate.sh`, `scripts/demo.sh`, `scripts/reproduce.sh`; synthetic fixtures & SHA-256 manifest in `data/fixtures/manifest.json`.
  - *Evidence:* Clean `bash scripts/reproduce.sh` execution logged in `reports/validation_report.md` and `reports/test_report.txt`.
- [x] **Architecture, dependency assumptions, known limitations and runbook are committed.**
  - *Evidence:* `docs/architecture.md`, `docs/ADRs.md`, `docs/assumptions_and_methodology.md`, `docs/api_and_schema.md`, `docs/runbook.md`, `docs/unsupported_features.md`.
- [x] **Relevant CI checks pass and raw evaluation or test evidence is retained.**
  - *Evidence:* `reports/test_report.txt` (25/25 passing tests), `reports/validation_report.json`, `reports/validation_report.md`, and `reports/demo_exports/` (JSON, MD, CSV, HTML for v1, v2, and hand-worked cases).
- [x] **Security, data licensing, IP and any deployment approval requirements are reviewed.**
  - *Evidence:* Clean-room adapter boundary (`src/pae/adapters/contracts.py`, ADR-004), permanent order-execution prohibition (`OrderExecutionProhibitedError`), RBAC + dual-control approval enforcement (`src/pae/security/rbac.py`), synthetic entitlement manifest, and pre-sandbox checklist in `docs/runbook.md`.

## 2. Mandatory Scenarios Matrix (`plan.md` Section 5)

| Required Scenario | Automated Test | Verified Behavior |
|---|---|---|
| External cash flow | `tests/scenarios/test_required_scenarios.py::test_scenario_1_external_cash_flow` | Large SOD subscription & EOD redemption leave TWR exact (`+0.50%`) and residual `0.0`. |
| Zero nav | `tests/scenarios/test_required_scenarios.py::test_scenario_2_zero_nav` | Capital base `0.0` emits `CRITICAL` `ZERO_NAV` (`DEGRADED_ZERO_NAV`, finite outputs) or raises `ZeroNavError` in strict mode. |
| Fee corrections | `tests/scenarios/test_required_scenarios.py::test_scenario_3_fee_corrections` | Retroactive rebate emits `FEE_CORRECTION_APPLIED`, creates immutable `v2` superseding `v1` with `revision_diff`. |
| Currency conversion | `tests/scenarios/test_required_scenarios.py::test_scenario_4_currency_conversion` | Exact inverse-quote `USD/JPY` Karnosky-Singer FX decomposition + `MISSING_FX_RATE` detection. |
| Missing benchmark | `tests/scenarios/test_required_scenarios.py::test_scenario_5_missing_benchmark` | Missing benchmark date emits `MISSING_BENCHMARK` & `INCOMPLETE_BENCHMARK_SERIES`, withholds cumulative active return, preserves portfolio reconciliation. |
| Rebalance | `tests/scenarios/test_required_scenarios.py::test_scenario_6_rebalance` | Intraday full exit + partial trim + new entry reconcile cleanly to `0.0` residual with exact trade/position provenance and TCA attribution. |
| Rounding residual | `tests/scenarios/test_required_scenarios.py::test_scenario_7_rounding_residual` | Sub-tolerance custodian rounding (`0.0003 bps`) passes with explicit residual surfaced; above-tolerance (`5.0 bps`) flags `UNRECONCILED` + `RESIDUAL_EXCEEDS_TOLERANCE` `ERROR`. |

## 3. Milestones Summary (M1–M4)

- [x] **M1 — Foundation and design:** Canonical contracts, return/cash-flow/benchmark conventions, config versioning, PostgreSQL DDL, ADRs, unit tests.
- [x] **M2 — Core implementation:** Daily position/strategy/instrument/sector/currency contributions, exact FX decomposition, Carino linking, strict Gross-vs-Net separation.
- [x] **M3 — Integration and validation:** Brinson-Fachler/BHB sector analysis, OLS risk-factor attribution (non-causal), Perold Implementation Shortfall TCA linked to trades, upstream adapters.
- [x] **M4 — Hardening and delivery:** Reconciliation report with residuals & exceptions, 3 hand-worked portfolio verifications, immutable report versioning, RBAC & dual-control approvals, REST API, zero-CDN SVG dashboard, full documentation & retained evidence.
