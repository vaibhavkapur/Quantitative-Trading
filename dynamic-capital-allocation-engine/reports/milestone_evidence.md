# DCAE Milestone & Acceptance Evidence

- **Validated UTC:** `2026-11-27T09:50:54.536255Z`
- **Environment:** Python `3.9.6` on `macOS-26.5.1-arm64-arm-64bit`
- **Total validation runtime:** `7.11s`
- **Automated test suite:** `28` tests run, `0` failures, `0` errors
- **Required scenarios verified:** `7 / 7` (`all_passed=True`)

## Milestone Verification

| Milestone | Scope | Evidence |
|---|---|---|
| M1 — Foundation & design | Canonical strategy NAV, returns, eligibility, exposure, position & constraint contracts; Quantum-Trading contract audit & clean-room boundary | `src/dcae/domain/models.py`, `src/dcae/snapshots/publisher.py`, `docs/architecture.md`, `docs/adr/ADR-001-clean-contracts-and-solver.md` |
| M2 — Core implementation | Point-in-time estimator, equal-weight, inverse-volatility, risk-budget, and constrained ADMM + Dykstra optimizer | `src/dcae/estimators/point_in_time.py`, `src/dcae/allocators/`, `tests/unit/` |
| M3 — Integration & validation | Transaction costs, caps, correlation stress, no-feasible-solution fallback, independent constraint verifier, dual-control proposal review, hash-chained audit trail | `src/dcae/verification/verifier.py`, `src/dcae/proposals/service.py`, `src/dcae/audit/store.py`, `reports/scenario_validation_report.md` |
| M4 — Hardening & delivery | Idempotent approval-gated paper ledger, HTTP/FastAPI service layer, strictly chronological walk-forward comparison across baselines & constrained allocators | `src/dcae/paper/ledger.py`, `src/dcae/evaluation/walkforward.py`, `src/dcae/api/`, `reports/walkforward_evaluation.md` |

## Acceptance Criteria Checklist

- [x] **Allocation sum and cash reserve reconcile:** Verified independently in `ConstraintVerifier.verify()` (`allocation_sum_reconciliation`) and unit/integration/walk-forward tests.
- [x] **Every constraint is independently checked:** `ConstraintVerifier` checks sum reconciliation, cash min/max, position freshness, non-negativity, eligibility, strategy min/max, capacity, ADV liquidity, group caps, gross leverage, one-way turnover, portfolio volatility, and stressed volatility independently of the solver.
- [x] **Infeasible problems return safe no-change or cash fallback:** Verified in `infeasible_caps` scenario (`FALLBACK_NO_CHANGE` default, `FALLBACK_CASH` under `cash_only` policy).
- [x] **Decisions expose inputs and objective:** `AllocationDecision` exposes `inputs_summary`, `objective` (`ObjectiveBreakdown`), `snapshot_hash`, `config_hash`, and `diagnostics`.
- [x] **Baselines evaluated with same costs and periods:** Verified across `149` periods and `30` rebalances in `reports/walkforward_evaluation.md`.
- [x] **Clean checkout runs setup, tests, and demo with permitted fixtures:** Zero third-party dependencies required for core/tests/demo; synthetic fixtures manifest in `data/fixtures/manifest.json`.
- [x] **Architecture, dependency assumptions, known limitations, and runbook committed:** `docs/architecture.md`, `docs/api_and_schemas.md`, `docs/adr/`, `docs/runbook.md`.
- [x] **CI checks pass and raw test/evaluation evidence retained:** `reports/test_report.txt`, `reports/scenario_validation_report.json`, `reports/walkforward_evaluation.json`.
- [x] **Security, data licensing, IP, and deployment approvals reviewed:** Documented in `docs/runbook.md` and `docs/adr/ADR-001-clean-contracts-and-solver.md`; live execution is hard-disabled.
