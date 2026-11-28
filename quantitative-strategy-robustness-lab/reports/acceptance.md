# Acceptance Evidence Report — Quantitative Strategy Robustness Lab

- **Evaluation Date:** `2026-11-28`
- **Environment:** macOS (arm64), Python `3.12.15`, NumPy `2.5.5`, SciPy `1.18.1`, pandas `2.3.3`, pytest `8.4.2`
- **Configuration:** `config/default_policy.json` (`policy_version: 1.0.0`, `seed: 42`, SHA-256 `43727f3390f894e0a59231cfed79907f5c0c63f3f2d2e4ced816c5cb510eca75`)
- **Validation Command:** `bash scripts/ci_check.sh` -> **27/27 pytest assertions passed; 5/5 demo scenarios replayed with exact logical equality; CLI `verify-registry` and `replay` passed.**
- **Data Provenance:** All measured numbers below come from deterministic **synthetic fixtures** (`provenance="synthetic_fixture"`) and hand-calculated analytical oracles; no live market feeds or real trading records were used.

## 1. Project-Specific Acceptance Criteria Matrix (`plan.md` Section 5)

| Acceptance Criterion | Status | Evidence (Tests, Code & Artifacts) |
|---|---|---|
| **Track all trials** (including failures) | **PASS** | `src/robustness_lab/registry.py` logs `COMPLETED`, `FAILED`, `REJECTED`, `ABORTED` in an append-only SHA-256 hash chain; `tests/test_registry.py` (5 tests) verifies all statuses, mandatory `failure_reason`, tamper detection, and deletion detection; `reports/baseline/known_signal.json` records `9` logged trials (`8` completed + `1` `FAILED`, 100% grid coverage). |
| **Publish uncertainty and OOS results** | **PASS** | `src/robustness_lab/bootstrap.py`, `walk_forward.py`, and `reporting.py` publish stationary/circular block bootstrap 95% CIs (Sharpe, annualized mean return, max drawdown) and purged walk-forward fold tables with exact split boundaries and benchmark comparison in `reports/baseline/*.md` and `*.json`. |
| **Reject promotion when evidence incomplete** | **PASS** | `src/robustness_lab/promotion.py` enforces `evidence_completeness` as a hard required gate; verified directly by `tests/test_pipeline_and_promotion.py::test_promotion_strictly_rejects_when_any_evidence_is_incomplete` and end-to-end on `insufficient_sample_experiment` (`reports/baseline/insufficient_sample.json`: `decision="REJECT"`, `evidence_complete=false`, 4 missing diagnostics). |
| **All required test scenarios pass** (or explicitly marked not run) | **PASS** | 27/27 tests in `tests/` pass via `bash scripts/ci_check.sh`. Untested/out-of-scope items (real market feeds, concurrent DB writers, sibling repo live imports) are explicitly listed in Section 4 below and in `README.md`. |
| **README documents setup, demo, validation, assumptions, and limitations** | **PASS** | `README.md` documents clean setup, CLI & demo commands, measured synthetic results explicitly labeled `synthetic_fixture`, methodology assumptions, and limitations. |
| **No real transactions, customer data, credentials, or proprietary employer/SAAT artifacts** | **PASS** | Verified across `src/`, `tests/`, `scripts/`, `config/`, `data/fixtures/`, `docs/`, and `reports/`; all inputs are synthetic or hand-calculated fixtures. |

## 2. Test & Validation Matrix Evidence (`plan.md` Section 4)

| Test Area | Required Scenario | Automated Tests & Measured Evidence |
|---|---|---|
| **Domain correctness** | Null strategies, correlated trials, and known signal | `tests/test_contracts_and_costs.py::test_hand_calculated_reference_oracle` (exact match `< 1e-10` vs analytical oracle); `tests/test_dsr_and_pbo.py` (5 tests: Mertens PSR exact check, AR(1) Lo VIF inflation `> 1.8`, 24-trial null strategies yield `DSR < 0.85` & `PBO > 0.35`, 12 correlated trials collapse $N_{\text{eff}} < 1.5$ vs `> 9.5` independent, known signal achieves `DSR > 0.98` & `PBO < 0.10`). |
| **Integration and resilience** | Duplicate or cherry-picked trial detection | `tests/test_integrity_and_cherry_picking.py` (3 tests) + `reports/baseline/cherry_picked.json`: detects sequence gaps (`[3, 5, 6, 8]`), low grid coverage (`5/9 = 55.6% < 85%`), exact duplicates (`cherry_seq4` & `cherry_seq7`), near-duplicates (`corr >= 0.995`), and return-series SHA-256 mismatches -> verdict **`REJECT`**. |
| **Boundary and failure behavior** | Insufficient sample and extreme outliers | `tests/test_boundaries_and_outliers.py` (3 tests) + `reports/baseline/insufficient_sample.json` (`24` obs -> `evidence_complete=false`, `REJECT`) + `reports/baseline/extreme_outlier.json` (2 spikes, max robust `|z| = 48.28`, excess kurtosis `> 150` -> `outlier_dominance_check` fails, `REJECT`). |
| **Reproducibility** | Versioned inputs/config + fixed seed | `scripts/run_demo.py`, `tests/test_pipeline_and_promotion.py::test_end_to_end_known_signal_promotes_and_replays_deterministically`, and CLI `replay` verify identical deterministic `run_id` and exact logical equality across repeated runs (`replay_matched: true` for all 5 baseline scenarios). |
| **Operational review** | Missing config, unsupported inputs, tampered registry, restart/rerun | `tests/test_contracts_and_costs.py` (missing/incomplete policy JSON, unsupported frequency, NaN, non-monotonic timestamps, return `< -1.0`), `tests/test_pipeline_and_promotion.py::test_cli_evaluate_verify_replay_and_error_exit_codes` (CLI exits `2` on tampered registry or missing policy — zero silent successes), and `docs/runbook.md` recovery table. |

## 3. Milestone Completion Summary (M1–M4)

- **M1 — Foundation and reference fixtures:** Complete (`pyproject.toml`, `requirements.lock`, `config/default_policy.json`, `models.py`, `costs.py`, `registry.py`, `data/fixtures/` valid/invalid/boundary + `hand_calculated_reference.json`, `docs/contracts.md`, `docs/architecture.md`, `docs/adr/0001-immutable-registry-and-domain-boundaries.md`).
- **M2 — Core domain implementation:** Complete (`statistics.py`, `pbo.py`, isolated pure domain API, unit tests for null strategies, correlated trials, known signal, and analytical PSR/Lo-VIF checks).
- **M3 — Integration and evaluation:** Complete (`bootstrap.py`, `walk_forward.py`, `perturbation.py`, `integrity.py`, `pipeline.py`, structured JSON event logs, counters, deterministic replay, raw baseline outputs in `reports/baseline/`).
- **M4 — Hardening, benchmarking and delivery:** Complete (`promotion.py`, `reporting.py`, `cli.py`, boundary/outlier regression tests, `../.github/workflows/quantitative-strategy-robustness-lab.yml`, `scripts/ci_check.sh`, `docs/integration.md`, `docs/runbook.md`, `README.md`, `reports/acceptance.md`).

## 4. Explicitly Untested / Out-of-Scope Capabilities

- **Live market data feeds & broker execution:** Not tested and excluded by project scope.
- **Concurrent multi-process registry writers:** Single-writer offline JSONL ledger only; distributed/database-backed locking is not implemented or tested.
- **Sibling repository live imports:** Sibling projects are standalone peer packages at the repository root; cross-project integration is covered via documented adapter contracts in `docs/integration.md` rather than runtime cross-imports.
- **Remote GitHub Actions execution:** `../.github/workflows/quantitative-strategy-robustness-lab.yml` is configured at the repository root and mirrors `scripts/ci_check.sh`, which was executed locally on macOS Python 3.12.15.
