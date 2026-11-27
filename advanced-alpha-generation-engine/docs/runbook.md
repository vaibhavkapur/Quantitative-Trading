# Operator Runbook — Advanced Alpha Generation Engine

## 1. Clean Checkout Setup & Verification

```bash
./scripts/setup.sh
.venv/bin/python scripts/demo.py
.venv/bin/python scripts/validate.py
```

- `scripts/setup.sh` installs pinned dependencies from `uv.lock` (Python 3.12) and generates deterministic synthetic PIT fixtures in `data/fixtures/`.
- `scripts/demo.py` runs M1–M4 end-to-end and writes reports to `reports/` and paper-only exports to `reports/paper_exports/`.
- `scripts/validate.py` verifies all 7 failure-path scenarios, runs `pytest` and `ruff`, and generates `reports/acceptance_evidence.md`.

## 2. Promotion & Paper-Export Governance Workflow

1. **Automated gate evaluation**: Every candidate signal or ML baseline is evaluated via `WalkForwardEvaluator` and registered via `AlphaCatalog.register_batch()`. Both passing and failing trials are appended permanently to `reports/catalog/trials.jsonl`.
2. **Hard gate enforcement**: Candidates failing any policy gate (OOS IC, ICIR, BH-adjusted p-value, turnover, sector exposure, peer duplication, net Sharpe at base cost, or coefficient stability) are marked `rejected` and **cannot be promoted** (`PromotionGateError`).
3. **Human approval required**: Promoting a passing candidate requires explicit `approver` and `approval_ticket` metadata recorded in `catalog.json`.
4. **Paper-only export**: `PaperSignalExporter` exports only `promoted` signals with `mode: "paper_only"`, `execution_enabled: false`, UTC `as_of_utc` causality check, and SHA-256 provenance digest. Any attempt to pass `request_live_execution=True` or `execution_enabled=True` raises `ExecutionDisabledError`.

## 3. Health, Failure, and Decision Metrics

Each pipeline run emits structured observability metrics to `reports/engine_metrics.json`:
- `status`: `"healthy"` or `"degraded"`
- `counters`: `datasets_validated`, `factors_configured`, `signals_evaluated`, `models_evaluated`, `trials_recorded`, `candidates_promoted`, `candidates_rejected`, `paper_exports_written`, `errors_total`
- `stage_durations_ms`: Latency per milestone stage (`m1_pit_data_load_and_validate` through `m4_catalog_and_paper_export`)
- `decisions`: Audit trail of every `catalog_promotion` decision (`promoted` vs `rejected` with gate failure reasons)
- `failures`: Structured error records (e.g., non-fatal `OPTIONAL_DEPENDENCY_ERROR` if optional `xgboost` runtime is absent)

## 4. Operator Incident Response & Safe Fallbacks

| Failure Code | Meaning | Operator Action & Safe Fallback |
|---|---|---|
| `LOOK_AHEAD_VIOLATION` | Signal or input timestamp exceeds `as_of_utc` or precedes source availability | **Halt pipeline immediately.** Do not export signals. Inspect offending timestamps in error `details` and fix upstream ETL/factor lag. |
| `SURVIVORSHIP_VIOLATION` | Delisted symbols missing from history or bars present post-exit | **Halt research run.** Reconcile universe membership table against vendor delisting feed before trusting any backtest metric. |
| `CORPORATE_ACTION_INTEGRITY_ERROR` | Unadjusted split discontinuity detected in `adjusted_close` | **Halt pipeline.** Verify corporate action adjustment pipeline; never evaluate price-based factors on unadjusted series. |
| `MISSING_SYMBOL_ERROR` | Active symbol coverage dropped below `min_symbol_coverage_ratio` | Inspect missing symbols/dates. Fall back to holding prior paper weights (no rebalance) until feed recovers. |
| `PROMOTION_GATE_ERROR` / `FACTOR_DUPLICATION_ERROR` | Candidate failed validation or duplicates promoted signal | **Expected governance block.** Retain negative trial in `trials.jsonl`; do not override thresholds without formal committee review and config version bump. |
| `OPTIONAL_DEPENDENCY_ERROR` | Optional boosting runtime (`xgboost` / `libomp`) missing | Safe degradation: core deterministic baselines (`EqualRankComposite`, `Ridge`, `sklearn GBDT`) still run. Install `brew install libomp` if XGBoost is required. |
| `EXECUTION_DISABLED_ERROR` | Attempted live execution in research engine | **Hard security block.** Live capital deployment requires separate legal/regulatory, broker, security, and risk sign-off outside this MVP repository. |

## 5. Security, Data Licensing & IP Checklist

- **No secrets in repo**: `config/*.json` contains versioned non-secret policies only. Future vendor credentials must be injected via secret manager / environment variables outside source control.
- **Data provenance**: `data/fixtures/manifest.json` records `source_type: "synthetic"`, `entitlements`, `license`, and SHA-256 checksums verified on every load.
- **Background IP**: External personal repos (`Quantum-Trading`, `Trading-Software`) were not copied; see `docs/foundation_audit.md` and `docs/adr/ADR-001-clean-room-adapter-contracts.md`.
- **Role separation**: Researchers submit candidates; independent approvers (`approver`, `approval_ticket`) authorize catalog promotion; order execution remains disabled.
