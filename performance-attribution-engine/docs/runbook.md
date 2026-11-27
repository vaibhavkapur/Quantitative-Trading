# Operations Runbook & Controls

## 1. Operational Guardrails

- **Order execution permanently disabled:** `AttributionConfig` rejects `order_execution_enabled=true` with `OrderExecutionProhibitedError`. No broker/OMS order submission client exists in this codebase.
- **No live capital claims:** Only `SIMULATED_PROTOTYPE` and `PAPER_SANDBOX` environment modes are permitted in MVP. Live deployment requires separate legal, compliance, security, and risk sign-off.
- **Credentials & secrets:** No secrets or DSNs are committed. `PostgresAttributionStore` reads its connection string strictly from `PAE_POSTGRES_DSN`.
- **Immutable versions & dual-control:** Corrections create revised report versions (`v(N+1)`) referencing `supersedes_version_id`. Independent `REVIEWER` and `APPROVER` actions are required before a report becomes visible to `VIEWER`.

## 2. Health & Telemetry Monitoring

Poll `GET /api/v1/health`:
- `status`: must be `"OK"`. `"DEGRADED_AUDIT_INTEGRITY"` indicates a broken or tampered audit hash chain.
- `audit_chain_valid`: must be `true`.
- `order_execution_enabled`: must always be `false`.
- `metrics`: tracks `runs_total`, `runs_failed`, `revisions_total`, `reviews_total`, `approvals_total`, and `last_run_latency_ms`.

## 3. Incident Response & Safe Fallbacks

| Incident / Signal | Diagnosis | Safe Fallback & Operator Action |
|---|---|---|
| `reconciliation.status == "UNRECONCILED"` (`RESIDUAL_EXCEEDS_TOLERANCE`) | Daily gross/net residual exceeded `return_tolerance` (0.01 bps default) or `QUANTITY_MISMATCH` / `MISSING_FX_RATE` occurred. | **Do NOT approve or publish to viewers.** Inspect `reconciliation.exceptions` and `daily_summaries`, verify custodian NAV vs position MV and trades, restate inputs, and generate revised `v(N+1)` report. |
| `reconciliation.status == "DEGRADED_ZERO_NAV"` (`ZERO_NAV`) | Portfolio `start_nav + start_of_day_cf <= 0` on one or more dates (e.g., full redemption or wipeout). | Confirm whether the fund was closed/fully redeemed. Returns on the zero-NAV day are withheld (`0.0`, excluded from compounding) rather than emitting `NaN`/`Inf`. Set `fail_on_zero_nav=true` in strict batch pipelines if hard halt is preferred. |
| `MISSING_BENCHMARK` / `INCOMPLETE_BENCHMARK_SERIES` | Benchmark series is missing one or more portfolio dates or lacks FX alignment. | Portfolio returns and strategy/instrument contributions remain valid (`RECONCILED_WITH_WARNINGS`). Cumulative active return is withheld until aligned benchmark data is backfilled and a revised report version is run. |
| `STALE_FX_RATE` | No same-day FX fixing; prior date fixing carried forward or fixing age > `fx_max_staleness_hours`. | Verify FX feed health; re-run as a revised version once authoritative EOD fixing arrives. |
| `audit_chain_valid == false` | One or more rows in `pae_audit_log` were altered or deleted outside append-only APIs. | **Halt report approvals immediately.** Preserve database snapshot for forensic investigation, inspect `audit_error` (`seq_id`), and restore from verified backup. |

## 4. Security, Data Licensing & IP Review Checklist

- [x] Synthetic fixtures (`data/fixtures/manifest.json`) carry explicit `entitlement_tag: PERMITTED_SYNTHETIC` and SHA-256 digests.
- [x] Zero third-party credentials or API keys in source or `config/default_config.json`.
- [x] Zero code copied from unconfirmed external/personal repositories; clean adapter contracts isolated in `src/pae/adapters/contracts.py`.
- [ ] **Pre-sandbox gate (pending SAAT decision):** Confirm actual SAAT asset classes, market-data/benchmark vendor licenses, corporate repository ownership, and target PostgreSQL/IdP infrastructure before connecting real feeds.
