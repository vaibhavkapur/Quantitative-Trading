# Operational Runbook & Governance Controls

## 1. Standard Operator Workflow (Paper / Sandbox)

1. **Snapshot Ingestion:** Publish a validated `FundSnapshot` (`POST /v1/snapshots` or `ProposalService.register_snapshot`). Verify `position_age_hours <= max_position_staleness_hours` and inspect `provenance.entitlement`.
2. **Proposal Generation:** Proposer (`pm_*`) submits `POST /v1/allocations/propose` with an explicit `idempotency_key`. Inspect:
   - `decision.status` (`OPTIMAL` vs `FALLBACK_NO_CHANGE` / `FALLBACK_CASH`)
   - `feasibility_report.feasible` and every check in `feasibility_report.checks`
   - `decision.objective` (expected return, vol, stressed vol, turnover, estimated USD cost)
   - `decision.inputs_summary.warnings` (dropped future observations, PSD repair, correlated drawdown stress)
3. **Independent Review (Dual Control):** A distinct risk reviewer (`risk_*`, `reviewer_id != proposer_id`) inspects the proposal and approves or rejects via `POST /v1/proposals/{id}/review` before `expires_at_utc`.
4. **Paper Execution:** An operations actor (`ops_*`) applies the approved proposal via `POST /v1/proposals/{id}/execute-paper` with an idempotency key. Re-sending the same key safely replays the receipt without double-charging costs.
5. **Audit Verification:** Periodically run `GET /v1/audit` or `SQLiteAuditStore.verify_chain()` and archive JSONL exports.

## 2. Failure Modes & Operator Response

| Condition | Automated System Behavior | Operator Response |
|---|---|---|
| **Stale positions (`STALE_DATA`)** | Hard rejection (`409`); no proposal created. | Refresh position feed from paper ledger / custodian, confirm `updated_at_utc`, and resubmit. |
| **Non-PSD covariance** | Under `repair` policy: clips eigenvalues to `psd_min_eigenvalue`, logs warning and Frobenius delta. Under `reject` policy: raises `NON_PSD_COVARIANCE`. | Inspect `psd_repair_frobenius_delta`; if large (`> 1e-2`), investigate upstream risk model / missing observations before approving. |
| **Zero volatility / Insufficient history** | Under `exclude` policy: blocks new capital to affected strategy and logs `excluded_reasons`. Under `reject` policy: fails fast. | Verify whether strategy feed is stuck (flat NAV) or newly launched; keep excluded until minimum history is accumulated. |
| **Infeasible constraints (`FALLBACK_NO_CHANGE` / `FALLBACK_CASH`)** | Never emits an infeasible target; returns safe hold-current (`no_change`) or 100% cash (`cash_fallback`) with `infeasibility_reasons`. | Review conflicting caps/floors/turnover/ADV limits in `infeasibility_reasons`; adjust config through change control or hold positions safely. |
| **Correlated drawdowns** | Automatically activates stressed covariance overlay (`correlation_stress_active=True`) when >=50% (at least 2) of eligible strategies breach `drawdown_stress_trigger`. | Review elevated cash buffer and stressed volatility check; confirm risk committee alignment before approving rebalance. |
| **Audit hash-chain failure (`AUDIT_INTEGRITY_ERROR`)** | Raises `500` on `/health` and `/v1/audit`. | Halt all paper executions immediately; inspect SQLite/PostgreSQL access logs and restore from verified backup. |

## 3. Security, Data Licensing, IP & Deployment Controls

- **Secrets management:** `config/default_config.json` contains zero credentials. PostgreSQL connections read `DCAE_POSTGRES_DSN` strictly from environment variables (`PostgresAuditStore`).
- **Role separation:** `enforce_dual_control=True` prevents proposer self-approval; production deployment should bind actor identities to authenticated SSO/RBAC principals rather than caller-supplied strings.
- **Data provenance & licensing:** All committed fixtures under `data/fixtures/` are synthetically generated (`entitlement="synthetic_internal"`) with checksums in `data/fixtures/manifest.json`. Do not ingest third-party market or fund data without written entitlement verification.
- **Code IP boundary:** Implemented cleanly without copying unlicensed external/personal repositories.
- **Live deployment gate:** Automated live reallocation is disabled in code (`LiveExecutionProhibitedError`) and configuration validation (`live_execution_enabled=False`). Any move beyond approved paper/sandbox requires separate legal, regulatory, broker, security, and risk-committee sign-off.

## 4. Explicit List of Unsupported Features (MVP Scope)

- Live broker / OMS order routing or automated live capital transfers
- Multi-currency portfolios and FX hedging
- Short strategy allocations or fund-level borrowing (`gross_leverage > 1.0` default off)
- Nonlinear market-impact cost curves or intraday execution slicing
- Real-time streaming market data ingestion or distributed multi-node solver scheduling
- Authentication/SSO middleware (must be fronted by API gateway / IAM in sandbox deployment)
