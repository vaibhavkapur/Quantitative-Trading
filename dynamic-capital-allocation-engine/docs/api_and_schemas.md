# API & Domain Schema Reference

## 1. Canonical Domain Schemas (`schema_version = "1.0.0"`)

All timestamps must be timezone-aware UTC ISO-8601 strings (canonical format ending in `Z`). All monetary fields are denominated in `USD`.

### `FundSnapshot`
- `snapshot_id` (string): stable unique identifier.
- `schema_version` (string): `"1.0.0"`.
- `as_of_utc` (UTC ISO-8601): decision timestamp.
- `published_at_utc` (UTC ISO-8601): snapshot publication timestamp.
- `provenance`: `{ source, entitlement, generator_version, seed, notes }`.
- `positions` (`PositionSnapshot`): `{ as_of_utc, updated_at_utc, currency, total_nav_usd, cash_usd, strategy_notional_usd }`. Must reconcile `cash_usd + sum(strategy_notional_usd) == total_nav_usd`.
- `strategies` (list of `StrategySnapshot`):
  - `strategy_id`, `name`, `currency`, `eligible`, `eligibility_reason`
  - `exposure`: `{ strategy_group, asset_class, gross_exposure, net_exposure, market_beta, adv_usd, max_capacity_usd, transaction_cost_bps, liquidity_days_to_exit }`
  - `returns`: strictly chronological list of `{ event_at_utc, available_at_utc, nav_usd, period_return }` where `available_at_utc >= event_at_utc`.
  - Optional overrides: `min_weight`, `max_weight`, `risk_budget`, `current_drawdown`.
- Optional `external_covariance` & `external_covariance_ids`: square matrix override (used for external risk model integration and non-PSD fault injection).

### `AllocationDecision`
- `decision_id`, `allocator_name`, `snapshot_id`, `snapshot_hash`, `config_version`, `config_hash`, `as_of_utc`
- `status`: `OPTIMAL | CONVERGED_APPROX | FALLBACK_NO_CHANGE | FALLBACK_CASH | REJECTED`
- `target_weights`, `cash_weight`, `current_weights`, `current_cash_weight`
- `objective` (`ObjectiveBreakdown`): `expected_return_annual`, `portfolio_volatility_annual`, `stressed_volatility_annual`, `variance_penalty`, `transaction_cost_drag`, `turnover_l2_penalty`, `net_objective`, `one_way_turnover`, `estimated_cost_usd`
- `inputs_summary`, `diagnostics`, `infeasibility_reasons`, `fallback_applied`, `feasibility_report`

### `RebalanceProposal`
- `proposal_id`, `idempotency_key`, `status` (`PENDING_APPROVAL | APPROVED | REJECTED | EXPIRED | PAPER_EXECUTED`)
- `proposer_id`, `reviewer_id`, `review_notes`, `created_at_utc`, `updated_at_utc`, `expires_at_utc`
- `decision`, `feasibility_report`, `trades[]`, `cash_delta_usd`, `target_cash_usd`, `requires_manual_approval`

## 2. HTTP API Endpoints (`dcae.api.http_server` & `dcae.api.fastapi_app`)

| Method | Path | Purpose | Status Codes |
|---|---|---|---|
| `GET` | `/health` | Service health, execution policy flags, and audit hash-chain status | `200` |
| `GET` | `/v1/metrics` | Operational counts (proposals, approvals, fallbacks, latency p50/p95) | `200` |
| `GET` | `/v1/audit` | Audit events list and hash-chain verification summary | `200`, `500` |
| `POST` | `/v1/snapshots` | Validate and publish a `FundSnapshot` | `201`, `409`, `422` |
| `POST` | `/v1/allocations/propose` | Run allocator, independently verify constraints, create proposal | `201`, `409`, `422` |
| `GET` | `/v1/proposals` | List all persisted proposals | `200` |
| `GET` | `/v1/proposals/{id}` | Retrieve a proposal with full feasibility report and trades | `200`, `404` |
| `POST` | `/v1/proposals/{id}/review` | Approve or reject a proposal (enforces dual control and TTL) | `200`, `403`, `409` |
| `POST` | `/v1/proposals/{id}/execute-paper` | Idempotently apply an `APPROVED` proposal to the paper ledger | `200`, `403`, `409` |

## 3. Structured Error Catalog

Every error response returns `{ "error_code": "...", "message": "...", "context": { ... } }`:

- `VALIDATION_ERROR` (`422`): Schema, bounds, reconciliation, or timestamp violation.
- `POINT_IN_TIME_VIOLATION` (`422`): Position timestamp lies after `as_of_utc`.
- `STALE_DATA` (`409`): Position snapshot exceeds `max_position_staleness_hours`.
- `INSUFFICIENT_HISTORY` (`422`): Strategy has fewer than `min_history_observations` (when policy=`reject`).
- `ZERO_VOLATILITY` (`422`): Strategy has degenerate zero volatility (when policy=`reject`).
- `NON_PSD_COVARIANCE` (`422`): Non-PSD covariance detected (when policy=`reject`).
- `DUAL_CONTROL_VIOLATION` (`403`): Proposer attempted to self-approve/review.
- `APPROVAL_REQUIRED` (`403`): Paper execution attempted on non-`APPROVED` proposal.
- `LIVE_EXECUTION_PROHIBITED` (`403`): Execution requested with `mode != "paper"`.
- `INVALID_PROPOSAL_STATE` (`409`): Proposal expired, missing, or already transitioned.
- `AUDIT_INTEGRITY_ERROR` (`500`): Hash chain mismatch or tampered audit payload detected.
