# API Reference & Domain Schemas

## 1. Authentication & Role Headers

All mutating endpoints enforce role separation via HTTP headers (for local prototype evaluation; replace with OIDC/mTLS identity tokens prior to shared sandbox deployment):

- `X-Actor`: Operator or approver identifier (audit trail attribution).
- `X-Role`: One of `VIEWER` (default, read-only), `OPERATOR` (event/OMS/market/reconciliation execution), or `APPROVER` (required for authorizing reconciliation `AdjustmentEvent` postings).

## 2. REST Endpoints

| Method | Path | Role Required | Description |
|---|---|---|---|
| `GET` | `/` | Public/Viewer | Interactive Portfolio, Subledger, Reconciliation & COD Web UI |
| `GET` | `/api/v1/health` | `VIEWER` | System status, hash-chain verification, control flags, and operational metrics |
| `GET` | `/api/v1/master` | `VIEWER` | Versioned fund, account, and instrument master registry |
| `GET` | `/api/v1/events` | `VIEWER` | Immutable append-only event log (optional `?as_of=<ISO8601-UTC>`) |
| `POST` | `/api/v1/events` | `OPERATOR` / `APPROVER` | Append validated immutable ledger event (`409 Conflict` on duplicate `idempotency_key`) |
| `GET` | `/api/v1/journals` | `VIEWER` | Balanced double-entry journal entries and per-currency trial balance |
| `GET` | `/api/v1/portfolio/valuation` | `VIEWER` | Consolidated MTM valuation, P&L explainability, consolidation check, unpriced exceptions |
| `GET` | `/api/v1/portfolio/lots` | `VIEWER` | Open accounting lots (filterable by `?account_id=&instrument_id=`) |
| `POST` | `/api/v1/markets/snapshots` | `OPERATOR` / `APPROVER` | Register versioned market price & FX rate snapshot |
| `POST` | `/api/v1/oms/paper-fills` | `OPERATOR` / `APPROVER` | Ingest paper OMS execution report with automatic deduplication |
| `POST` | `/api/v1/oms/live-orders` | Blocked (`403`) | Hard safety guard: always returns `CONTROL_VIOLATION` while live execution is disabled |
| `GET` | `/api/v1/reconciliation/runs` | `VIEWER` | List read-only broker reconciliation runs and break records |
| `POST` | `/api/v1/reconciliation/run` | `OPERATOR` / `APPROVER` | Execute read-only broker reconciliation (optional statement payload) |
| `POST` | `/api/v1/reconciliation/breaks/{break_id}/resolve` | `APPROVER` | Append an audited `AdjustmentEvent` resolving an open break without mutating history |
| `GET` | `/api/v1/reports/cod` | `VIEWER` | Generate Close-of-Day report (`?format=json` or `?format=markdown`) |
| `POST` | `/api/v1/recovery/verify` | `VIEWER` | Clear materialized read-model cache and prove identical state hash on full replay |

## 3. Immutable Event Contracts (`LedgerEvent` Discriminated Union)

All numeric quantities, prices, fees, and FX rates use exact fixed-point `Decimal` strings. All timestamps require explicit UTC timezone offsets.

1. `TRADE_FILL`: `account_id`, `instrument_id`, `order_id`, `execution_id`, `side` (`BUY`|`SELL`), `quantity`, `price`, `currency`, `fee`, `venue`.
2. `CASH_TRANSFER`: `account_id`, `currency`, `amount`, `direction` (`DEPOSIT`|`WITHDRAWAL`), `counterparty_account_id`, `description`.
3. `STRATEGY_ALLOCATION`: `from_account_id`, `to_account_id`, `currency`, `amount`, `allocation_reason`.
4. `FX_CONVERSION`: `account_id`, `sell_currency`, `sell_amount`, `buy_currency`, `buy_amount`, `fx_rate`, `fee_currency`, `fee_amount`.
5. `FEE`: `account_id`, `instrument_id?`, `currency`, `amount`, `fee_category`, `is_late_posting`, `description`.
6. `CORPORATE_ACTION`: `account_id`, `instrument_id`, `action_type` (`SPLIT`|`CASH_DIVIDEND`), `ratio_numerator`, `ratio_denominator`, `cash_per_share`, `withholding_tax_rate`, `currency`.
7. `ADJUSTMENT`: `account_id`, `adjustment_type` (`CASH`|`POSITION`), `instrument_id?`, `currency`, `cash_delta`, `quantity_delta`, `unit_cost`, `reconciliation_break_id`, `approved_by`, `reason`.

## 4. Structured Error Codes

| HTTP Status | `error_code` | Meaning |
|---|---|---|
| `400` | `VALIDATION_ERROR` | Malformed input or invalid workflow state transition |
| `403` | `AUTHORIZATION_ERROR` | Caller `X-Role` lacks permission for the requested action |
| `403` | `CONTROL_VIOLATION` | Blocked by operational safety policy (e.g., live execution disabled) |
| `409` | `DUPLICATE_EVENT` | Duplicate `idempotency_key` or `event_id` safely rejected |
| `409` | `IMMUTABILITY_VIOLATION` | Attempted mutation or deletion of immutable event history |
| `422` | `MASTER_DATA_ERROR` | Unknown account/instrument or currency/account-type mismatch |
