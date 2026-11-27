# API & Schema Specification

## 1. REST API Endpoints

Available via the built-in zero-dependency HTTP server (`pae.api.http_server`) and the optional FastAPI ASGI app (`pae.api.fastapi_app`).

### Authentication & Role Headers

Callers pass identity headers (credentials/IdP tokens remain outside repository code):
- `X-Actor-Id`: stable caller identifier (e.g. `analyst-01`)
- `X-Actor-Role`: one of `VIEWER`, `ANALYST`, `REVIEWER`, `APPROVER`, `ADMIN`

### Endpoints

| Method | Path | Required Permission | Description |
|---|---|---|---|
| `GET` | `/api/v1/health` | Public / Ops | Service health, `methodology_version`, `config_hash`, `order_execution_enabled=false`, audit chain status, and latency/run metrics. |
| `POST` | `/api/v1/attribution/run` | `RUN_ATTRIBUTION` or `REVISE_REPORT` | Validate `AttributionInputBatch`, run pipeline, and persist an immutable report version (`v1` or `v(N+1)` superseding prior version with `revision_diff`). |
| `GET` | `/api/v1/reports` | `READ_REPORT` | List stored report versions (filterable by `portfolio_id`, `report_id`). `VIEWER` sees `APPROVED` reports only. |
| `GET` | `/api/v1/reports/{version_id}` | `READ_REPORT` | Fetch full immutable `AttributionReport` JSON and workflow metadata. |
| `GET` | `/api/v1/reports/{version_id}/export?format={json\|md\|csv\|html}` | `READ_REPORT` | Export immutable report in JSON, Markdown, CSV (position-level provenance), or standalone HTML. |
| `POST` | `/api/v1/reports/{version_id}/review` | `REVIEW_REPORT` | Transition `DRAFT -> REVIEWED` (enforces `reviewer_id != author_id`). |
| `POST` | `/api/v1/reports/{version_id}/approve` | `APPROVE_REPORT` | Transition `REVIEWED -> APPROVED` (enforces `approver_id != author_id` and `approver_id != reviewer_id`). |
| `GET` | `/api/v1/audit` | `READ_AUDIT` | Retrieve hash-chained audit log and live chain verification result. |
| `GET` | `/dashboard` | Local UI | Interactive zero-CDN HTML index of stored report versions. |
| `GET` | `/reports/{version_id}/html` | Local UI | Interactive HTML/SVG attribution dashboard for a specific version. |

## 2. RBAC Permission Matrix

| Role | Read Approved | Read Draft/Reviewed | Run / Revise | Review | Approve | Read Audit |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `VIEWER` | Yes | No | No | No | No | No |
| `ANALYST` | Yes | Yes | Yes | No | No | No |
| `REVIEWER` | Yes | Yes | No | Yes | No | Yes |
| `APPROVER` | Yes | Yes | No | No | Yes | Yes |
| `ADMIN` | Yes | Yes | Yes | Yes* | Yes* | Yes |

\* Dual-control still blocks `ADMIN` from reviewing or approving a report they authored.

## 3. Structured Error Format

Every domain/API failure returns JSON:

```json
{
  "error_code": "AUTHORIZATION_ERROR",
  "message": "Actor 'viewer-1' with role VIEWER lacks permission RUN_ATTRIBUTION.",
  "details": {"actor_id": "viewer-1", "role": "VIEWER", "required": "RUN_ATTRIBUTION"}
}
```

Defined codes: `VALIDATION_ERROR` (422), `CONFIGURATION_ERROR` (400), `ORDER_EXECUTION_PROHIBITED` (403), `ZERO_NAV` (422), `CURRENCY_CONVERSION_ERROR` (422), `BENCHMARK_ALIGNMENT_ERROR` (422), `RECONCILIATION_ERROR` (422), `AUTHORIZATION_ERROR` (403), `WORKFLOW_VIOLATION` (409), `NOT_FOUND` (404), `INTEGRITY_ERROR` (500).

## 4. Relational Schema

- **Local Prototype:** SQLite schema auto-initialized by `SQLiteAttributionStore` (`pae_input_batches`, `pae_attribution_reports`, `pae_audit_log`).
- **Paper/Sandbox PostgreSQL:** Committed DDL in [`config/postgres_schema.sql`](../config/postgres_schema.sql) including `CHECK` constraints enforcing `reviewer_id <> author_id`, `approver_id <> author_id`, `version_number >= 1`, and indexed provenance tables (`pae_contribution_provenance`, `pae_data_quality_exceptions`, `pae_audit_log`).
