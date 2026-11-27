# Milestone 4 Report — Hardening and Delivery

**Milestone Scope (`plan.md` §4 M4)**:
- Integrate paper OMS and broker reconciliation; demonstrate close-of-day reports and recovery.
- Document interfaces, assumptions, and design decisions introduced in this milestone.
- Add automated tests and store a reproducible demonstration or validation report.

## Deliverables Completed

1. **Paper OMS Adapter & Live-Execution Safety Guard** (`src/pms/adapters/paper_oms.py`):
   - Translates `PaperFillReport` into validated `TradeFillEvent` records with automatic idempotent deduplication (`BOOKED` vs `DUPLICATE_IGNORED`).
   - Enforces `live_order_execution_enabled = false` (`ControlViolationError` / `HTTP 403`).
2. **Read-Only Broker Reconciliation & Approver Workflow** (`src/pms/reconciliation/`):
   - Compares internal ledger positions, cash balances, and mark prices against external `BrokerStatement` feeds without ever mutating historical events (`mutated_history = false`).
   - Enforces `Role.APPROVER` authorization to append new audited `AdjustmentEvent` records resolving breaks; `VIEWER` and `OPERATOR` roles are blocked (`HTTP 403`).
3. **Close-of-Day (COD) Reports & Portfolio/Reconciliation Web UI** (`src/pms/reporting/cod.py`, `src/pms/api.py`, `src/pms/web/index.html`, `src/pms/cli.py`):
   - Deterministic COD reports in JSON and Markdown (`reports/evidence/cod_report.json`, `reports/evidence/cod_report.md`).
   - Offline-ready FastAPI Web UI displaying KPI summary, consolidation table, P&L explainability bridge, open FIFO lots, double-entry trial balance, immutable event log, reconciliation workflow, and live recovery verification.
4. **Recovery Verification & Baseline Latency Measurement** (`scripts/demo.py`, `reports/evidence/demo_summary.json`):
   - Materialized SQLite cache wiped and replayed twice from raw `ledger_events`: `recovered_identically: true`, `hash_chain_valid: true`, identical state hash `05bd03d1d8cd623b8486477537e8cc02a771b7a902259306c1deac9350c30d7e`.
   - Local single-process SQLite baseline replay latency (`n = 50`, `18` events/replay, macOS Apple Silicon, Python 3.12.15): median `0.583 ms`, p95 `1.069 ms`, max `1.196 ms` (baseline measurement only; not a production throughput claim).

## Operational Docs & Hardening Artifacts

- `docs/runbook.md`
- `docs/adr/ADR-003-reconciliation-immutability-and-controls.md`
- `docs/unsupported_features.md`
- `tests/integration/test_paper_oms_and_reconciliation.py` (3 passed)
- `tests/integration/test_api_and_ui.py` (2 passed)
