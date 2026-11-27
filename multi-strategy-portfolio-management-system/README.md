# Multi-Strategy Portfolio Management System

An authoritative, event-sourced multi-strategy portfolio management prototype that maintains reconciled fund and strategy positions, multi-currency cash balances, tax/accounting lots, mark-to-market exposures, and fully explainable P&L from an immutable transaction event log.

> **Status & Scope Notice**: Local reproducible prototype built with synthetic permitted fixtures. Order execution is disabled by default (`live_order_execution_enabled = false`). This system does **not** replace independent fund administration or audited fund accounting without formal validation and sign-off.

## Core Capabilities (MVP Deliverables)

1. **Instrument & Account Master (`src/pms/domain/master.py`)**: Versioned canonical registry for funds, strategy accounts, unallocated treasury accounts, and instruments (equities, futures, ETFs, FX) with strict ISO-4217 currencies, UTC timestamps, multipliers, and lot accounting conventions.
2. **Immutable Double-Entry Subledger (`src/pms/ledger/`)**: Append-only event store (SQLite WAL with database-level `UPDATE`/`DELETE` abort triggers + SHA-256 hash chain; PostgreSQL-compatible DDL in `docs/schema.sql`), strict `Decimal` fixed-point arithmetic, idempotency-key deduplication, and per-currency balanced double-entry journal postings (`sum(debits) == sum(credits)`).
3. **Strategy Allocation & Lot Accounting (`src/pms/ledger/lots.py`, `engine.py`)**: Deterministic event-sourced replay supporting FIFO, LIFO, and average-cost lots, partial fills, Long↔Short position flips, cost-basis-preserving stock splits, cash dividends with withholding tax, late fee postings, and audited reconciliation adjustments.
4. **Mark-to-Market Valuation & Consolidation (`src/pms/valuation/`)**: Versioned price and FX rate snapshots with UTC pricing and availability timestamps, explicit `MISSING_MARK` / `STALE_MARK` / `MISSING_FX` flagging (unpriced assets are never silently zeroed), exact base-currency P&L attribution (zero unexplained residual), P&L bridge across snapshots, and verified `Fund == sum(Strategies) + Unallocated Cash` consolidation.
5. **Paper OMS, Broker Reconciliation & Portfolio Query UI (`src/pms/adapters/`, `reconciliation/`, `api.py`, `web/index.html`)**: Clean paper OMS fill ingestion with hard live-execution kill switch, strictly read-only broker reconciliation that never silently mutates history, approver-gated adjustment workflow, Close-of-Day (COD) report generator, cache-wipe replay recovery verification, FastAPI REST API, and an interactive offline-ready Web UI.

## Run locally

```bash
# 1. Set up the project with uv and Python 3.12 (works offline with cached wheels)
bash scripts/setup.sh

# 2. Run lint, unit/integration/8-scenario test suite, end-to-end demo, and reproducibility check
bash scripts/validate.sh

# 3. Launch the interactive Portfolio & Reconciliation Web UI + REST API
.venv/bin/pms serve --host 127.0.0.1 --port 8080
# Open http://127.0.0.1:8080 in your browser

# 4. Run CLI reports & recovery verification
.venv/bin/pms cod --format markdown
.venv/bin/pms verify-recovery
.venv/bin/pms health
```

## Repository Layout

```text
multi-strategy-portfolio-management-system/
  plan.md
  README.md
  pyproject.toml
  uv.lock
  config/                    # Versioned settings & master registry (no secrets)
  data/fixtures/             # Permitted synthetic fixtures & provenance manifest
  docs/                      # Architecture, API/schema, Postgres DDL, ADRs, runbook, audits
  src/pms/                   # Domain, ledger, valuation, reconciliation, adapters, API, UI
  tests/
    unit/                    # Domain contracts, double-entry, lots, valuation/consolidation
    integration/             # SQLite immutability/recovery, paper OMS, recon, FastAPI/UI
    scenarios/               # All 8 required verification scenarios from plan.md §5
  scripts/                   # setup.sh, demo.py, reproduce.py, validate.sh
  reports/                   # M1–M4 milestone reports, acceptance checklist, raw evidence/
```

## Documentation & Evidence

- [Architecture & Component Boundaries](docs/architecture.md)
- [API Reference & Event Schemas](docs/api_and_schema.md)
- [PostgreSQL / SQLite DDL Schema](docs/schema.sql)
- [Operations Runbook & Recovery](docs/runbook.md)
- [Dependency, Licensing & IP Audit](docs/dependency_and_ip_audit.md)
- [Explicitly Unsupported Features](docs/unsupported_features.md)
- [Architecture Decision Records (ADRs)](docs/adr/ADR-001-event-sourced-double-entry-ledger.md)
- [Acceptance Checklist & Evidence](reports/acceptance_checklist.md)
