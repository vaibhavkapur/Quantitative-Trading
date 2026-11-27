# Performance Attribution Engine (PAE)

> **SIMULATED / SYNTHETIC PROTOTYPE — NOT LIVE OR AUDITED PERFORMANCE.**  
> Implements the complete MVP scope of [plan.md](plan.md) with deterministic synthetic fixtures, strict **Gross vs. Net** separation, non-causal factor disclosures, versioned methodology (`pae-v1.0.0`), and zero required third-party runtime dependencies. **Order execution is permanently disabled (`order_execution_enabled: false`).**

## Overview

The Performance Attribution Engine explains portfolio returns and execution costs by **strategy**, **instrument**, **sector**, **currency**, and **risk factors** using reconciled position, cash-flow, FX, benchmark, and trade history.

### Implemented MVP Deliverables

1. **Return & Benchmark Calculation** (`src/pae/domain/`, `src/pae/engine/returns.py`):
   - Canonical versioned contracts (`AttributionInputBatch`, `DailyNavSnapshot`, `PositionSnapshot`, `TradeExecution`, `CashFlow`, `FeeAdjustment`, `FxRateObservation`, `BenchmarkObservation`, `FactorObservation`) with UTC `as_of_utc` / `availability_utc` timestamps, ISO-4217 currencies, and SHA-256 content hashes.
   - Daily Time-Weighted Return (TWR) invariant to `START_OF_DAY` / `END_OF_DAY` external cash flows, Modified Dietz cross-check, strict Gross-vs-Net separation (`r_gross - fee_drag == r_net`), and date/currency-aligned benchmark conversion.
2. **Strategy / Instrument / Sector / Currency Contribution** (`src/pae/engine/contribution.py`):
   - Exact daily additive P&L decomposition: local price hold P&L + FX translation + FX interaction (`Karnosky-Singer` identity) + intraday rebalance trading P&L + income - explicit fees.
   - Exact multi-period **Carino logarithmic linking** preserving sum-to-total additivity.
   - Full provenance: every contribution record carries its underlying `position_ids` and `trade_ids`.
3. **Brinson & Risk-Factor Attribution** (`src/pae/engine/brinson.py`, `src/pae/engine/factors.py`):
   - **Brinson-Fachler** and **Brinson-Hood-Beebower** sector attribution with exact Carino active-return linking and intraday rebalance isolation.
   - **Multi-factor OLS regression attribution** with standard errors, t-statistics, R², Carino-linked factor contributions, idiosyncratic residual, and mandatory non-causation disclosure ("factor correlation does not prove causation").
4. **Execution-Cost Attribution (TCA)** (`src/pae/engine/tca.py`, `src/pae/adapters/contracts.py`):
   - Perold Implementation Shortfall decomposition (delay cost, market impact, opportunity cost on unfilled quantities, explicit commissions/exchange fees/taxes) plus VWAP slippage, aggregated by strategy, instrument, and venue.
   - Clean upstream adapter contracts (`UpstreamTCAAdapter`, `HFTFillAdapter`, `PortfolioOptimizerTargetAdapter`, `OptionalPolarsAdapter`) isolating external dependencies and IP boundaries.
5. **Reconciliation, Versioned Report API & Dashboard** (`src/pae/engine/reconciliation.py`, `src/pae/storage/`, `src/pae/api/`, `src/pae/web/`):
   - Automated reconciliation checking daily/multi-period identities, valuation rounding residuals, and emitting structured `DataQualityException` records instead of silently hiding gaps.
   - Content-addressed immutable report versions (`SQLiteAttributionStore` + optional `PostgresAttributionStore` & DDL in `config/postgres_schema.sql`), revision chaining (`supersedes_version_id`, `revision_diff`) when fee corrections occur, RBAC & dual-control approval workflow, SHA-256 hash-chained audit log, multi-format exports (`JSON`, `Markdown`, `CSV`, `HTML`), zero-CDN SVG dashboard, stdlib HTTP server, and optional FastAPI ASGI wrapper.

## Run locally

Requires only **Python 3.9+** (standard library only; no network or third-party packages required).

```bash
# 1. Verify Python runtime and regenerate deterministic synthetic fixtures + manifest
bash scripts/setup.sh

# 2. Run all unit, hand-worked, integration, and 7 mandatory scenario tests + produce validation evidence
bash scripts/validate.sh

# 3. Run only the reproducible end-to-end demo (v1 -> review -> approve -> v2 fee correction -> exports)
bash scripts/demo.sh

# 4. Full clean one-command reproduction
bash scripts/reproduce.sh

# 5. (Optional) Start the local REST API & interactive attribution dashboard
PYTHONPATH=src python3 -m pae.api.http_server --host 127.0.0.1 --port 8090
```

Optional extras (`pyproject.toml`): `pip install -e .[analytics,api,postgres,dev]` enables optional Polars/NumPy, FastAPI/Uvicorn, PostgreSQL (`psycopg`), and `pytest`.

## Documentation & Retained Evidence

- [docs/architecture.md](docs/architecture.md) — Architecture diagram, component boundaries, upstream foundation audit, and M1–M4 milestone summary
- [docs/ADRs.md](docs/ADRs.md) — Architecture Decision Records (ADR-001 through ADR-005)
- [docs/assumptions_and_methodology.md](docs/assumptions_and_methodology.md) — Mathematical formulas, conventions, tolerances, and data-quality exception catalog
- [docs/api_and_schema.md](docs/api_and_schema.md) — REST API specification, RBAC matrix, structured errors, and database schemas
- [docs/runbook.md](docs/runbook.md) — Operator workflow, failure handling, operational controls, and security/IP checklist
- [docs/unsupported_features.md](docs/unsupported_features.md) — Explicit scope boundaries and unsupported features
- [reports/acceptance_evidence.md](reports/acceptance_evidence.md) — Acceptance criteria traceability matrix
- [reports/validation_report.md](reports/validation_report.md) — Automated test summary, demo evidence, and baseline latency measurement
- [reports/test_report.txt](reports/test_report.txt) — Raw test execution transcript
