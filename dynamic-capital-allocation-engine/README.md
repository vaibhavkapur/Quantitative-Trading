# Dynamic Capital Allocation Engine (DCAE)

> **Status:** Local reproducible MVP prototype (synthetic fixtures; paper-only execution gated on dual-control approval; automated live reallocation is strictly disabled).

The Dynamic Capital Allocation Engine allocates a fund-level risk and capital budget across independently tracked strategies subject to cash reserve, exposure, liquidity/ADV, turnover, and portfolio/stressed volatility constraints.

## Key Capabilities

- **Versioned Point-in-Time Snapshots (`dcae.domain`, `dcae.snapshots`):** Canonical UTC schemas for strategy NAV/returns, eligibility, exposure/liquidity, and current paper positions with SHA-256 content hashes and strict look-ahead filtering (`available_at_utc <= as_of_utc`).
- **Point-in-Time Estimator (`dcae.estimators`):** Return & covariance shrinkage, spectral non-PSD detection with configurable `reject` or `repair` policies, zero-volatility screening, insufficient-history guards, and automatic correlated-drawdown correlation stress overlays.
- **Baselines & Constrained Allocator (`dcae.allocators`):**
  - `equal_weight`
  - `inverse_volatility`
  - `risk_budget` (Spinu/cyclical coordinate descent ERC / general risk budgets)
  - `constrained_mv` (ADMM + Dykstra alternating projections with exact nonsmooth L1 transaction costs and safe `no_change` / `cash_fallback` on infeasibility)
- **Independent Constraint Verification (`dcae.verification`):** Decoupled post-solve `ConstraintVerifier` checking every hard constraint independently and producing a structured `FeasibilityReport`.
- **Approval-Gated Proposals & Audit Trail (`dcae.proposals`, `dcae.audit`):** Dual-control approval workflow (`proposer != reviewer`), TTL expiry, tamper-evident SHA-256 hash-chained SQLite audit store (+ JSONL export and PostgreSQL production DDL/adapter).
- **Paper Portfolio Ledger & Walk-Forward Evaluation (`dcae.paper`, `dcae.evaluation`):** Idempotent approval-gated paper execution and strictly chronological walk-forward comparison of all baselines and constrained allocators on identical periods and transaction costs.
- **Zero-Dependency Core + Optional Stack Adapters (`dcae.api`):** Runs out-of-the-box on Python 3.9+ with zero third-party packages; includes optional FastAPI (`dcae.api.fastapi_app`) and PostgreSQL (`dcae.audit.postgres_store`) adapters.

## Run locally

Run these commands from the project directory with Python 3.9 or later.

```bash
bash scripts/setup.sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/run_demo.py
PYTHONPATH=src python3 scripts/run_walkforward.py
PYTHONPATH=src python3 scripts/validate.py
```

## Repository Layout

```text
dynamic-capital-allocation-engine/
  plan.md                   # Original implementation plan
  README.md                 # Project overview and quickstart
  pyproject.toml            # Packaging and optional dependency groups
  config/
    default_config.json     # Versioned settings (no secrets)
    postgres_schema.sql     # PostgreSQL DDL for approved paper/sandbox deployment
  data/fixtures/            # Deterministic synthetic snapshots, scenarios, and manifest.json
  docs/
    architecture.md         # Component boundaries, Mermaid diagram, foundation audit
    api_and_schemas.md      # HTTP endpoints, domain contracts, and structured errors
    runbook.md              # Operator workflow, failure handling, security/IP/unsupported list
    adr/                    # Architecture Decision Records (ADR-001 through ADR-003)
  src/dcae/                 # Domain, math, snapshots, estimators, allocators, verifier, audit, proposals, paper, evaluation, API
  tests/
    unit/                   # Domain, math, allocators, verifier unit tests
    integration/            # End-to-end approval, idempotency, tamper detection, HTTP API, walk-forward
    scenarios/              # All 7 required failure and stress scenarios
  scripts/                  # Setup, fixture generation, demo, walk-forward, and full validation runner
  reports/                  # Retained raw test, scenario, walk-forward, and milestone acceptance evidence
```

## Explicit Non-Goals & Unsupported Features

- **No automated live reallocation:** `live_execution_enabled=True` is rejected by configuration validation and `mode != "paper"` raises `LiveExecutionProhibitedError`.
- **No guaranteed target Sharpe ratio or live performance claims:** Walk-forward diagnostics evaluate synthetic fixtures only.
- **Unsupported in MVP:** Multi-currency FX hedging (USD only), short-selling / negative strategy capital weights, intraday real-time tick streaming, live broker OMS connectivity, and distributed cluster orchestration.

## Documentation and evidence

- [Architecture and component boundaries](docs/architecture.md)
- [API and data schemas](docs/api_and_schemas.md)
- [Operator runbook](docs/runbook.md)
- [Walk-forward evaluation](reports/walkforward_evaluation.md) and [scenario validation](reports/scenario_validation_report.md)
