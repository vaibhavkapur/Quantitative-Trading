# ADR-001: Clean-Room Domain Contracts and Deterministic Pure-Python Convex Solver

- **Status:** Accepted
- **Date:** 2026-11-27

## Context

`plan.md` requires a locally reproducible prototype that runs cleanly from a fresh checkout while isolating optional external dependencies (`NumPy`, `SciPy`, `CVXPY`, `FastAPI`, `PostgreSQL`) and respecting unresolved licensing/IP boundaries on referenced external prototype repositories (`Quantum-Trading`).

## Decision

1. Implement canonical domain contracts (`dcae.domain`), snapshot validation (`dcae.snapshots`), linear algebra (`dcae.math.linalg`), exact convex projections (`dcae.math.projections`), point-in-time estimation (`dcae.estimators`), baselines, and constrained optimization (`dcae.allocators`) in pure Python 3.9+ with **zero mandatory third-party dependencies**.
2. Solve the constrained mean-variance + L1 transaction cost problem via **2-block ADMM** (exact coordinate descent with soft-thresholding for the nonsmooth L1 transaction-cost block + **Dykstra's alternating projection algorithm** onto box, budget sum, group exposure, gross leverage, L1 turnover ball, and volatility ellipsoid constraints).
3. Isolate `FastAPI` (`dcae.api.fastapi_app`) and `PostgreSQL` (`dcae.audit.postgres_store`, `config/postgres_schema.sql`) behind thin adapters while providing full stdlib HTTP and SQLite equivalents by default.

## Consequences

- Clean checkouts execute setup, unit/integration/scenario tests, demos, and walk-forward evaluations reproducibly without network access or compiled C extensions.
- Exact soft-thresholding naturally produces realistic no-trade bands under high transaction costs.
- Future scaling to >500 strategies can plug a `scipy`/`cvxpy` backend behind `ConstrainedAllocator` without altering contracts, verification, or audit trails.
