# Explicitly Unsupported Features (MVP Scope Boundary)

Per `plan.md` §1 and §8, the following features are intentionally out of scope for this local MVP prototype and must not be assumed present:

1. **Independent Fund Administration & Official NAV Certification**: This system is an internal portfolio accounting and reconciliation subledger; it does **not** replace an independent fund administrator, auditor, or official LP NAV calculation.
2. **Live Order Routing & Real-Capital Execution**: Live broker order submission is explicitly disabled (`ControlViolationError`). Only paper OMS execution report ingestion is supported.
3. **Complex Asset Classes & Derivatives Lifecycle**: Options exercise/assignment, Greeks/vol surfaces, bond accrued interest/amortization, repo/financing accruals, swaps, margin-call calculations, and crypto staking are not implemented in MVP.
4. **Complex Corporate Actions**: Mergers, spin-offs, rights issues, tender offers, and fractional-share cash-in-lieu are not yet modeled (MVP covers stock splits/reverse splits and cash dividends with withholding tax).
5. **Cross-Currency Non-Base FX Pairs**: Direct non-base-to-non-base FX conversions (e.g., `EUR -> GBP` without routing through base currency `USD`) do not yet split cross-rate FX basis automatically; route cross-currency FX via base currency in MVP.
6. **Multi-Node Distributed Consensus & Production Auth**: MVP runs as a single-process FastAPI + SQLite prototype with header-based roles (`X-Role`) and PostgreSQL DDL readiness (`docs/schema.sql`). Production deployment requires PostgreSQL, OIDC/SSO + RBAC middleware, TLS, and centralized secret management.
7. **Tax-Lot Wash-Sale Rules & Regulatory Tax Reporting**: FIFO, LIFO, and AVG_COST lot tracking are supported for management accounting; jurisdiction-specific wash-sale disallowances and K-1/1099 generation are out of scope.
