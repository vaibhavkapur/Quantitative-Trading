# Architecture Decision Records (ADRs)

## ADR-001: Daily Sub-Period TWR with Carino Logarithmic Multi-Period Linking

- **Status:** Accepted (`pae-v1.0.0`)
- **Context:** Portfolios experience external investor subscriptions and redemptions across periods, while attribution requires multi-day contributions to sum exactly to the cumulative portfolio return.
- **Decision:** Compute daily Time-Weighted Return (TWR) using exact cash-flow timing (`START_OF_DAY` adjusts denominator `B_t = start_nav + CF_sod`; `END_OF_DAY` adjusts numerator `E_t = end_nav - CF_eod`), compute Modified Dietz as a period cross-check, and link daily contributions across periods using **Carino (1999) logarithmic smoothing** (and exact active Carino weights for Brinson active returns).
- **Consequences:** Contributions + explicit residual sum to cumulative return to floating-point precision (`< 1e-12`). Any zero-NAV denominator (`B_t <= 0`) is explicitly caught as `ZERO_NAV` rather than causing division errors.

## ADR-002: Strict Gross vs. Net Separation & Exact Additive FX Decomposition

- **Status:** Accepted
- **Context:** Plan non-goal prohibits misrepresenting gross returns as net returns. Multi-currency holdings also introduce non-linear FX interaction terms that often become hidden residuals.
- **Decision:**
  1. Enforce `gross_vs_net_strict_separation=True` as a non-overridable configuration invariant (`ConfigurationError` if False). Every daily and cumulative view reports `gross_return`, `fee_drag_return`, and `net_return` separately.
  2. Decompose start-of-day multi-currency position P&L via the exact Karnosky-Singer identity: `local_price_pnl + fx_translation_pnl + fx_interaction_pnl == q_0 * (P_1*fx_1 - P_0*fx_0)`.
  3. Capture intraday rebalances explicitly via mark-to-close trading P&L `dq * (P_1 - P_exec) * fx_1`.
- **Consequences:** Zero artificial FX or rebalance residuals on clean ledgers; any non-zero residual reflects genuine custodian rounding or missing ledger entries and is surfaced transparently.

## ADR-003: Non-Causal Factor Attribution & Graceful Optional Module Degradation

- **Status:** Accepted
- **Context:** Factor data or sector-level benchmark weights may be unavailable or incomplete on certain dates, and factor correlations must never be overstated as causal proof.
- **Decision:**
  1. Embed an immutable `non_causation_disclaimer` in every `FactorAttributionReport` and report export.
  2. Never silently forward-fill missing benchmark dates with `0.0` or crash portfolio attribution when benchmark/factor series are absent. Instead, withhold cumulative active return on incomplete benchmark series, mark Brinson/Factor status as `PARTIAL` / `UNAVAILABLE` / `INSUFFICIENT_DATA`, and emit structured `DataQualityException` entries.
- **Consequences:** Operators always receive auditable core return/contribution numbers alongside explicit data-quality visibility.

## ADR-004: Clean Upstream Adapter Contracts (IP & Licensing Isolation)

- **Status:** Accepted
- **Context:** External repositories referenced in `plan.md` have unconfirmed corporate IP/licensing status.
- **Decision:** Implement zero-copy, clean-room domain models and versioned adapter contracts (`UpstreamTCAAdapter`, `HFTFillAdapter`, `PortfolioOptimizerTargetAdapter`) in `src/pae/adapters/contracts.py`.
- **Consequences:** Clean IP provenance for SAAT, zero tight coupling to prototype internals, and straightforward integration once formal data/code entitlements are approved.

## ADR-005: Zero-Dependency Core Runtime with Optional Stack Adapters & Immutable Report Versioning

- **Status:** Accepted
- **Context:** A clean checkout must run setup, tests, and demo deterministically offline while still honoring the suggested production stack (`Python, Polars, NumPy, PostgreSQL, FastAPI, dashboard`).
- **Decision:** Build the entire mathematical core, SQLite versioned store, SHA-256 hash-chained audit log, HTTP API, and SVG dashboard using Python 3.9+ standard library only, and provide clean optional adapters for Polars (`OptionalPolarsAdapter`), FastAPI (`pae.api.fastapi_app`), and PostgreSQL (`config/postgres_schema.sql` + `PostgresAttributionStore`). Corrections (e.g., fee restatements) never overwrite prior versions; they create immutable `v(N+1)` reports linked via `supersedes_version_id` with a computed `revision_diff`.
- **Consequences:** Immediate clean-checkout reproducibility in any environment plus a clear migration path to approved paper/sandbox PostgreSQL/FastAPI infrastructure.
