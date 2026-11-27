# ADR-002: Exact Decimal Fixed-Point Arithmetic and Cost-Basis-Preserving Lot Accounting

- **Status**: Accepted
- **Milestone**: M2 — Core Implementation

## Context
Binary floating-point (`float`) arithmetic introduces silent rounding drift in cash balances, cost basis, splits (such as 3-for-1), FX translation, and P&L attribution, breaking double-entry equality and P&L explainability checks.

## Decision
1. Use Python `decimal.Decimal` exclusively across all monetary amounts, share quantities, multipliers, prices, fees, and FX rates, serialized as strings in JSON APIs and storage.
2. Track `cost_basis_abs` explicitly on each `OpenLot` so closing an entire lot releases exact remaining cost basis without rounding drift, and stock splits adjust share quantity and unit cost while preserving exact total cost basis (`new_basis == old_basis`, zero phantom P&L).
3. Decompose position flips (Long→Short and Short→Long) into a closing leg (realizing exact P&L on existing lots) plus an opening leg (establishing a new opposite-direction lot at the fill price), ensuring both legs produce balanced journal entries.

## Consequences
- **Positive**: Exact trial balance equality, zero unexplained P&L residual (`Decimal("0")`), clean handling of 3-for-1 splits and futures contract multipliers.
- **Trade-off**: Slightly higher CPU cost than native floats, well within portfolio accounting latency budgets.
