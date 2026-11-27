# ADR-001: Event-Sourced Double-Entry Subledger with Hash Chaining

- **Status**: Accepted
- **Milestone**: M1 — Foundation and Design

## Context
A multi-strategy fund requires an authoritative record of positions, cash, and P&L that survives crashes, explains every balance movement, prevents duplicate fill booking, and guarantees auditability across strategies and unallocated fund cash.

## Decision
1. Represent all state mutations as immutable, typed transaction events (`TradeFillEvent`, `CashTransferEvent`, `StrategyAllocationEvent`, `FxConversionEvent`, `FeeEvent`, `CorporateActionEvent`, `AdjustmentEvent`) with mandatory UTC `event_time`, `recorded_time`, and unique `idempotency_key`.
2. Persist events in an append-only `ledger_events` table protected by database-level `BEFORE UPDATE` and `BEFORE DELETE` abort triggers and linked by a SHA-256 hash chain (`prev_hash`, `event_hash`).
3. Derive every balance, lot, and P&L metric via deterministic replay from `seq_no = 1`, generating a balanced double-entry `JournalEntry` per event where `sum(debits) == sum(credits)` independently for every currency.

## Consequences
- **Positive**: 100% reproducible recovery from raw event logs, cryptographic tamper evidence, exact audit lineage from any position or cash balance back to source events.
- **Trade-off**: Full replay scales linearly with event count; periodic verified checkpoints (`materialized_state_cache`) can accelerate reads while keeping the raw event log authoritative.
