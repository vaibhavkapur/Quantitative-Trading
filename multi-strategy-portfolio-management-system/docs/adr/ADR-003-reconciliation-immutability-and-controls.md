# ADR-003: Non-Destructive Broker Reconciliation, Missing-Mark Flagging, and Operational Controls

- **Status**: Accepted
- **Milestone**: M3 & M4 — Integration, Validation & Hardening

## Context
Two common portfolio-system failure modes cause silent NAV corruption: (a) silently valuing unpriced/stale securities at zero, and (b) letting automated broker reconciliation overwrite historical internal trades or balances.

## Decision
1. **Unpriced Asset Handling**: When a position lacks a market mark (`MISSING_MARK`), exceeds the staleness threshold (`STALE_MARK`), or lacks an FX rate (`MISSING_FX`), never silently value it at zero. Carry it at cost basis, mark `is_priced = False` and `valuation_complete = False`, and emit explicit `ValuationException` records.
2. **Read-Only Reconciliation**: `ReconciliationEngine.reconcile()` operates strictly on immutable `ReplaySnapshot` objects and never mutates the event store (`mutated_history = False`).
3. **Approver-Gated Adjustments**: Resolving a break requires `Role.APPROVER` and appends a new immutable `AdjustmentEvent` referencing `reconciliation_break_id`, `approved_by`, and `reason`.
4. **Live Execution Guard**: `live_order_execution_enabled` defaults to `False` and raises `ControlViolationError` (`HTTP 403`) on any live order attempt.

## Consequences
- **Positive**: Complete audit trail, clear separation of duties between operators and approvers, zero silent NAV distortions.
- **Trade-off**: Requires explicit operator/approver workflow steps to clear breaks and certify incomplete valuations.
