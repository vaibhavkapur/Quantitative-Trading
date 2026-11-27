# Milestone 2 Report — Core Implementation

**Milestone Scope (`plan.md` §4 M2)**:
- Implement event-sourced balances, realized/unrealized P&L, and deterministic replay.
- Document interfaces, assumptions, and design decisions introduced in this milestone.
- Add automated tests and store a reproducible demonstration or validation report.

## Deliverables Completed

1. **Immutable Append-Only Store** (`src/pms/ledger/store.py`):
   - Both `InMemoryEventStore` and `SQLiteEventStore` (WAL mode) with unique `idempotency_key` and `event_id` constraints, SHA-256 hash chaining (`prev_hash` → `event_hash`), and database-level `BEFORE UPDATE` / `BEFORE DELETE` abort triggers preventing silent history modification.
2. **Double-Entry Balanced Journal** (`src/pms/ledger/journal.py`):
   - Every event produces balanced `JournalLine` entries verified per currency (`sum(debits) == sum(credits)`).
   - Baseline 17-event fixture produces balanced trial balances independently in `USD`, `EUR`, and `GBP`.
3. **Exact Decimal Lot Accounting** (`src/pms/ledger/lots.py`):
   - Supports `FIFO`, `LIFO`, and `AVG_COST` lot methods, futures contract multipliers, partial fills, Long↔Short position flips, and exact cost-basis-preserving stock splits (including 3-for-1 splits with zero rounding drift).
4. **Deterministic Replay Engine** (`src/pms/ledger/engine.py`):
   - Produces canonical `ReplaySnapshot` with SHA-256 `state_hash` and optional point-in-time `as_of` replay.
   - Design documented in `docs/adr/ADR-002-fixed-point-decimal-and-lot-accounting.md`.

## Automated Verification & Reproducibility Evidence

- Unit tests: `tests/unit/test_lot_accounting.py` (3 passed), `tests/unit/test_double_entry_journal.py` (2 passed).
- Integration tests: `tests/integration/test_sqlite_store_and_recovery.py` (2 passed — verifies SQLite immutability triggers and identical cold-start replay).
- Cross-run reproducibility (`reports/evidence/reproducibility_check.json`):
  - Clean run 1 state hash: `4a8622ef3ffa2b31cbbfef3fa9ea1e08e28edf95f8c4c230270057989228829e`
  - Clean run 2 state hash: `4a8622ef3ffa2b31cbbfef3fa9ea1e08e28edf95f8c4c230270057989228829e`
  - `reproducible_across_clean_builds: true`
