# Milestone 3 Report — Integration and Validation

**Milestone Scope (`plan.md` §4 M3)**:
- Add strategy-to-fund consolidation, pricing timestamps, FX, and exception handling.
- Document interfaces, assumptions, and design decisions introduced in this milestone.
- Add automated tests and store a reproducible demonstration or validation report.

## Deliverables Completed

1. **Mark-to-Market Snapshots & Timestamp Validation** (`src/pms/valuation/pricing.py`):
   - Versioned `MarketSnapshot` holding `PriceMark` and `FxRateMark` records with explicit UTC `price_timestamp` / `rate_timestamp`, `availability_timestamp`, and `source`.
2. **Strategy-to-Fund Consolidation & Exception Handling** (`src/pms/valuation/consolidation.py`):
   - Computes multi-currency cash and position valuation in base currency (`USD`) and verifies `Fund totals == sum(Strategy totals) + Unallocated Fund Cash` across NAV, cash, and P&L (`nav_difference_base == 0`, `cash_difference_base == 0`, `pnl_difference_base == 0`).
   - Explicitly flags missing marks (`MISSING_MARK`), stale marks (`STALE_MARK` beyond configurable `mark_staleness_seconds`), and missing FX (`MISSING_FX`) without silently zeroing asset carrying values.
3. **Complete P&L Explainability & Snapshot Bridge** (`src/pms/valuation/consolidation.py`, `pnl.py`):
   - Decomposes total P&L into realized trading P&L, unrealized price P&L, FX translation P&L, dividend income, fees/commissions (with late-fee subset), and audited adjustments, enforcing `unexplained_residual_base == 0`.
   - `compute_pnl_bridge()` proves exact delta NAV attribution across mark moves and late fees.

## Automated Verification Evidence

- Unit tests: `tests/unit/test_valuation_and_consolidation.py` (3 passed).
- All 8 required verification scenarios from `plan.md` §5 (`tests/scenarios/test_eight_required_scenarios.py` — 8/8 passed):
  1. Duplicate fill (`test_scenario_1_duplicate_fill`)
  2. Position flip (`test_scenario_2_position_flip`)
  3. Partial fill (`test_scenario_3_partial_fill`)
  4. Multi-currency cash & FX (`test_scenario_4_multi_currency_cash`)
  5. Late fee (`test_scenario_5_late_fee`)
  6. Split & dividend (`test_scenario_6_split_and_dividend`)
  7. Missing mark (`test_scenario_7_missing_mark`)
  8. Reconciliation break (`test_scenario_8_reconciliation_break`)
- Raw evidence: `reports/evidence/pytest_output.txt`, `reports/evidence/cod_report.json`.
