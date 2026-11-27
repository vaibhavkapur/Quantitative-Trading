# Operations Runbook — Multi-Strategy PMS

## 1. Local Deployment & Verification

```bash
bash scripts/setup.sh
bash scripts/validate.sh
.venv/bin/pms serve --host 127.0.0.1 --port 8080
```

## 2. Standard Close-of-Day (COD) Procedure

1. Confirm `/api/v1/health` reports `"status": "OK"`, `"hash_chain_valid": true`, and `"live_order_execution_enabled": false`.
2. Ingest the official closing `MarketSnapshot` (`POST /api/v1/markets/snapshots`, role `OPERATOR`) with explicit UTC `price_timestamp` and `availability_timestamp`.
3. Inspect `/api/v1/portfolio/valuation`:
   - Confirm `consolidation_check.passed == true` (`Fund == sum(Strategies) + Unallocated`).
   - Confirm `pnl.unexplained_residual_base == "0"`.
   - Review `unpriced_instruments` and `exceptions`; escalate any `MISSING_MARK` or `STALE_MARK` to the valuation committee before certifying NAV.
4. Run broker reconciliation (`POST /api/v1/reconciliation/run`) against the prime-broker EOD statement.
5. Export COD package (`.venv/bin/pms cod --format markdown` and JSON archive) with SHA-256 `ledger_state_hash`.

## 3. Handling Operational Exceptions

### Duplicate Fill Delivery
- **Detection**: `POST /api/v1/events` returns `409 Conflict (DUPLICATE_EVENT)`; Paper OMS returns `status="DUPLICATE_IGNORED"`.
- **Operator Action**: Verify the original `seq_no` and `event_id` in the subledger; no manual intervention required because balances and lots remain unchanged.

### Missing or Stale Market Marks
- **Detection**: `valuation_complete == false`, non-empty `unpriced_instruments`, and structured `MISSING_MARK` / `STALE_MARK` exceptions in the UI banner and COD report.
- **Safe Fallback**: Position is carried at cost basis (never silently valued at `0`) and flagged visibly so uncertified NAV cannot pass as complete.
- **Operator Action**: Obtain an approved valuation mark from the pricing vendor or valuation committee and publish a new versioned `MarketSnapshot`.

### Late Fees Posted After Initial Close
- **Detection**: `FeeEvent` with `is_late_posting=true` and `recorded_time > event_time`.
- **Operator Action**: Append the immutable `FeeEvent`, re-run COD valuation, and retain the P&L Bridge (`compute_pnl_bridge`) proving `delta_nav == -late_fee` with zero unexplained residual.

### Broker Reconciliation Break
- **Detection**: `ReconciliationRun` lists `POSITION_QUANTITY_BREAK`, `CASH_BALANCE_BREAK`, `MISSING_INTERNAL_POSITION`, `MISSING_BROKER_POSITION`, or `PRICE_DISCREPANCY`.
- **Immutability Control**: Never attempt to `UPDATE` or `DELETE` rows in `ledger_events` (blocked by SQLite/PostgreSQL triggers).
- **Operator Action**:
  1. Investigate whether a fill, corporate action, FX conversion, or fee is missing from internal or broker records.
  2. If an authentic ledger adjustment is required, submit `POST /api/v1/reconciliation/breaks/{break_id}/resolve` with `X-Role: APPROVER`, `X-Actor: <approver_id>`, and a documented `reason`.
  3. Re-run reconciliation and confirm `is_clean == true` and `unexplained_residual_base == "0"`.

## 4. Disaster Recovery (Corrupted Cache / Read-Model Loss)

1. Run `.venv/bin/pms verify-recovery` (or `POST /api/v1/recovery/verify`).
2. The procedure deletes `materialized_state_cache`, verifies the SHA-256 hash chain across `ledger_events`, replays all events from `seq_no = 1` twice, and confirms `recovered_identically == true` with matching `state_hash`.
3. If `hash_chain_valid == false`, halt COD sign-off immediately, isolate the database file for forensic inspection, and restore from the latest verified immutable backup.
