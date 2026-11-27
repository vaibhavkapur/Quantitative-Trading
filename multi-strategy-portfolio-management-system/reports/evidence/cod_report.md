# Close-of-Day Portfolio Report — COD-SNAP-COD-20260115-SEQ18

- **Fund**: SAAT Multi-Strategy Master Fund (`FUND-SAAT-MAIN`)
- **As-Of Time (UTC)**: `2026-01-15T21:00:00+00:00`
- **Market Snapshot**: `SNAP-COD-20260115`
- **Ledger Events / Last Seq**: `18` / `18`
- **Ledger State Hash (SHA-256)**: `05bd03d1d8cd623b8486477537e8cc02a771b7a902259306c1deac9350c30d7e`
- **Event Hash Chain Valid**: `True`
- **Valuation Complete**: `False` (Unpriced instruments: `['EQ-UNPRICED-US']`)
- **Consolidation Check (Fund == Strategies + Unallocated)**: `PASS`

## Fund Summary

| Metric | Amount |
|---|---:|
| Fund NAV | 1,014,733.96 USD |
| Net Capital Flows | 1,000,000.00 USD |
| Total Cash (Base) | 721,595.96 USD |
| Strategy Cash (Base) | 621,715.96 USD |
| Unallocated Cash (Base) | 99,880.00 USD |
| Priced Market Value | 288,138.00 USD |
| Unpriced Cost Basis (Flagged) | 5,000.00 USD |
| Gross Exposure | 374,138.00 USD |
| Net Exposure | 293,138.00 USD |

## P&L Explainability

| Component | Amount |
|---|---:|
| Realized Trading P&L | 2,000.00 USD |
| Unrealized Price P&L | 9,458.00 USD |
| FX Translation P&L | 3,000.00 USD |
| Dividend Income | 500.00 USD |
| Fees & Commissions (incl. Late Fees) | -224.04 USD |
| Late Fees Subset | -45.50 USD |
| Audited Adjustments | 0.00 USD |
| **Explained Total P&L** | **14,733.96 USD** |
| **NAV - Net Capital** | **14,733.96 USD** |
| **Unexplained Residual** | **0.00 USD** |

## Strategy & Unallocated Breakdown

| Account | Strategy | Cash (Base) | NAV (Base) | Total P&L | Unpriced |
|---|---|---:|---:|---:|---|
| `ACCT-STRAT-MACRO` | STRAT-MACRO | 157,878.76 USD | 202,248.76 USD | 2,248.76 USD | EQ-UNPRICED-US |
| `ACCT-STRAT-MOM` | STRAT-MOMENTUM | 251,464.00 USD | 407,944.00 USD | 7,944.00 USD | None |
| `ACCT-STRAT-STATARB` | STRAT-STATARB | 212,373.20 USD | 304,661.20 USD | 4,661.20 USD | None |
| `ACCT-FUND-UNALLOC` | UNALLOCATED | 99,880.00 USD | 99,880.00 USD | -120.00 USD | None |

## Valuation Exceptions

- **[MISSING_MARK]** `ACCT-STRAT-MACRO` / `EQ-UNPRICED-US`: No market price for ILLIQ (EQ-UNPRICED-US) in snapshot SNAP-COD-20260115; carrying at cost basis and flagging unpriced.

## Broker Reconciliation Summary

- **Run ID**: `RECON-0001-PB-STMT-20260115-EOD` (Statement `PB-STMT-20260115-EOD` from `SYNTHETIC_PRIME_BROKER_ALPHA`)
- **Positions Checked**: `5` | **Cash Balances Checked**: `6`
- **Total Breaks**: `2` | **Open Breaks**: `2` | **Mutated History**: `False`

| Break ID | Type | Account | Instrument/CCY | Internal | Broker | Diff | Status |
|---|---|---|---|---:|---:|---:|---|
| `RECON-0001-PB-STMT-20260115-EOD-BRK-001` | `POSITION_QUANTITY_BREAK` | `ACCT-STRAT-STATARB` | `EQ-SAP-DE` | 400 | 395 | -5 | `OPEN` |
| `RECON-0001-PB-STMT-20260115-EOD-BRK-002` | `CASH_BALANCE_BREAK` | `ACCT-STRAT-STATARB` | `EUR` | 19985.00 | 19960.00 | -25.00 | `OPEN` |
