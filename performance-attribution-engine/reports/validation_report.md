# Performance Attribution Engine — Validation & Benchmark Report

- **Generated (UTC):** `2026-11-27T10:17:29Z`
- **Methodology Version:** `pae-v1.0.0`
- **Environment:** Python `3.9.6` on `macOS-26.5.1-arm64-arm-64bit`
- **Order Execution Enabled:** `False` (strictly disabled)

## 1. Automated Test Suite Summary

| Metric | Value |
|---|---:|
| Total Tests Executed | **25** |
| Failures | **0** |
| Errors | **0** |
| Skipped | 0 |
| Overall Result | **PASS** |
| Runtime | 0.087s |

Raw runner transcript is retained in [`reports/test_report.txt`](test_report.txt).

## 2. Reproducible End-to-End Demo & Versioning Evidence

| Artifact | Version ID | Reconciliation | Gross Return | Net Return | Linked Net Residual |
|---|---|---|---:|---:|---:|
| Multi-Strategy v1 Baseline | `RPT-PORT_SAAT_MULTI-20260105-20260116-v1-e9884ff5` | `RECONCILED` | +4.4132% | +4.3274% | +0.000000 bps |
| Multi-Strategy v2 Fee Correction | `RPT-PORT_SAAT_MULTI-20260105-20260116-v2-56c9bae2` | `RECONCILED` | +4.4125% | +4.3650% | +0.000000 bps |
| Hand-Worked Portfolio | `RPT-PORT_HAND_WORKED-20260105-20260106-v1-a136798a` | `RECONCILED` | +13.4240% | +13.1645% | -0.000000 bps |

- **v2 Supersedes:** `RPT-PORT_SAAT_MULTI-20260105-20260116-v1-e9884ff5`
- **v2 Delta Net Return:** `+0.037608%` (Delta Fees: `-449.87 USD`)
- **Audit Hash Chain Verified:** `True` (7 entries)

## 3. Baseline Performance Measurement (Local Synthetic Prototype)

- **Workload:** 10 trading days, 4 multi-currency instruments, 3 strategies, 4 trades, Brinson + 4-factor OLS + TCA
- **Sample Size:** N = 25 runs
- **Latency (ms):** median `5.244 ms`, mean `5.263 ms`, stdev `0.088 ms` (min `5.143 ms`, max `5.457 ms`)

_Note: Measurements reflect synthetic local execution only; live/production throughput targets require representative production sizing._
