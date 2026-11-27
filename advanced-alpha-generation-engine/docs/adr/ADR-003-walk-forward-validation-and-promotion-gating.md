# ADR-003: Walk-Forward Validation, Multiple-Testing Correction, and Non-Bypassable Promotion Gates

- **Status**: Accepted
- **Date**: 2026-11-27
- **Milestone**: M3 / M4 — Validation, Hardening, and Delivery

## Context

Evaluating many alpha hypotheses without recording failed trials or enforcing out-of-sample cost, turnover, stability, and duplication thresholds causes severe selection bias and overfitting.

## Decision

1. All signals and predictive models are evaluated using strictly chronological walk-forward splits with a purge/embargo gap `embargo_days >= forward_horizon_days`.
2. Every trial — passing or failing — is appended to an immutable `trials.jsonl` ledger, and Benjamini-Hochberg FDR adjusted p-values are computed across the cumulative trial history.
3. `AlphaCatalog.promote()` enforces policy gates in code (OOS IC, ICIR, BH p-value, turnover, sector exposure, peer correlation duplication, net Sharpe at base cost, coefficient stability, and operator approval metadata) and raises `PromotionGateError` / `FactorDuplicationError` on any violation.
4. `PaperSignalExporter` exports only promoted signals with `execution_enabled=False` and raises `ExecutionDisabledError` on any live-execution request.

## Consequences

- **Positive**: Prevents cherry-picking, blocks near-duplicate or high-turnover factors from reaching paper trading, and preserves complete auditability of negative results.
- **Trade-off**: Stricter multiple-testing and cost gates reject marginal signals that might appear attractive on gross in-sample backtests.
