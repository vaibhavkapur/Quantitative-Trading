# ADR-0001: Cross-Asset Return Normalization, Explicit Carry Conventions, and As-Of Timestamp Gating

**Status:** Accepted  
**Date:** 2026-11-28  
**Milestone:** M1 — Foundation and reference fixtures

---

## Context

Combining factor signals across equities, FX, and commodities introduces four well-known research failure modes:
1. **Contract-splice artifacts:** Naive price ratios across futures contract rolls inject artificial jumps from backwardation/contango spreads into returns, momentum, and volatility estimates.
2. **Currency-conversion mismatch:** Unfunded futures contracts earn P&L in their quote currency (only variation margin is exposed to FX), whereas funded spot positions expose full notional principal to FX. Reverse-quoted FX pairs (`USDJPY`) invert the quote/base direction.
3. **Heterogeneous carry definitions:** Carry has different economic definitions across equities, FX, and commodities; conflating them or omitting financing/roll costs distorts cross-asset comparisons.
4. **Look-ahead bias from publication lag and full-sample scaling:** Joining corporate fundamentals by reference period `date` instead of `publication_ts`, or z-scoring factors across the entire dataset, silently leaks future information into historical weights.

## Decision

1. **Base-currency excess return normalization (`normalization.py`):**
   - Standardize all instruments into base-currency (`USD`) excess returns before factor computation and portfolio optimization.
   - Require explicit `prev_front_price_new_contract` on every futures contract code transition (`RollNormalizationError` if missing).
   - Separate `gross_excess_return_base`, `roll_cost_drag`, and `financing_cost_drag` into distinct columns so cost attribution is auditable per asset and asset class.
2. **Documented per-class carry conventions (`contracts.py`, `factors.py`):**
   - Equities: `dividend_minus_financing`
   - FX: `interest_rate_differential`
   - Commodities: `futures_term_structure_roll`
   - Enforce class-to-convention consistency during schema validation.
3. **Mandatory point-in-time timestamp gate (`factors.py`):**
   - Every signal row carries `as_of_ts`, `source_max_publication_ts`, and `provenance_hash`.
   - Enforce `source_max_publication_ts <= as_of_ts` as a hard invariant (`TimestampGateError` on violation).
4. **Training-window-only `FactorScaler` (`factors.py`, `evaluation.py`):**
   - Fit within-asset-class z-score mean/std strictly on each walk-forward fold's training window and apply frozen parameters to the test window.
5. **Independent analytical oracle (`data/fixtures/valid_small/expected_reference.json`):**
   - Maintain a pure-Python hand-calculated fixture verified to `1e-10` absolute tolerance before scaling to multi-year synthetic evaluations.

## Consequences

- **Positive:** Complete elimination of contract-roll jumps, currency-basis errors, and look-ahead leakage; every signal and rebalance is independently auditable.
- **Trade-off:** Inputs require explicit `publication_ts` and roll-transition pricing (`prev_front_price_new_contract`); raw vendor feeds must pass through an adapter that populates these fields before entering the engine.
- **Rejected alternatives:**
  - *Backward-adjusted Panama canal price series:* Rejected because additive back-adjustment distorts percentage returns and can produce negative prices for high-roll commodities.
  - *Expanding full-history cross-sectional rank without fold separation:* Rejected because walk-forward evaluation requires explicitly stored fold boundaries and training-only fitted scalers.
