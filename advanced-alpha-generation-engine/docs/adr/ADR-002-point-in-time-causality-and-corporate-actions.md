# ADR-002: Strict UTC Point-in-Time Causality and Corporate Action Verification

- **Status**: Accepted
- **Date**: 2026-11-27
- **Milestone**: M1 — Foundation and design

## Context

Quantitative alpha research frequently suffers from subtle data leakage: unlagged fundamental filings, omitted delisted tickers (survivorship bias), unadjusted split price cliffs, and naive local timestamps.

## Decision

1. Every domain contract and Polars frame enforces timezone-aware UTC `event_ts_utc` and `available_at_utc` with `available_at_utc >= event_ts_utc`.
2. Fundamental data is joined strictly via backward `join_asof` on `available_at_utc` (respecting realistic filing lags) rather than `period_end_utc`.
3. `PointInTimeValidator` runs automatically on load and before signal evaluation to fail fast with structured domain errors (`LookAheadViolationError`, `SurvivorshipViolationError`, `CorporateActionIntegrityError`, `MissingSymbolError`).

## Consequences

- **Positive**: Eliminates silent look-ahead and survivorship bugs; guarantees deterministic reproducibility across runs and environments.
- **Trade-off**: Strict validation requires upstream datasets to supply explicit `available_at_utc`, delisting exit timestamps, and corporate action tables.
