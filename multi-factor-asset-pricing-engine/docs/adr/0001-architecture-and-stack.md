# ADR-0001: Layered Python Architecture and Econometric Stack

- **Status:** Accepted
- **Date:** 2026-11-28
- **Project:** Multi-Factor Asset Pricing Engine (`multi-factor-asset-pricing-engine`)

## Context

The engine must align point-in-time asset returns, factor returns, portfolio memberships, and risk-free series; fit time-series CAPM and multi-factor regressions; estimate Fama-MacBeth cross-sectional risk premia with robust standard errors; and compare pricing errors and out-of-sample (OOS) explanatory power cleanly and reproducibly.

Key architectural forces:
1. Strict separation between I/O/table loading and econometric domain logic so core estimators can be tested against exact analytical oracles without disk dependencies.
2. Fast, deterministic tabular validation and point-in-time joins.
3. Econometrically standard OLS, HC3, and Newey-West HAC covariance estimators verifiable against an established reference library.
4. Zero hidden configuration parsers or non-deterministic runtime dependencies.

## Decision

1. **Stack Selection:** Use Python ($\ge 3.9$) with **Polars** for tabular loading, schema validation, and point-in-time excess-return alignment; **NumPy** and **SciPy** for linear algebra (SVD rank/conditioning checks, cross-sectional OLS, GRS F-distribution, Shanken EIV covariance); and **statsmodels** as the reference OLS / HC3 / Newey-West HAC estimator.
2. **Configuration & Manifest Format:** Use canonical **JSON** (`config/*.json`, `replay_manifest.json`) validated into frozen dataclasses (`EngineConfig`, `FactorModelSpec`, `SplitConfig`). JSON requires zero third-party parsers and serializes canonically (`sort_keys=True`) for deterministic SHA-256 `run_id` hashing.
3. **Pure Domain Boundary:** `time_series.py`, `fama_macbeth.py`, and `evaluation.py` accept only in-memory `AlignedPanel` / dataclass objects and perform zero filesystem I/O. All disk access and artifact serialization live exclusively in `io.py` and `engine.py`.
4. **Explicit Failure Policies:** All data-quality anomalies (missing risk-free rates, misaligned dates, mid-sample universe changes, rank deficiency, date leakage) default to raising typed `MFAPEError` subclasses unless the user explicitly opts into a documented recovery policy (`rf_missing_policy="drop_period"`, `period_alignment_policy="inner"`, `universe_policy="pit_active"`).

## Consequences

- **Positive:** Domain algorithms are fast, deterministic, easy to unit-test against hand-calculated and `statsmodels` oracles, and straightforward to wrap as adapters by sibling projects in `Trading-Quantitative`.
- **Positive:** Replay integrity is verified cryptographically via SHA-256 input fingerprints.
- **Negative / Trade-off:** `statsmodels` pulls `pandas` and `patsy` as transitive dependencies even though the alignment pipeline itself operates natively on Polars and NumPy. This is accepted to keep HAC/HC3 covariance estimation aligned with the widely audited `statsmodels` implementation.
