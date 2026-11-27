# ADR-003: Non-PSD Covariance Handling, Correlated Drawdown Stress, and Safe Infeasibility Fallback

- **Status:** Accepted
- **Date:** 2026-11-27

## Context

Real-world multi-strategy risk inputs frequently exhibit non-PSD covariance matrices (asynchronous returns or external risk model patches), degenerate zero-volatility series, short histories, correlated drawdown regimes, and temporarily infeasible constraint intersections.

## Decision

1. **Non-PSD covariance:** Detected via exact Jacobi symmetric eigendecomposition. Configurable `non_psd_policy`:
   - `reject`: raises `NonPSDCovarianceError` with exact minimum eigenvalue.
   - `repair` (default): applies spectral eigenvalue clipping to `psd_min_eigenvalue`, recording both the original min eigenvalue and Frobenius repair delta in diagnostics and warnings.
2. **Correlated drawdowns:** Automatically activates correlation stress (blending off-diagonal correlations upward toward `correlation_stress_floor` and re-projecting onto the PSD cone) whenever at least `max(2, ceil(50%))` eligible strategies breach `drawdown_stress_trigger`.
3. **Infeasible constraints:** Detected pre-solve (contradictory bounds/caps), during Dykstra feasibility projection, or by the independent post-solve `ConstraintVerifier`. Deterministically returns `FALLBACK_NO_CHANGE` (holding current reconciled positions) or `FALLBACK_CASH` (100% cash) with explicit `infeasibility_reasons`.

## Consequences

- The engine never emits NaN weights, division-by-zero allocations, or silently infeasible portfolios.
- Operators receive full diagnostic transparency on every fallback or covariance repair.
