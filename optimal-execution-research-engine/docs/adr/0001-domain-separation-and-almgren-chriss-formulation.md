# ADR-0001: Domain Separation and Discrete-Time Almgren-Chriss Formulation

- **Status:** Accepted
- **Date:** 2026-11-28
- **Project:** Optimal Execution Research Engine (`optimal-execution-research-engine`)

## Context

The engine needs to compare baseline execution heuristics (TWAP, VWAP, POV/participation) against optimal risk-averse schedules (Almgren-Chriss) under realistic spread, impact, capacity, and timing-risk assumptions, while remaining deterministic, testable, and independently verifiable against both a hand-calculated fixture and a numerical optimizer.

## Decision

1. **Strict Pure-Domain vs. I/O Separation**:
   - Keep `contracts.py`, `schedules.py`, `simulator.py`, `analytics.py`, and `sensitivity.py` completely free of filesystem, network, and CLI dependencies.
   - Isolate JSON parsing, content hashing, structured logging, replay manifests, and report serialization inside `io.py` and `cli.py`.
2. **Exact Discrete-Time Almgren-Chriss Dispersion Relation**:
   - Use the exact discrete-time Almgren-Chriss (2001) objective with $\tilde{\eta} = \eta - \frac{1}{2}\gamma\tau$ and dispersion relation $\kappa\tau = 2\operatorname{arcsinh}\!\left(\frac{1}{2}\tilde{\kappa}\tau\right)$, rather than the continuous-time approximation $\kappa \approx \tilde{\kappa}$.
   - Evaluate $\sinh(\kappa\tau(N-j)) / \sinh(\kappa\tau N)$ via the overflow- and cancellation-safe form $\exp(-\kappa\tau j)\frac{-\operatorname{expm1}(-2\kappa\tau(N-j))}{-\operatorname{expm1}(-2\kappa\tau N)}$.
3. **Dual Analytic & Numeric Solvers**:
   - Implement an independent SciPy SLSQP solver (`almgren_chriss_numeric_schedule`) minimizing the exact discrete objective with analytical gradients.
   - Use analytic-vs-numeric agreement (`relative_l2_diff <= 1e-4`) as an automated regression gate, and leverage the numeric solver for nonlinear temporary impact ($\alpha \ne 1$) and hard capacity bounds.
4. **Perold (1988) Opportunity Cost & Capacity Policies**:
   - Enforce `filled_k <= capacity_k` and `filled + residual == parent` as hard invariants.
   - Explicitly separate execution shortfall on filled shares from opportunity cost on unfilled residual $s \cdot R \cdot (S_N - S_0)$, and expose configurable `CapacityPolicy.NONE` vs `CapacityPolicy.SPILLOVER`.

## Consequences

- **Positive**: Unconstrained analytic and numeric schedules agree to `< 1e-6` relative L2 error across orders of magnitude of $\lambda$; high-$\lambda$ regimes never overflow `float64`; Monte Carlo empirical mean and standard deviation converge directly to closed-form analytic expectations and timing risk.
- **Trade-offs**: Arithmetic Brownian motion and deterministic per-regime volume curves simplify variance expressions at the expense of omitting stochastic intraday volume and L2 order-book queue dynamics (documented explicitly as modeling limitations).
