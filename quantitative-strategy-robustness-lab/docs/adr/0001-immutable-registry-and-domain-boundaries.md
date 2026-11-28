# ADR 0001: Immutable Hash-Chained Experiment Registry and Pure Domain Boundaries

- **Status:** Accepted
- **Date:** 2026-11-28
- **Context:** Milestone M1 — Foundation and reference fixtures

## Context

Backtest overfitting frequently occurs not only from parameter tuning inside a single run, but from *unreported trials*—failed runs, discarded parameter sweeps, or selectively deleted experiments that understate the effective trial count $N$ in Deflated Sharpe Ratio (DSR) and hide selection bias in CSCV PBO. In addition, mixing CSV parsing and filesystem state into statistical estimators makes unit testing brittle and complicates independent verification against hand-calculated reference oracles.

## Decision

1. **Append-Only SHA-256 Hash-Chained JSONL Registry (`registry.py`)**:
   - Every trial attempt—whether `COMPLETED`, `FAILED`, `REJECTED`, or `ABORTED`—must be appended to `registry.jsonl` with a strictly increasing `sequence_number` and a SHA-256 `entry_hash` committing to `(prev_hash, canonical_json(record))`.
   - Non-completed trials must record a non-empty `failure_reason` and are counted conservatively in multiple-testing diagnostics (`extra_unobserved_failed_trials`) and parameter grid coverage audits.
   - Appending to a ledger that fails hash verification raises `RegistryIntegrityError` rather than silently continuing.
2. **Strict Domain / I/O Separation**:
   - All statistical computations (`costs.py`, `statistics.py`, `pbo.py`, `bootstrap.py`, `walk_forward.py`, `perturbation.py`, `integrity.py`, `promotion.py`) are pure functions over typed dataclasses and NumPy arrays with zero file I/O.
   - All CSV/JSON reading and writing lives exclusively in `io_adapters.py` and `reporting.py`.
3. **Fail-Closed Promotion Gating**:
   - Missing or skipped diagnostics mark `evidence_complete = False` and strictly produce `PromotionDecision.REJECT`.

## Consequences

- **Positive:** Tampering, silent trial deletion, and cherry-picking become cryptographically or statistically detectable; pure statistical modules can be tested deterministically against analytical oracles in milliseconds; upstream strategy engines can integrate via either JSONL/CSV bundles or direct Python calls.
- **Negative / Trade-offs:** Single-file JSONL is designed for offline research workflows, not concurrent multi-process writers; high-frequency distributed sweeps would require a database-backed implementation of the same `ImmutableExperimentRegistry` interface in a future phase.
