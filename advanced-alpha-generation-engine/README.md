# Advanced Alpha Generation Engine

Reusable library and evaluation framework for causal cross-sectional and time-series alpha signals, with point-in-time (PIT) validation, risk neutralization, cost-aware position research, multiple-testing tracking, and a versioned paper-only alpha catalog.

> **Disclaimer:** This is a local reproducible research prototype (`execution_mode: paper_only`, `execution_enabled: false`). It makes **no promise of predictive edge or profitable live execution**. Never infer live performance from simulated tests.

## Run locally

```bash
# 1. Use uv to install the locked Python 3.12 environment and generate synthetic fixtures
./scripts/setup.sh

# 2. Run reproducible M1–M4 demonstration (generates milestone reports & paper exports)
.venv/bin/python scripts/demo.py

# 3. Run full validation (all 7 failure scenarios, pytest, ruff, acceptance evidence)
.venv/bin/python scripts/validate.py
```

## Key Capabilities Delivered (M1–M4)

- **M1 — Foundation & PIT Validation**: Canonical UTC Pydantic schemas (`MarketBar`, `FundamentalRecord`, `CorporateAction`, `UniverseMembership`, `SignalDefinition`, `SignalRecord`), SHA-256 fixture manifests, `PointInTimeValidator` (causality, corporate actions, survivorship, coverage), and clean-room legacy adapters (`docs/foundation_audit.md`).
- **M2 — Core Signals & Baselines**: Versioned causal factor library (`mom_20_5`, `mom_60_5`, `rev_3d`, `inv_vol_20d`, PIT `val_earnings_yield`, PIT `qual_roe`), causal `FeatureMatrixBuilder`, and statistical/ML baselines (`EqualRankCompositeBaseline`, `RidgeAlphaBaseline`, `GradientBoostedAlphaBaseline`, and isolated optional `XGBoostAlphaBaseline`).
- **M3 — Walk-Forward Evaluation, Neutralization & Costs**: Per-date winsorization + sector/beta/volatility neutralization, strictly chronological walk-forward splits with purge/embargo, OOS Spearman/Pearson IC, ICIR, bootstrap 95% CI, stability diagnostics, pairwise correlation auditing, and cost-aware signal-to-position research across `[0, 5, 10, 25, 50]` bps with turnover smoothing and convex impact.
- **M4 — Versioned Alpha Catalog & Paper-Only Export**: Append-only trial ledger (`trials.jsonl`) recording passing and negative results, Benjamini-Hochberg FDR multiple-testing adjustment, non-bypassable promotion gates (`PromotionGateError`, `FactorDuplicationError`), explicit approval metadata, paper-only exports with SHA-256 digest and hard `ExecutionDisabledError` guard, and structured observability metrics.

## Repository Layout

```text
advanced-alpha-generation-engine/
  plan.md
  README.md
  pyproject.toml / uv.lock
  config/                  # Versioned JSON policies (no secrets; execution_enabled=false)
  data/fixtures/           # Synthetic PIT Parquet fixtures, generator, SHA-256 manifest
  docs/                    # Architecture, foundation audit, API schemas, assumptions, runbook, ADRs
  src/alpha_engine/        # Contracts, PIT data, factors, models, adapters, evaluation, catalog, pipeline
  tests/                   # unit/, integration/, scenarios/ (all 7 Section 5 failure scenarios)
  scripts/                 # setup.sh, demo.py, validate.py
  reports/                 # M1–M4 JSON reports, scenario evidence, catalog, paper exports, acceptance evidence
```

## Documentation

- [Architecture & Unsupported Features](docs/architecture.md)
- [M1 Foundation Audit & IP Status](docs/foundation_audit.md)
- [API & Domain Schema Reference](docs/api_schema.md)
- [Assumptions & Known Limitations](docs/assumptions_and_limitations.md)
- [Operator Runbook & Governance](docs/runbook.md)
- [Acceptance Evidence](reports/acceptance_evidence.md)
