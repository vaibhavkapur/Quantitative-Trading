# Cross-Asset Factor Investing Engine

An independently developed, offline quantitative research prototype for constructing comparable **momentum**, **carry**, and **defensive** factor signals across **equities**, **FX**, and **commodities**, normalizing heterogeneous returns/rolls/currencies, and evaluating volatility-scaled constrained portfolios in walk-forward out-of-sample backtests.

> **Scope & Provenance Disclaimer:** All datasets and metrics reported here come from deterministic synthetic research fixtures (`data/fixtures/valid_small/` and `data/fixtures/synthetic_multi_year/`). Measured returns, Sharpe ratios, and drawdowns validate software correctness and cost accounting — they are **not** live market performance or investment recommendations. Live capital, real customer records, credentials, and proprietary artifacts are strictly excluded.

---

## Key Capabilities

- **Cross-Asset Normalization (`normalization.py`):** Converts equities (USD/EUR/JPY futures or spot), FX forwards (standard and reverse-quoted pairs such as `USDJPY`), and commodity futures into comparable base-currency (`USD`) excess returns with zero contract-splice jump contamination and explicit roll & financing cost accounting.
- **Documented Cross-Asset Factors with Timestamp Gates (`factors.py`):**
  - **Momentum:** Volatility-adjusted skip-horizon cumulative excess return.
  - **Carry:** Explicit class-specific conventions — `dividend_minus_financing` (equities), `interest_rate_differential` (FX), and `futures_term_structure_roll` (commodities).
  - **Defensive:** Low-risk / downside semi-volatility blended with lagged point-in-time `earnings_quality` for equities (with auditable low-vol fallback or strict error mode).
  - **As-Of Provenance:** Every signal carries `as_of_ts`, `source_max_publication_ts`, and `provenance_hash`, strictly enforcing `source_max_publication_ts <= as_of_ts`.
  - **Training-Only Scaling:** `FactorScaler` fits within-asset-class z-score parameters strictly on each walk-forward fold's training window.
- **Constrained Volatility-Scaled Portfolio Construction (`portfolio.py`):** Exact split-variable convex SLSQP optimizer enforcing single-asset weight caps, gross exposure cap, net exposure band, per-asset-class gross caps, ex-ante volatility cap, and one-way turnover cap — with strict post-solve verification (`verify_portfolio_constraints`).
- **Walk-Forward Evaluation & Reporting (`evaluation.py`, `reporting.py`):** Rolling/anchored folds with embargo gap, `execution_lag_days >= 1`, equal-weight and inverse-volatility benchmarks, full cost/turnover attribution by asset class, structured JSONL diagnostics, and deterministic replay verification.

---

## Project Layout

```text
cross-asset-factor-investing-engine/
├── config/                         # Versioned YAML configurations (default & reference fixture)
├── data/fixtures/                  # Valid (with analytical oracle), synthetic multi-year, invalid & boundary fixtures
├── docs/
│   ├── architecture.md             # Component diagram, data flow, invariants & trade-offs
│   ├── contracts.md                # Canonical schemas, units, carry definitions & error codes
│   ├── integration.md              # Repository boundaries, adapters & dependency licenses
│   ├── runbook.md                  # Developer/operator commands, diagnostics & recovery steps
│   └── adr/0001-cross-asset-normalization-and-asof-gating.md
├── reports/
│   ├── acceptance.md               # Acceptance criteria mapped to verified test & benchmark evidence
│   └── runs/                       # Versioned raw JSON/CSV evaluation outputs & runtime metadata
├── scripts/
│   ├── generate_fixtures.py        # Reproducible synthetic fixture & analytical oracle generator
│   ├── run_demo.py                 # End-to-end walk-forward demo & replay verification
│   └── run_acceptance.py           # Full validation & acceptance command runner
├── src/cross_asset_factor_engine/  # Pure typed library + `cafe` CLI
├── tests/                          # 25 unit, integration, boundary, reproducibility & CLI tests
├── plan.md                         # Original implementation plan
├── pyproject.toml                  # Build metadata & pinned dependency ranges
└── uv.lock                         # Exact locked dependency versions
```

CI workflow: [`../.github/workflows/cross-asset-factor-investing-engine.yml`](../.github/workflows/cross-asset-factor-investing-engine.yml)

---

## Quickstart & Validation Commands

Run all commands from `cross-asset-factor-investing-engine/`:

### 1. Environment Setup
```bash
python3 -m uv sync --python 3.12
```

### 2. Validate Fixtures Against Data Contracts
```bash
.venv/bin/cafe validate --dataset data/fixtures/valid_small
.venv/bin/cafe validate --dataset data/fixtures/synthetic_multi_year
```

### 3. Run End-to-End Demo & Deterministic Replay Check
```bash
.venv/bin/python scripts/run_demo.py
.venv/bin/cafe verify-replay --config config/default.yaml
```

### 4. Run Full Test & Acceptance Suite
```bash
.venv/bin/pytest tests -v
# Or run the full chained acceptance runner:
.venv/bin/python scripts/run_acceptance.py
```

---

## Example Demo Output (Synthetic Multi-Year Fixture)

Measured from verified run `default_cross_asset_walkforward_a2c4def8_c420020e` (6 rolling walk-forward folds, 755 OOS days, 9 synthetic assets across equities/FX/commodities, `seed=42`, Python 3.12.15, Polars 1.44.2, SciPy 1.18.1):

| Strategy | Net Ann Return | Ann Vol | Net Sharpe | Max Drawdown | Ann Turnover | Total Cost Drag | Constraint Violations |
|---|---:|---:|---:|---:|---:|---:|---:|
| `momentum` | 6.96% | 8.57% | 0.83 | -7.65% | 6.98x | 0.69% | 0 / 151 |
| `carry` | 4.61% | 7.52% | 0.64 | -12.36% | 3.35x | 0.58% | 0 / 151 |
| `defensive` | 4.82% | 7.61% | 0.66 | -8.42% | 6.75x | 0.67% | 0 / 151 |
| **`blended`** | **9.23%** | **7.86%** | **1.16** | **-7.67%** | **6.35x** | **0.64%** | **0 / 151** |
| `benchmark_equal_weight` | 8.57% | 9.70% | 0.90 | -11.93% | 0.61x | 0.34% | N/A (unconstrained) |
| `benchmark_inv_vol` | 9.04% | 10.87% | 0.85 | -13.54% | 1.70x | 0.45% | N/A (unconstrained) |

Full per-asset-class (`equity`, `fx`, `commodity`) attribution, constraint audit, signal provenance audit (`27,180` signals, `0` look-ahead violations), and raw artifacts are documented in [`reports/acceptance.md`](reports/acceptance.md) and stored in `reports/runs/default_cross_asset_walkforward_a2c4def8_c420020e/`.

---

## Documentation & Acceptance Evidence

- **Acceptance Evidence:** [`reports/acceptance.md`](reports/acceptance.md)
- **Architecture & Data Flow:** [`docs/architecture.md`](docs/architecture.md)
- **Data Contracts & Carry Formulas:** [`docs/contracts.md`](docs/contracts.md)
- **ADR-0001 (Normalization & Gating):** [`docs/adr/0001-cross-asset-normalization-and-asof-gating.md`](docs/adr/0001-cross-asset-normalization-and-asof-gating.md)
- **Integration & Dependency Licenses:** [`docs/integration.md`](docs/integration.md)
- **Operator & Developer Runbook:** [`docs/runbook.md`](docs/runbook.md)

---

## Assumptions & Known Limitations

1. **Synthetic Data Only:** Fixtures use a deterministic synthetic data-generating process with realistic rolls, FX conversions, and 15-day equity fundamental publication lags. Real-world market regimes, corporate actions, dividend ex-dates, and repo specials require an external point-in-time data adapter.
2. **Linear Cost Model:** Models proportional one-way transaction costs (`tc_bps`), discrete futures roll slippage (`roll_cost_bps`), and daily financing spreads (`financing_spread_bps`). Nonlinear market impact and intraday liquidity limits are excluded.
3. **Covariance Estimation:** Uses trailing shrinkage-regularized sample covariance fit strictly on past returns; dynamic volatility forecasting (GARCH / HAR-RV) is left to pluggable adapters (`docs/integration.md`).
4. **Non-Production Scope:** Offline educational/research prototype only. Any production deployment requires independent security, legal, compliance, operational, and model-risk review.
