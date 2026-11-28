# Statistical Arbitrage Research Platform

An offline, reproducible quantitative research platform for evaluating classical equity pairs and cointegrated basket mean reversion under point-in-time universe filtering, strict walk-forward boundaries, realistic transaction and short-borrow costs, and multiple-testing controls.

> **Scope & Disclaimer:** Independently developed educational and research prototype using synthetic fixtures only. Cointegration relationships can break down out of sample, and paper backtests do not imply executable live edge. Live trading, brokerage APIs, real customer data, and production credentials are strictly excluded.

## Key Capabilities

- **Point-in-Time Universe Filtering (`statarb.universe`):** Enforces active listing status, minimum price, median ADV, history depth, and sector rules strictly on `date <= train_end`.
- **Cointegration & Hedge Ratio Estimation (`statarb.cointegration`):** OLS and Total Least Squares (TLS) pair hedge ratios, Engle-Granger & ADF tests, discrete-time OU half-life estimation, 3+ asset Johansen basket cointegration, and Benjamini-Hochberg FDR / Bonferroni multiple-testing adjustments.
- **Pure Causal Signal State Machine (`statarb.signals`):** Entry, mean-reversion exit, stop-loss with cooldown, time-stop, and frozen-training or causal-rolling z-scores isolated from I/O.
- **Walk-Forward Backtest & Exact Reconciliation (`statarb.backtest`):** Rolling or expanding folds with embargo gaps, `T+1` execution delay, commissions, slippage, daily short borrow accrual, hard locate/recall enforcement, delisting liquidation, and mandatory independent trade-to-position & equity replay verification.
- **Evaluation & Multiple-Testing Governance (`statarb.evaluation`, `statarb.pipeline`):** OOS net/gross performance, Lo/Mertens analytical Sharpe SE, circular block-bootstrap confidence intervals, Deflated Sharpe Ratio across all attempted configurations, fold stability & hedge-ratio drift, and benchmark comparisons (equal-weight universe and zero-cost frictionless oracle).

## Project Layout

```text
statistical-arbitrage-research-platform/
├── config/                     # Versioned TOML configurations (default.toml, reference.toml)
├── data/fixtures/              # Deterministic synthetic valid, reference, and invalid fixtures
├── docs/                       # Architecture + ADRs, contracts, integration, runbook
├── reports/                    # Versioned machine-readable JSON/CSV outputs & acceptance.md
├── scripts/                    # generate_fixtures.py, run_demo.py
├── src/statarb/                # Typed library & CLI
├── tests/                      # 30 unit, integration, leakage, boundary & reproducibility tests
├── pyproject.toml              # Package metadata & pinned constraints
└── uv.lock                     # Locked dependency graph
```

CI workflow: [`../.github/workflows/statistical-arbitrage-research-platform.yml`](../.github/workflows/statistical-arbitrage-research-platform.yml)

## Quickstart

### 1. Setup

```bash
cd statistical-arbitrage-research-platform
uv sync --python 3.12 --extra dev --frozen
```

### 2. Generate Fixtures & Run Demo

```bash
uv run python scripts/generate_fixtures.py
uv run statarb validate --config config/default.toml --project-root .
uv run python scripts/run_demo.py
```

### 3. Run Validation Suite

```bash
uv run ruff check src tests scripts
uv run pytest
```

## Measured Synthetic Results

> **Synthetic Data Notice:** All results below are actual measured runs on **deterministic synthetic fixtures** (`data_provenance = "synthetic_fixture"`), **not** real historical market data.

| Metric | Walk-Forward Synthetic Universe (`default.toml`) | Reference Stationary Pair (`reference.toml`) |
|---|---|---|
| Run ID | `default_walkforward_demo_659102efff8c` | `reference_deterministic_fixture_caf5c9d29032` |
| OOS Trading Days / Folds | 360 days / 6 rolling folds | 120 days / 3 rolling folds |
| OOS Net Return (Gross) | `-0.19%` (`+0.12%`) | `+11.02%` (`+11.45%`) |
| OOS Net Sharpe (Gross) | `-0.0653` (`+0.0545`) | `2.8762` (`2.9763`) |
| 95% Bootstrap Sharpe CI | `[-1.6875, +1.5005]` | `[-0.5725, +5.8211]` |
| Max Drawdown (Net) | `-2.42%` | `-3.82%` |
| Annualized Turnover / Trades | `4.19x` / 261 trades | `24.33x` / 187 trades |
| Cost Drag (Comm + Slip + Borrow) | `30.39 bps` (`\$3,051.37`) | `42.49 bps` (`\$400.29`) |
| Deflated Sharpe Prob (Trials) | `0.2786` (4 `z_entry` configs) | `0.9777` (2 `z_entry` configs) |
| Equal-Weight Benchmark Return / Sharpe | `+21.00%` / `1.3984` | `-10.26%` / `-0.8425` |
| Reconciliation (`max_pos_diff` / `max_eq_diff`) | **PASS** (`0.0` / `0.0`) | **PASS** (`0.0` / `0.0`) |

Notice how the multi-sector synthetic universe (which embeds a regime-breaking pair, a mid-sample delisting, a short-borrow outage, and independent random-walk decoys) turns from slightly positive gross (`+0.12%`) to slightly negative net (`-0.19%`) after realistic costs and turnover—highlighting why strict multiple-testing controls, OOS uncertainty bands, and friction accounting are essential.

## Documentation & Acceptance Evidence

- [Acceptance Verification Report](reports/acceptance.md) — Maps every acceptance criterion and test matrix scenario to automated test evidence and measured outputs.
- [Architecture & ADRs](docs/architecture.md) — System diagram, module boundaries, invariants, and design decisions.
- [Data Contracts & Fixture Inventory](docs/contracts.md) — Input/output schemas, corporate action validation, error types, and fixture provenance.
- [Integration & Dependency Licenses](docs/integration.md) — Adapter boundaries and open-source license audit.
- [Operator & Developer Runbook](docs/runbook.md) — Setup, CLI usage, diagnostics, and recovery procedures.
- Raw versioned outputs: [`reports/default_walkforward_demo_659102efff8c_summary.json`](reports/default_walkforward_demo_659102efff8c_summary.json), [`reports/reference_deterministic_fixture_caf5c9d29032_summary.json`](reports/reference_deterministic_fixture_caf5c9d29032_summary.json).

## Assumptions & Known Limitations

- **Daily close-to-next-close execution:** Assumes fills execute at next trading day's adjusted close (`execution_delay_days = 1`) with linear bps slippage; intraday spread dynamics, queue priority, and nonlinear market impact are not modeled.
- **Structural breaks:** Cointegration tests have finite-sample power limits and cannot predict future structural breaks (as demonstrated by `FIN_BREAK_A`/`FIN_BREAK_B` and low OOS stationarity retention in the demo fixture).
- **Fractional adjusted shares & simplified financing:** Positions use fractional split-adjusted share equivalents without lot-size rounding, margin maintenance rules, or short-sale cash rebate interest.
- **Offline synthetic validation only:** No live market data feeds or vendor historical databases were evaluated in this environment.
