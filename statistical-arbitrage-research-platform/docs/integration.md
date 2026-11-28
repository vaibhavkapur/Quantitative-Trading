# Integration, Dependencies & Compatibility

## 1. Repository Context & Adapter Boundaries

This project resides at:

`Trading-Quantitative/statistical-arbitrage-research-platform/`

Sibling project directories at the root of `Trading-Quantitative/`:
- `cross-asset-factor-investing-engine/`
- `multi-factor-asset-pricing-engine/`
- `optimal-execution-research-engine/`
- `quantitative-strategy-robustness-lab/`
- `volatility-forecasting-engine/`

At implementation time, sibling projects are independent standalone prototypes without a shared workspace Python package. Per the architecture decision in `plan.md`, this platform has **zero runtime imports from sibling projects** and **zero dependency on private employer or SAAT code**.

### Planned Integration Interfaces (Runnable Adapters)

To compose this platform with neighboring engines without copying implementations, integrate via the versioned contracts in `docs/contracts.md`:

| Direction | Interface | Contract / Data Handoff |
|---|---|---|
| Upstream Market / Factor Data | Parquet or CSV tables conforming to `PRICE_REQUIRED_COLUMNS` and `BORROW_REQUIRED_COLUMNS` | Any external or sibling data-prep pipeline can emit adjusted prices and borrow tables and call `statarb.load_and_validate_inputs(...)` |
| Downstream Robustness Analysis (`quantitative-strategy-robustness-lab`) | `reports/<run_id>_equity_curve.csv` & `reports/<run_id>_summary.json` | Exposes daily net/gross returns, fold boundaries, and `multiple_testing.attempted_configurations` for cross-strategy PBO/DSR evaluation |
| Downstream Execution Modeling (`optimal-execution-research-engine`) | `reports/<run_id>_trades.csv` | Provides dated symbol-level share deltas (`quantity`, `price_adj`, `notional`) for intraday scheduling/impact simulation |
| Programmatic Python API | `statarb` top-level exports (`src/statarb/__init__.py`) | Typed library entry points (`filter_universe`, `select_spreads_on_training_window`, `step_spread_signal`, `run_walk_forward_backtest`, `execute_research_pipeline`) |

## 2. Pinned Dependencies & License Audit

Dependencies are pinned via `pyproject.toml` and `uv.lock` (Python `>=3.11`, verified on CPython `3.12.15`).

| Package | Resolved Version | Role | License |
|---|---|---|---|
| `numpy` | `2.5.5` | Numerical arrays, SVD (TLS), bootstrap & RNG | BSD-3-Clause |
| `polars` | `1.44.2` | Columnar DataFrame I/O, validation, and point-in-time filtering | MIT |
| `statsmodels` | `0.15.0` | OLS, Engle-Granger `coint`, ADF `adfuller`, Johansen `coint_johansen` | BSD-3-Clause |
| `scipy` | `1.18.1` | Normal CDF/PPF for Deflated Sharpe Ratio | BSD-3-Clause |
| `pydantic` | `2.14.4` | Strict typed configuration schema validation | MIT |
| `pytest` (dev) | `8.4.2` | Automated test runner | MIT |
| `ruff` (dev) | `0.17.6` | Linting and code formatting | MIT |

All runtime and development dependencies use permissive open-source licenses (MIT / BSD-3-Clause).

## 3. Out-of-Scope Production Integrations

Live brokerage connectivity, real-time market data adapters, order management systems (OMS), real locate/prime-broker APIs, production credential stores, and live capital deployment are intentionally excluded from this prototype.
