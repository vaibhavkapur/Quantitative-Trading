# Integration Boundaries & Dependency Inventory

## 1. Repository Context (`Trading-Quantitative`)

Sibling projects in the `Trading-Quantitative` repository root:

| Sibling Project Path | State | Integration Boundary |
|---|---|---|
| `multi-factor-asset-pricing-engine/` | Standalone sibling package | Consumer of `MarketDataset` and `oos_signals` for cross-sectional risk premia regression |
| `volatility-forecasting-engine/` | Standalone sibling package | Pluggable covariance/volatility forecaster adapter replacing `estimate_covariance()` |
| `optimal-execution-research-engine/` | Standalone sibling package | Nonlinear slippage/execution simulator consuming `portfolio_weights.csv` rebalance schedules |
| `quantitative-strategy-robustness-lab/` | Standalone sibling package | Robustness harness consuming `daily_returns.csv` and `summary.json` for PSR/DSR/stress tests |
| `statistical-arbitrage-research-platform/` | Standalone sibling package | Independent sibling project; no shared runtime code |

To remain independently installable and avoid fragile cross-project imports, this engine is fully self-contained with zero runtime imports from sibling packages and zero dependency on private employer or proprietary artifacts.

---

## 2. Reusable Public Adapter Interfaces

External or sibling engines should integrate via the typed Python interfaces exported from `cross_asset_factor_engine`:

1. **Data ingestion adapter (`io.py` -> `contracts.py`):**
   - External feeds (Parquet, SQL, vendor REST snapshots in an offline pipeline) convert raw tables into Polars DataFrames conforming to `docs/contracts.md` and pass them to `validate_dataset(...) -> MarketDataset`.
2. **Return normalization adapter (`normalization.py`):**
   - `normalize_market_data(dataset: MarketDataset, annualization_days: int = 252) -> pl.DataFrame`
3. **Signal adapter (`factors.py`):**
   - `compute_raw_factor_signals(normalized_returns, dataset, config) -> pl.DataFrame`
   - `FactorScaler.fit(train_df).transform(eval_df) -> pl.DataFrame`
4. **Portfolio optimizer adapter (`portfolio.py`):**
   - `optimize_constrained_portfolio(signals, cov, prev_weights, asset_classes, portfolio_config) -> OptimizationResult`
   - `verify_portfolio_constraints(...) -> dict`
   - A sibling volatility engine can inject custom ex-ante covariance matrices `cov` directly without modifying optimizer internals.
5. **Evaluation & reporting contract (`evaluation.py`, `reporting.py`):**
   - `run_walk_forward_evaluation(dataset, config) -> EvaluationOutput`
   - Versioned CSV/JSON outputs in `reports/runs/<run_id>/` provide language-agnostic integration boundaries.

---

## 3. Dependency & License Inventory

All dependencies are pinned via `pyproject.toml` and locked in `uv.lock`:

| Package | License | Role |
|---|---|---|
| `polars` | MIT | High-performance tabular transforms, typed schema enforcement, CSV I/O |
| `numpy` | BSD-3-Clause | Vectorized linear algebra, return/risk arrays, seeded synthetic fixture DGP |
| `scipy` | BSD-3-Clause | Constrained convex optimization (`scipy.optimize.minimize` SLSQP) |
| `pyyaml` | MIT | YAML configuration parsing |
| `pytest` (dev) | MIT | Automated unit, integration, boundary, and CLI test runner |
| `hatchling` (build) | MIT | PEP-517 wheel build backend |

---

## 4. Production Exclusions

Any future production deployment is strictly outside scope of this prototype and would require separate:
- Point-in-time market data vendor licensing and corporate action reconciliation
- Order management system (OMS), pre-trade risk gateway, and brokerage connectivity
- Security, compliance, legal, and model-risk governance sign-off
