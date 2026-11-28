# Integration & Dependency Notes

## 1. Repository Context (`Trading-Quantitative`)

This project resides at `optimal-execution-research-engine/` at the root of the `Trading-Quantitative` repository alongside independent sibling research projects, including:

- `multi-factor-asset-pricing-engine/`
- `cross-asset-factor-investing-engine/`
- `quantitative-strategy-robustness-lab/`
- `statistical-arbitrage-research-platform/`
- `volatility-forecasting-engine/`

Each project in `Trading-Quantitative` is maintained as a **self-contained, independently runnable Python package** (no shared root workspace package). Therefore, `optimal_execution` exposes clean adapter boundaries rather than fragile cross-directory imports.

## 2. Adapter Boundaries for Future Sibling Integration

If downstream or upstream sibling projects are implemented later, integrate via typed contract adapters (never by duplicating core schedule/simulation logic):

1. **Upstream Volatility Forecasting Engine (`volatility-forecasting-engine`)**:
   - Adapter target: map forecasted intraday/daily volatility $\hat{\sigma}_{\text{day}}$ into `MarketProfile.daily_volatility`.
2. **Upstream Portfolio / Factor / Stat-Arb Engines**:
   - Adapter target: convert target rebalance trade lists `(symbol, side, quantity, arrival_price)` into `ParentOrder` instances and query `simulate_monte_carlo` for realistic pre-trade transaction cost estimates (`mc_mean_is_bps`, `mc_cvar_95_bps`).
3. **Downstream Robustness Lab (`quantitative-strategy-robustness-lab`)**:
   - Adapter target: consume `reports/sensitivity_grid.csv` or call `run_sensitivity_analysis(...)` to evaluate strategy net-of-cost degradation across stressed liquidity regimes.

## 3. External Dependencies & Licenses

All runtime and test dependencies are open-source scientific Python packages pinned in `requirements.lock`:

| Package | Role | License |
|---|---|---|
| `numpy` | Vectorized numeric arrays & PCG64 fixed-seed PRNG | BSD-3-Clause |
| `scipy` | Constrained SLSQP optimization (`scipy.optimize.minimize`) | BSD-3-Clause |
| `polars` | Typed tabular outputs & CSV report export | MIT |
| `pytest` (dev) | Unit, integration, boundary, and CLI test suite | MIT |

## 4. Compatibility & Scope Boundaries

- **Python compatibility**: Tested on Python `3.9+` (compatible with macOS system Python 3.9 and modern Python 3.10–3.13).
- **No external credentials or live execution**: No broker APIs, market data subscriptions, FIX engines, or proprietary code are referenced or required. Any future production trading integration requires independent security, operational, legal, and risk review.
