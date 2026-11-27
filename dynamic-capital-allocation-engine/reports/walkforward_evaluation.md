# Walk-Forward Evaluation Report

> **Disclaimer:** Synthetic/historical walk-forward diagnostic only; never infer live performance or target Sharpe ratio from simulated evaluation.

## Evaluation Configuration

- **Data source:** `dcae.fixtures.generate_walkforward_strategies` (`synthetic_internal`, seed `20260409`)
- **Period:** `2025-02-13T21:00:00Z` to `2025-09-10T21:00:00Z` (149 trading days, warmup 30 days)
- **Rebalance cadence:** every 5 trading days (30 rebalances, lookback 60 days)
- **Initial paper capital:** \$10,000,000.00 USD
- **Identical cost model:** all allocators pay the exact same per-strategy transaction cost schedule and cash reserve rules.

## Comparative Results (Net of Transaction Costs)

| Allocator | Final NAV (USD) | Cum. Return | Ann. Return | Ann. Vol | Sharpe (Diag.) | Max Drawdown | Avg 1-Way Turnover | Total Costs (USD) | Avg Cash | Fallbacks | Verifier Violations | Failed Steps | Stress Steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `equal_weight` | \$8,711,281.47 | -12.89% | -20.81% | 11.33% | -1.84 | -21.22% | 2.16% | \$11,210.13 | 5.00% | 0 | 10 | 0 | 19 |
| `inverse_volatility` | \$8,633,504.83 | -13.66% | -22.00% | 10.16% | -2.17 | -20.41% | 3.25% | \$14,375.53 | 5.00% | 0 | 11 | 0 | 19 |
| `risk_budget` | \$8,513,752.09 | -14.86% | -23.82% | 9.70% | -2.46 | -19.61% | 4.19% | \$17,312.38 | 5.00% | 0 | 5 | 0 | 19 |
| `constrained_mv` | \$8,909,597.04 | -10.90% | -17.74% | 7.55% | -2.35 | -13.86% | 8.32% | \$39,572.27 | 30.21% | 0 | 0 | 0 | 19 |
| `constrained_mv_stressed` | \$8,881,181.41 | -11.19% | -18.18% | 7.51% | -2.42 | -14.13% | 8.33% | \$39,502.20 | 30.72% | 0 | 0 | 0 | 30 |

## Observations & Honest Reporting

- Baselines (`equal_weight`, `inverse_volatility`, `risk_budget`) do not enforce turnover, ADV liquidity, group exposure, or portfolio volatility constraints; any resulting hard constraint violations are counted transparently under **Verifier Violations**.
- `constrained_mv` and `constrained_mv_stressed` enforce all constraints independently verified at every rebalance step (zero hard verifier violations when feasible, safe fallback otherwise).
- During the injected correlated drawdown window (days 90-110), automatic drawdown stress triggers on `constrained_mv` (`stress_active_steps`), while `constrained_mv_stressed` keeps correlation stress permanently active.
- These metrics reflect synthetic test series only and are **not** evidence of live trading performance.
