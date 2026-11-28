# Quantitative Trading

Eleven independent Python projects use synthetic market, factor, and portfolio data to detect market regimes, evaluate alpha and cross-asset factor signals, estimate asset-pricing premia, test statistical arbitrage and volatility forecasts, compare optimal execution schedules, audit strategy robustness, allocate capital, reconcile positions, and explain portfolio returns.

## Getting started

Start with the Market Regime Detection Engine. Its offline demo requires Python 3.9 or later with SQLite. It compares a threshold rule and a hidden Markov model on the same sequence of daily bars.

```bash
git clone https://github.com/vaibhavkapur/Trading-Quantitative.git
cd Trading-Quantitative/market-regime-detection-engine
./scripts/setup
./scripts/demo
```

The demo prints a run ID and transition counts. Open `.local/evaluation.json` for the full results, or read the [checked-in evaluation summary](market-regime-detection-engine/reports/acceptance.md). Each project has its own dependencies, demo, and validation commands. Run them from that project's directory.

## Projects

- [Market Regime Detection Engine](market-regime-detection-engine/) — Labels daily conditions as rising, falling, stressed, or unknown and lets you query the labels available at a given time.
- [Advanced Alpha Generation Engine](advanced-alpha-generation-engine/) — Tests signals on later time windows, measures the effect of trading costs, and records which signals qualify for paper evaluation.
- [Multi-Factor Asset Pricing Engine](multi-factor-asset-pricing-engine/) — Estimates time-series CAPM and multi-factor regressions and Fama-MacBeth cross-sectional risk premia with HAC and Shanken standard errors, GRS tests, and zero-leakage out-of-sample pricing evaluation.
- [Cross-Asset Factor Investing Engine](cross-asset-factor-investing-engine/) — Normalizes equities, FX, and commodity futures into comparable excess returns, builds point-in-time momentum, carry, and defensive signals, and evaluates constrained volatility-scaled portfolios in walk-forward backtests.
- [Statistical Arbitrage Research Platform](statistical-arbitrage-research-platform/) — Evaluates equity pairs and cointegrated baskets under point-in-time universe filtering, walk-forward boundaries, transaction and short-borrow costs, trade-to-position reconciliation, and multiple-testing controls.
- [Volatility Forecasting Engine](volatility-forecasting-engine/) — Compares naive, rolling, EWMA, GARCH(1,1), and machine-learning realized-volatility baselines across purged chronological splits with QLIKE and MAE calibration, conformal forecast intervals, and regime diagnostics.
- [Optimal Execution Research Engine](optimal-execution-research-engine/) — Compares TWAP, VWAP, participation, and analytic or numeric Almgren-Chriss schedules under spread, permanent and temporary impact, slice capacity limits, price shocks, and Monte Carlo timing risk.
- [Quantitative Strategy Robustness Lab](quantitative-strategy-robustness-lab/) — Audits candidate strategies for overfitting and cherry-picking using an immutable hash-chained trial registry, Deflated Sharpe Ratio, CSCV Probability of Backtest Overfitting, block bootstrap confidence intervals, purged walk-forward tests, and fail-closed promotion gates.
- [Dynamic Capital Allocation Engine](dynamic-capital-allocation-engine/) — Compares allocation methods under liquidity, turnover, and risk limits. A second reviewer approves proposed changes before they reach the paper ledger.
- [Multi-Strategy Portfolio Management System](multi-strategy-portfolio-management-system/) — Replays fills, fees, and corporate actions to reconcile positions, cash, and P&L across strategy accounts and the fund.
- [Performance Attribution Engine](performance-attribution-engine/) — Breaks portfolio returns down by strategy, instrument, sector, and currency, then reconciles contributions and reports execution costs.

Each project README links to its architecture, data contracts or API reference, operator runbook, and validation evidence.
