# Quantitative Trading

Five independent Python projects use synthetic market and portfolio data to identify market regimes, evaluate trading signals, allocate capital, reconcile positions, and explain portfolio returns.

## Getting started

Start with the Market Regime Detection Engine. Its offline demo requires Python 3.9 or later with SQLite. It compares a threshold rule and a hidden Markov model on the same sequence of daily bars.

```bash
git clone https://github.com/vaibhavkapur/Quantitative-Trading.git
cd Quantitative-Trading/market-regime-detection-engine
./scripts/setup
./scripts/demo
```

The demo prints a run ID and transition counts. Open `.local/evaluation.json` for the full results, or read the [checked-in evaluation summary](market-regime-detection-engine/reports/acceptance.md). Each project has its own dependencies, demo, and validation commands. Run them from that project's directory.

## Projects

- [Market Regime Detection Engine](market-regime-detection-engine/) — Labels daily conditions as rising, falling, stressed, or unknown and lets you query the labels available at a given time.
- [Advanced Alpha Generation Engine](advanced-alpha-generation-engine/) — Tests signals on later time windows, measures the effect of trading costs, and records which signals qualify for paper evaluation.
- [Dynamic Capital Allocation Engine](dynamic-capital-allocation-engine/) — Compares allocation methods under liquidity, turnover, and risk limits. A second reviewer approves proposed changes before they reach the paper ledger.
- [Multi-Strategy Portfolio Management System](multi-strategy-portfolio-management-system/) — Replays fills, fees, and corporate actions to reconcile positions, cash, and P&L across strategy accounts and the fund.
- [Performance Attribution Engine](performance-attribution-engine/) — Breaks portfolio returns down by strategy, instrument, sector, and currency, then reconciles contributions and reports execution costs.

Each project README links to its architecture, data contracts or API reference, operator runbook, and validation evidence.
