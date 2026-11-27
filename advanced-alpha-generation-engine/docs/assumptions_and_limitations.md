# Assumptions and Known Limitations

## Explicit Research Disclaimer

- **No promise of predictive edge or profitable execution.** Simulated or synthetic out-of-sample IC, ICIR, and net Sharpe figures validate software invariants and research methodology only; never infer live trading performance from simulated tests.
- **Synthetic default dataset.** The committed fixtures (`data/fixtures/`) are deterministically generated synthetic equities (`seed=42`, 30 symbols across 5 sectors, 440 trading days, 4 mid-sample delistings, 3 splits/reverse splits, quarterly dividends, and 45-day lagged filings).

## Quantitative & Modeling Assumptions

1. **Point-in-time availability**: Every market bar is assumed available 5 minutes after UTC market close (`21:05 UTC`); fundamental filings are joined strictly backward on `available_at_utc` (`period_end_utc + 45d`).
2. **Walk-forward embargo**: Chronological splits enforce `embargo_days >= forward_horizon_days` (default `5` trading days) between training end and test start to prevent forward-return label leakage.
3. **Neutralization**: Cross-sectional OLS residualization removes sector means, 60-day causal rolling market beta, and 20-day causal realized volatility per trading date before z-score standardization.
4. **Transaction cost model**: Models linear spread/commission (`cost_bps` swept over `[0, 5, 10, 25, 50]` bps) plus a convex market-impact term proportional to `sum(|delta_weight|^1.5)`. It does not yet model intraday volume curves, short-borrow fees, margin financing, or hard-to-borrow constraints.
5. **Delisting liquidation**: When a symbol exits the PIT universe (`exit_ts_utc`), its position is liquidated to `0.0` weight and the full liquidation turnover is charged to trading costs. Distress recovery/delisting return assumptions depend on upstream data vendor accuracy when connected to real datasets.
6. **Multiple testing**: Benjamini-Hochberg FDR adjustment is computed across the append-only trial ledger (`trials.jsonl`). BH controls expected false discovery rate under independent/PRDS test statistics; correlated factor trials are additionally controlled via pairwise correlation auditing (`max_peer_factor_correlation = 0.85`).

## Known Technical Limitations

- **Optional XGBoost native dependency**: On macOS, `xgboost` requires the OpenMP native runtime (`brew install libomp`). The engine explicitly isolates `XGBoostAlphaBaseline` behind `OptionalDependencyError`, while providing a self-contained deterministic `GradientBoostedAlphaBaseline` (`scikit-learn` `HistGradientBoostingRegressor`) that runs with zero OS-level native dependencies.
- **Single-process execution**: Designed for local reproducible research and sandbox validation. Large universes (e.g., 5,000+ equities over 20 years or tick-level data) will require partitioning or out-of-core execution.
- **Order-book adapter proxy**: `QuantumOrderBookAdapter` uses daily OHLCV close-location-value pressure as a clean-room placeholder until licensed L2 market depth feeds are integrated.
- **No live execution or OMS integration**: Intentionally hard-disabled (`execution_enabled=false`) until independent risk, compliance, security, and broker reviews approve sandbox/paper deployment.
