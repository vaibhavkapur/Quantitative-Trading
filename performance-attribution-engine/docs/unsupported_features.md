# Explicit Non-Goals & Unsupported Features (MVP Scope)

Per [plan.md](../plan.md) Sections 1 and 8, the following items are **explicitly out of scope or unsupported** in this MVP release:

1. **No Causal Claims from Factor Attribution:** OLS risk-factor regressions quantify historical statistical co-movement only. PAE never asserts that factor correlation proves causation.
2. **No Gross-as-Net Representation:** Gross returns and net returns are strictly separated in configuration, calculations, exports, and dashboards (`gross_vs_net_strict_separation=True` cannot be disabled).
3. **No Order Execution / OMS Connectivity:** Order routing, execution, or live rebalancing is permanently disabled (`order_execution_enabled=False`).
4. **Unsupported Asset-Class & Analytics Features in MVP:**
   - Fixed-income duration/convexity/key-rate-duration (Campisi) attribution.
   - Options / non-linear derivatives Greeks attribution (Delta/Gamma/Vega/Theta decomposition).
   - Holdings-based Barra/Axioma commercial factor covariance models (requires vendor data licensing).
   - Intraday sub-minute tick-by-tick mark-to-market return chaining (current granularity is daily TWR with intraday trade mark-to-close attribution).
   - Automated FX hedging / forward-points carry-vs-spot currency attribution (Karnosky-Singer interest-rate differential decomposition).
   - Live production multi-node cluster orchestration, SSO/OIDC token verification middleware (current API accepts authenticated gateway headers `X-Actor-Id` / `X-Actor-Role`).
   - Tax-lot specific realized vs. unrealized capital gains attribution (FIFO/LIFO/HIFO).
