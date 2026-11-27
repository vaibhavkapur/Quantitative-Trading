# Assumptions & Attribution Methodology (`pae-v1.0.0`)

## 1. Return Conventions & External Cash Flows

For each trading date $t$ in reporting currency $C_{\text{rep}}$:

- **Capital Base (Start-of-Day Adjusted):**
  $$B_t = \text{NAV}_{t}^{\text{start}} + \text{CF}_{t}^{\text{SOD}}$$
- **Pre-EOD-Flow Ending NAV:**
  $$E_t = \text{NAV}_{t}^{\text{end}} - \text{CF}_{t}^{\text{EOD}}$$
- **Daily Net P&L & Gross P&L:**
  $$\text{PnL}_{t}^{\text{net}} = E_t - B_t, \qquad \text{PnL}_{t}^{\text{gross}} = \text{PnL}_{t}^{\text{net}} + \text{Fees}_{t}$$
  where $\text{Fees}_t$ is the sum (in $C_{\text{rep}}$) of non-trade position fees, trade explicit costs (`commission + exchange_fee + tax`), and `FeeAdjustment` entries on date $t$.
- **Daily Returns (strict Gross vs. Net separation):**
  $$r_{t}^{\text{gross}} = \frac{\text{PnL}_{t}^{\text{gross}}}{B_t}, \qquad d_{t}^{\text{fee}} = \frac{\text{Fees}_{t}}{B_t}, \qquad r_{t}^{\text{net}} = \frac{\text{PnL}_{t}^{\text{net}}}{B_t} = r_{t}^{\text{gross}} - d_{t}^{\text{fee}}$$
- **Zero-NAV Guard:** If $B_t \le 10^{-12}$, division is skipped, `ZERO_NAV` (`CRITICAL`) is emitted (or `ZeroNavError` raised when `fail_on_zero_nav=true`), and day $t$ is excluded from geometric compounding.
- **Cumulative Multi-Period Return (TWR):**
  $$R = \prod_{t \in \mathcal{V}} (1 + r_t) - 1$$
- **Modified Dietz Cross-Check (Period Net):**
  $$R_{\text{Dietz}} = \frac{\text{NAV}_{T}^{\text{end}} - \text{NAV}_{1}^{\text{start}} - \sum_k \text{CF}_k}{\text{NAV}_{1}^{\text{start}} + \sum_k w_k \text{CF}_k}$$
  where $w_k = (T - i)/T$ for `START_OF_DAY` flows on 0-indexed day $i$ and $w_k = (T - i - 1)/T$ for `END_OF_DAY` flows.

## 2. Benchmark Date & Currency Alignment

For a benchmark denominated in currency $C_b \ne C_{\text{rep}}$ with local daily return $r_{b,t}^{\text{loc}}$ and FX rate moving from $\text{fx}_{t-1}$ to $\text{fx}_t$:

$$r_{t}^{\text{fx}} = \frac{\text{fx}_t}{\text{fx}_{t-1}} - 1, \qquad r_{b,t}^{\text{aligned}} = (1 + r_{b,t}^{\text{loc}})(1 + r_{t}^{\text{fx}}) - 1$$

If any valid trading date lacks an aligned benchmark observation or required FX conversion rate, `MISSING_BENCHMARK` or `BENCHMARK_CURRENCY_MISMATCH` is emitted and cumulative benchmark/active return is withheld rather than misaligning dates.

## 3. Position Contribution, FX Decomposition & Intraday Rebalance

For position $i$ on day $t$ with start quantity $q_0$, end quantity $q_1$, local prices $P_0, P_1$, FX rates $\text{fx}_0, \text{fx}_1$, and intraday trades $\{k\}$ with signed quantities $\Delta q_k$ and execution prices $P_{\text{exec},k}$:

1. **Quantity Bridge Invariant:** $q_0 + \sum_k \Delta q_k = q_1$ (violation emits `QUANTITY_MISMATCH`).
2. **Exact Additive Hold P&L (Karnosky-Singer Identity):**
   - Local price P&L: $q_0 (P_1 - P_0)\,\text{fx}_0$
   - FX translation P&L: $q_0 P_0 (\text{fx}_1 - \text{fx}_0)$
   - FX interaction P&L: $q_0 (P_1 - P_0)(\text{fx}_1 - \text{fx}_0)$
3. **Intraday Rebalance / Trading P&L:** $\sum_k \Delta q_k (P_1 - P_{\text{exec},k})\,\text{fx}_1$
4. **Income P&L:** $\text{income}_{\text{loc}}\,\text{fx}_1$
5. **Position Contributions:** Dividing each P&L component by $B_t$ gives exact additive contributions to portfolio return.

## 4. Multi-Period Carino Linking

Given daily returns $\{r_t\}$ and cumulative return $R = \prod_t (1 + r_t) - 1$:

$$k_t = \begin{cases} \frac{\ln(1+r_t)}{r_t} & |r_t| \ge 10^{-12} \\ 1 & \text{otherwise} \end{cases}, \qquad k = \begin{cases} \frac{\ln(1+R)}{R} & |R| \ge 10^{-12} \\ 1 & \text{otherwise} \end{cases}, \qquad w_t^{\text{Carino}} = \frac{k_t}{k}$$

This satisfies $\sum_t w_t^{\text{Carino}} r_t = R$ identically, so linked strategy/instrument/sector/currency contributions plus cash minus fund fees plus explicit linked residual sum identically to $R$.

## 5. Brinson Sector Attribution

- **Brinson-Fachler (default):**
  - Allocation: $(w_{p,s} - w_{b,s})(r_{b,s} - R_b)$
  - Selection: $w_{b,s}(r_{p,s} - r_{b,s})$
  - Interaction: $(w_{p,s} - w_{b,s})(r_{p,s} - r_{b,s})$
- **Brinson-Hood-Beebower (optional):** Allocation uses $(w_{p,s} - w_{b,s}) r_{b,s}$.
- **Multi-Period Active Linking:** Carino (1999) active weights $k_t^{\text{act}} = (\ln(1+r_{p,t}) - \ln(1+r_{b,t}))/(r_{p,t} - r_{b,t})$ preserve $\text{Alloc} + \text{Select} + \text{Interact} + \text{IntradayRebal} + \text{Residual} = R_p^{\text{gross}} - R_b$.

## 6. Statistical Risk-Factor Attribution

Ordinary Least Squares (with configurable small ridge stabilizer $\lambda = 10^{-10}$ on factor slopes) regresses daily excess gross returns $(r_{t}^{\text{gross}} - r_{f,t})$ on point-in-time factor returns $\{f_{m,t}\}$:

$$r_{t}^{\text{gross}} - r_{f,t} = \alpha + \sum_{m=1}^{K} \beta_m f_{m,t} + \varepsilon_t$$

> **Explicit Non-Goal / Disclosure:** Factor attribution decomposes historical co-movement only; **factor correlation does not prove causation**.

## 7. Execution-Cost Attribution (Perold Implementation Shortfall)

For each trade with signed executed quantity $q_s$ (`+qty` BUY, `-qty` SELL) and signed unfilled quantity $q_u$ (positive = cost/drag):

- **Delay cost:** $q_s (P_{\text{arrival}} - P_{\text{decision}})\,\text{fx}_t$
- **Market impact / spread:** $q_s (P_{\text{exec}} - P_{\text{arrival}})\,\text{fx}_t$
- **Opportunity cost:** $q_u (P_{\text{close}} - P_{\text{decision}})\,\text{fx}_t$
- **Explicit costs:** $(\text{commission} + \text{exchange\_fee} + \text{tax})\,\text{fx}_t$
- **VWAP slippage diagnostic:** $q_s (P_{\text{exec}} - P_{\text{VWAP}})\,\text{fx}_t$

## 8. Reconciliation Tolerances & Data-Quality Exception Catalog

| Exception Code | Severity | Trigger Condition |
|---|---|---|
| `ZERO_NAV` | `CRITICAL` | Start-of-day capital base $B_t \le 0$ |
| `RESIDUAL_EXCEEDS_TOLERANCE` | `ERROR` | Daily gross or net residual exceeds `return_tolerance` (default `1e-6` = `0.01 bps`) |
| `QUANTITY_MISMATCH` | `ERROR` | `start_quantity + sum(signed_trades) != end_quantity` |
| `MISSING_FX_RATE` | `ERROR` | Required FX rate unavailable for non-reporting currency |
| `BENCHMARK_CURRENCY_MISMATCH` | `ERROR` | Benchmark currency differs from reporting currency and FX is missing |
| `MISSING_BENCHMARK` | `WARNING` | Benchmark observation absent for a portfolio trading date |
| `INCOMPLETE_BENCHMARK_SERIES` | `WARNING` | Partial benchmark coverage; cumulative active return withheld |
| `STALE_FX_RATE` | `WARNING` | Prior-date FX carried forward or fixing age exceeds `fx_max_staleness_hours` |
| `LOOKAHEAD_VIOLATION` | `WARNING` | Input `availability_utc > batch.as_of_utc` |
| `ORPHAN_TRADE` | `WARNING` | `TradeExecution` has no matching `PositionSnapshot` on that date |
| `ZERO_END_NAV` | `WARNING` | Portfolio ending NAV reached `0.0` |
| `ROUNDING_RESIDUAL_WITHIN_TOLERANCE` | `INFO` | Sub-tolerance custodian rounding delta detected and surfaced explicitly |
| `FEE_CORRECTION_APPLIED` | `INFO` | Retroactive `FeeAdjustment(is_correction=True)` applied |
| `INSUFFICIENT_FACTOR_OBSERVATIONS` | `INFO` | Aligned factor observations below `factor_min_observations` |
