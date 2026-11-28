# Strategy Robustness Evaluation Report — `exp-known-signal-v1`

- **Run ID:** `run-f525e1d44783dfe9`
- **Candidate Trial:** `trial_lb20_vt10`
- **Promotion Verdict:** **`PROMOTE`**
- **Data Provenance:** `synthetic_fixture`
- **Registry Head Hash:** `ed76659d7f001bfc...`
- **Policy Version / Seed:** `v1.0.0` / `seed=42`

> **Summary:** PROMOTED (research/paper stage): Candidate 'trial_lb20_vt10' passed all 16 robustness gates under policy v1.0.0.

## 1. Promotion Gate Scorecard

| Gate | Category | Result | Observed | Threshold |
|---|---|---|---|---|
| `evidence_completeness` | governance | **PASS** | `complete` | `all 7 diagnostics present` |
| `registry_and_trial_integrity` | governance | **PASS** | `{"registry_valid": true, "duplicate_fraction": 0.0, "grid_coverage": 1.0, "flags": 0}` | `valid hash chain, no sequence gaps, duplicates & grid coverage within policy` |
| `deflated_sharpe_ratio` | multiple_testing | **PASS** | `1.0` | `>= 0.95` |
| `psr_vs_benchmark` | multiple_testing | **PASS** | `1.0` | `>= 0.9` |
| `probability_of_backtest_overfitting` | multiple_testing | **PASS** | `0.0857` | `<= 0.25` |
| `pbo_oos_loss_probability` | multiple_testing | **PASS** | `0.0` | `<= 0.3` |
| `bootstrap_sharpe_ci_lower` | uncertainty | **PASS** | `4.138` | `>= 0.0` |
| `walk_forward_oos_sample_size` | out_of_sample | **PASS** | `200` | `>= 40` |
| `walk_forward_oos_net_sharpe` | out_of_sample | **PASS** | `5.1778` | `>= 0.35` |
| `walk_forward_is_to_oos_degradation` | out_of_sample | **PASS** | `0.1203` | `<= 0.65` |
| `walk_forward_oos_vs_benchmark` | out_of_sample | **PASS** | `4.7268` | `>= 0.0` |
| `perturbation_neighbor_coverage` | perturbation | **PASS** | `7` | `>= 1 local neighbor` |
| `perturbation_positive_neighbors` | perturbation | **PASS** | `1.0` | `>= 0.65` |
| `perturbation_sharpe_cliff` | perturbation | **PASS** | `0.278` | `<= 0.5` |
| `perturbation_stability_score` | perturbation | **PASS** | `0.861` | `>= 0.6` |
| `outlier_dominance_check` | tail_risk | **PASS** | `{"n_extreme": 0, "excess_kurtosis": 0.52, "winsorized_sharpe": 5.925, "impact_pct": -0.018}` | `no extreme outliers with excess kurtosis > 20.0 or >50% winsorized Sharpe drop` |

## 2. Transaction Costs & Net Performance

- **Total Gross Compounded Return:** `125.3943%`
- **Total Net Compounded Return:** `123.4622%`
- **Mean Per-Period Cost Drag:** `0.24 bps`
- **Annualized Cost Drag:** `0.6040%`

## 3. Multiple-Testing Diagnostics (DSR & PSR)

| Metric | Value |
|---|---|
| Observed Net Sharpe (Annualized) | `5.8200` |
| Probabilistic Sharpe Ratio vs Benchmark | `1.0000` |
| Deflated Sharpe Ratio (DSR) | `1.0000` |
| Raw Logged Trials / Effective Trials ($N_{eff}$) | `9` / `2.72` |
| Expected Max Null Sharpe ($SR^*$, Annualized) | `0.9846` |
| Skewness / Excess Kurtosis | `-0.155` / `0.515` |
| Autocorrelation Variance Inflation Factor (VIF) | `1.017` |

## 4. Probability of Backtest Overfitting (CSCV PBO)

| Metric | Value |
|---|---|
| Partitions ($S$) / Splits Evaluated | `8` / `70` (exact=True) |
| Probability of Backtest Overfitting (PBO) | `0.0857` |
| Probability of OOS Loss | `0.0000` |
| Median OOS Rank Logit | `0.6931` |
| Mean IS vs OOS Selected Sharpe | `5.9807` -> `5.2807` |
| IS-to-OOS Degradation Slope | `-0.7283` |

## 5. Block Bootstrap Uncertainty Bounds

- **Method:** `stationary` (block_length=`10`, resamples=`1000`, seed=`42`)
- **Annualized Sharpe 95% CI:** `[4.1380, 7.7682]` (SE=`0.9126`)
- **Annualized Mean Return 95% CI:** `[41.7236%, 72.2719%]`
- **Max Drawdown Observed / 95% Upper Bound:** `3.29%` / `5.28%`
- **Empirical Bootstrap p-value ($SR \le 0$):** `0.0000`

## 6. Walk-Forward Out-of-Sample Evaluation

- **Mode:** `expanding` | train=`120` | purge=`5` | test=`40` | folds=`5` | OOS obs=`200`
- **Mean IS Net Sharpe:** `5.8857`
- **Aggregate OOS Net Sharpe:** `5.1778` (Gross: `5.2570`)
- **Aggregate OOS Benchmark Sharpe:** `0.4510`
- **IS-to-OOS Degradation Ratio:** `0.1203` | **Fold Hit Rate:** `100.0%`

| Fold | Train Window | Test Window | Selected Trial | Train Net SR | Test Net SR | Bench SR |
|---|---|---|---|---|---|---|
| 0 | `2024-01-02` .. `2024-06-17` | `2024-06-25` .. `2024-08-19` | `trial_lb20_vt08` | `5.112` | `8.162` | `-3.509` |
| 1 | `2024-01-02` .. `2024-08-12` | `2024-08-20` .. `2024-10-14` | `trial_lb25_vt10` | `5.924` | `4.924` | `1.254` |
| 2 | `2024-01-02` .. `2024-10-07` | `2024-10-15` .. `2024-12-09` | `trial_lb20_vt08` | `6.347` | `6.524` | `0.882` |
| 3 | `2024-01-02` .. `2024-12-02` | `2024-12-10` .. `2025-02-03` | `trial_lb20_vt08` | `5.956` | `4.583` | `4.382` |
| 4 | `2024-01-02` .. `2025-01-27` | `2025-02-04` .. `2025-03-31` | `trial_lb20_vt10` | `6.090` | `3.916` | `1.194` |

## 7. Parameter Perturbation Stability

- **Center Trial:** `trial_lb20_vt10` (`{'lookback': 20, 'vol_target': 0.1}`, net SR=`5.8200`)
- **Neighbors Evaluated:** `7` | **Positive Fraction:** `100.0%`
- **Median / Min Neighbor Net Sharpe:** `4.2018` / `2.4302`
- **Sharpe Cliff Ratio:** `0.2780` | **Stability Score:** `0.8610`

## 8. Outlier & Tail-Risk Diagnostics

- **Max |Return| / Max Robust |z|:** `0.0181` / `3.45` (`0` extreme obs)
- **Raw vs 1%-Winsorized Annualized Sharpe:** `5.8200` vs `5.9254` (impact=`-1.8%`)

## 9. Registry Integrity & Cherry-Picking Audit

- **Hash-Chain Valid:** `True` | **Audit Passed:** `True`
- **Logged Trials:** `9` (completed=`8`, failed/rejected=`1`)
- **Duplicate Fraction:** `0.00%` (exact=`0`, near=`0`)
- **Grid Coverage:** `9/9` (`100.0%`)

## 10. Review Notes & Warnings

- None.

## 11. Methodology Assumptions & Limitations

> Statistical diagnostics (DSR, CSCV PBO, block bootstrap, walk-forward, perturbation) evaluate historical consistency under stated stationarity and dependence assumptions. They do NOT prove future returns. Offline research prototype only.

### Key Recorded Assumptions

- **[DSR]** Stationary return distribution over the evaluation window.
- **[DSR]** Sharpe ratio estimator standard error adjusts for sample skewness and Pearson kurtosis (Mertens 2002).
- **[DSR]** Autocorrelation adjusted via Bartlett-weighted VIF up to 5 lags (Lo 2002).
- **[DSR]** Effective independent trial count estimated via correlation matrix eigenvalue participation ratio.
- **[DSR]** Expected maximum Sharpe under null uses Bailey & Lopez de Prado (2014) extreme-value approximation; does not prove future alpha.
- **[PBO]** CSCV assumes return blocks are approximately stationary and exchangeable across time partitions.
- **[PBO]** Strong serial dependence or regime breaks spanning partition boundaries can distort PBO.
- **[PBO]** OOS relative rank uses (rank / (N + 1)) with average tie-breaking so logits remain finite.
- **[PBO]** Low PBO indicates consistency across historical partitions, not a guarantee of future out-of-sample returns.
- **[BOOTSTRAP]** Uses stationary block bootstrap (mean/fixed block_length=10) to preserve local serial dependence.
- **[BOOTSTRAP]** Assumes weak stationarity of the return-generating process across the sample.
- **[BOOTSTRAP]** Bootstrap percentile intervals reflect historical sampling variability, not structural regime shifts.
- **[WALK_FORWARD]** Walk-forward uses expanding training windows with a 5-period purge gap before each OOS test fold.
- **[WALK_FORWARD]** Candidate selection is performed strictly using training-window net Sharpe ratios.
- **[WALK_FORWARD]** Stitched OOS metrics assume non-overlapping test folds when step_size >= test_window.

### Runtime Environment

- Python `3.12.15` | NumPy `2.5.5` | SciPy `1.18.1` | pandas `2.3.3`
- Policy SHA-256: `43727f3390f894e0a59231cfed79907f5c0c63f3f2d2e4ced816c5cb510eca75`
