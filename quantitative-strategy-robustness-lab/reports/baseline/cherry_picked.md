# Strategy Robustness Evaluation Report — `exp-cherry-picked-v1`

- **Run ID:** `run-fc2ee4de32c3f2e8`
- **Candidate Trial:** `cherry_seq4`
- **Promotion Verdict:** **`REJECT`**
- **Data Provenance:** `synthetic_fixture`
- **Registry Head Hash:** `30a9be12f5b73282...`
- **Policy Version / Seed:** `v1.0.0` / `seed=42`

> **Summary:** REJECTED: Trial registry integrity or cherry-picking audit failed.

## 1. Promotion Gate Scorecard

| Gate | Category | Result | Observed | Threshold |
|---|---|---|---|---|
| `evidence_completeness` | governance | **PASS** | `complete` | `all 7 diagnostics present` |
| `registry_and_trial_integrity` | governance | **FAIL** | `{"registry_valid": true, "duplicate_fraction": 0.2, "grid_coverage": 0.5555555555555556, "flags": 4}` | `valid hash chain, no sequence gaps, duplicates & grid coverage within policy` |
| `deflated_sharpe_ratio` | multiple_testing | **PASS** | `0.9988` | `>= 0.95` |
| `psr_vs_benchmark` | multiple_testing | **PASS** | `1.0` | `>= 0.9` |
| `probability_of_backtest_overfitting` | multiple_testing | **FAIL** | `0.3571` | `<= 0.25` |
| `pbo_oos_loss_probability` | multiple_testing | **PASS** | `0.0` | `<= 0.3` |
| `bootstrap_sharpe_ci_lower` | uncertainty | **PASS** | `1.6972` | `>= 0.0` |
| `walk_forward_oos_sample_size` | out_of_sample | **PASS** | `200` | `>= 40` |
| `walk_forward_oos_net_sharpe` | out_of_sample | **PASS** | `3.2638` | `>= 0.35` |
| `walk_forward_is_to_oos_degradation` | out_of_sample | **PASS** | `-0.1435` | `<= 0.65` |
| `walk_forward_oos_vs_benchmark` | out_of_sample | **PASS** | `1.2064` | `>= 0.0` |
| `perturbation_neighbor_coverage` | perturbation | **PASS** | `3` | `>= 1 local neighbor` |
| `perturbation_positive_neighbors` | perturbation | **PASS** | `1.0` | `>= 0.65` |
| `perturbation_sharpe_cliff` | perturbation | **FAIL** | `0.5273` | `<= 0.5` |
| `perturbation_stability_score` | perturbation | **PASS** | `0.7363` | `>= 0.6` |
| `outlier_dominance_check` | tail_risk | **PASS** | `{"n_extreme": 0, "excess_kurtosis": 0.51, "winsorized_sharpe": 3.089, "impact_pct": -0.014}` | `no extreme outliers with excess kurtosis > 20.0 or >50% winsorized Sharpe drop` |

## 2. Transaction Costs & Net Performance

- **Total Gross Compounded Return:** `64.5551%`
- **Total Net Compounded Return:** `63.1898%`
- **Mean Per-Period Cost Drag:** `0.23 bps`
- **Annualized Cost Drag:** `0.5840%`

## 3. Multiple-Testing Diagnostics (DSR & PSR)

| Metric | Value |
|---|---|
| Observed Net Sharpe (Annualized) | `3.0471` |
| Probabilistic Sharpe Ratio vs Benchmark | `1.0000` |
| Deflated Sharpe Ratio (DSR) | `0.9988` |
| Raw Logged Trials / Effective Trials ($N_{eff}$) | `5` / `3.54` |
| Expected Max Null Sharpe ($SR^*$, Annualized) | `0.8175` |
| Skewness / Excess Kurtosis | `0.108` / `0.509` |
| Autocorrelation Variance Inflation Factor (VIF) | `0.762` |

## 4. Probability of Backtest Overfitting (CSCV PBO)

| Metric | Value |
|---|---|
| Partitions ($S$) / Splits Evaluated | `8` / `70` (exact=True) |
| Probability of Backtest Overfitting (PBO) | `0.3571` |
| Probability of OOS Loss | `0.0000` |
| Median OOS Rank Logit | `0.3365` |
| Mean IS vs OOS Selected Sharpe | `3.1893` -> `2.4602` |
| IS-to-OOS Degradation Slope | `-0.5256` |

## 5. Block Bootstrap Uncertainty Bounds

- **Method:** `stationary` (block_length=`10`, resamples=`1000`, seed=`42`)
- **Annualized Sharpe 95% CI:** `[1.6972, 4.5167]` (SE=`0.7019`)
- **Annualized Mean Return 95% CI:** `[19.2695%, 50.7027%]`
- **Max Drawdown Observed / 95% Upper Bound:** `4.05%` / `7.10%`
- **Empirical Bootstrap p-value ($SR \le 0$):** `0.0000`

## 6. Walk-Forward Out-of-Sample Evaluation

- **Mode:** `expanding` | train=`120` | purge=`5` | test=`40` | folds=`5` | OOS obs=`200`
- **Mean IS Net Sharpe:** `2.8543`
- **Aggregate OOS Net Sharpe:** `3.2638` (Gross: `3.3138`)
- **Aggregate OOS Benchmark Sharpe:** `2.0574`
- **IS-to-OOS Degradation Ratio:** `-0.1435` | **Fold Hit Rate:** `100.0%`

| Fold | Train Window | Test Window | Selected Trial | Train Net SR | Test Net SR | Bench SR |
|---|---|---|---|---|---|---|
| 0 | `2024-01-02` .. `2024-06-17` | `2024-06-25` .. `2024-08-19` | `cherry_seq4` | `3.729` | `0.441` | `0.120` |
| 1 | `2024-01-02` .. `2024-08-12` | `2024-08-20` .. `2024-10-14` | `cherry_seq4` | `2.475` | `2.651` | `6.193` |
| 2 | `2024-01-02` .. `2024-10-07` | `2024-10-15` .. `2024-12-09` | `cherry_seq1` | `2.400` | `3.449` | `-0.357` |
| 3 | `2024-01-02` .. `2024-12-02` | `2024-12-10` .. `2025-02-03` | `cherry_seq4` | `2.511` | `7.326` | `0.758` |
| 4 | `2024-01-02` .. `2025-01-27` | `2025-02-04` .. `2025-03-31` | `cherry_seq4` | `3.156` | `3.086` | `3.191` |

## 7. Parameter Perturbation Stability

- **Center Trial:** `cherry_seq4` (`{'lookback': 20, 'vol_target': 0.12}`, net SR=`3.0471`)
- **Neighbors Evaluated:** `3` | **Positive Fraction:** `100.0%`
- **Median / Min Neighbor Net Sharpe:** `1.4403` / `1.2996`
- **Sharpe Cliff Ratio:** `0.5273` | **Stability Score:** `0.7363`

## 8. Outlier & Tail-Risk Diagnostics

- **Max |Return| / Max Robust |z|:** `0.0250` / `3.32` (`0` extreme obs)
- **Raw vs 1%-Winsorized Annualized Sharpe:** `3.0471` vs `3.0894` (impact=`-1.4%`)

## 9. Registry Integrity & Cherry-Picking Audit

- **Hash-Chain Valid:** `True` | **Audit Passed:** `False`
- **Logged Trials:** `5` (completed=`5`, failed/rejected=`0`)
- **Duplicate Fraction:** `20.00%` (exact=`1`, near=`0`)
- **Grid Coverage:** `5/9` (`55.6%`)

## 10. Review Notes & Warnings

- [Integrity] Detected 4 missing sequence number(s) in ledger (possible omitted/cherry-picked trials): [3, 5, 6, 8]
- [Integrity] Detected 1 exact duplicate trial pair(s).
- [Integrity] Duplicate trial fraction (20.00%) exceeds allowed limit (15.00%).
- [Integrity] Parameter grid coverage is 55.6% (5/9; 4 unlogged configuration(s)), below required 85.0% — potential cherry-picking.

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
