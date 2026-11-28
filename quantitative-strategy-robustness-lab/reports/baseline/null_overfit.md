# Strategy Robustness Evaluation Report — `exp-null-noise-v1`

- **Run ID:** `run-ff104762a0ccc501`
- **Candidate Trial:** `null_f5_s80`
- **Promotion Verdict:** **`REJECT`**
- **Data Provenance:** `synthetic_fixture`
- **Registry Head Hash:** `0fd7eda51c8dbc65...`
- **Policy Version / Seed:** `v1.0.0` / `seed=42`

> **Summary:** REJECTED: Candidate 'null_f5_s80' failed 11 required gate(s): deflated_sharpe_ratio, psr_vs_benchmark, probability_of_backtest_overfitting, pbo_oos_loss_probability, bootstrap_sharpe_ci_lower, walk_forward_oos_net_sharpe, walk_forward_is_to_oos_degradation, walk_forward_oos_vs_benchmark, perturbation_positive_neighbors, perturbation_sharpe_cliff, perturbation_stability_score.

## 1. Promotion Gate Scorecard

| Gate | Category | Result | Observed | Threshold |
|---|---|---|---|---|
| `evidence_completeness` | governance | **PASS** | `complete` | `all 7 diagnostics present` |
| `registry_and_trial_integrity` | governance | **PASS** | `{"registry_valid": true, "duplicate_fraction": 0.0, "grid_coverage": 1.0, "flags": 0}` | `valid hash chain, no sequence gaps, duplicates & grid coverage within policy` |
| `deflated_sharpe_ratio` | multiple_testing | **FAIL** | `0.3633` | `>= 0.95` |
| `psr_vs_benchmark` | multiple_testing | **FAIL** | `0.8794` | `>= 0.9` |
| `probability_of_backtest_overfitting` | multiple_testing | **FAIL** | `0.5143` | `<= 0.25` |
| `pbo_oos_loss_probability` | multiple_testing | **FAIL** | `0.6714` | `<= 0.3` |
| `bootstrap_sharpe_ci_lower` | uncertainty | **FAIL** | `-0.8346` | `>= 0.0` |
| `walk_forward_oos_sample_size` | out_of_sample | **PASS** | `200` | `>= 40` |
| `walk_forward_oos_net_sharpe` | out_of_sample | **FAIL** | `0.1411` | `>= 0.35` |
| `walk_forward_is_to_oos_degradation` | out_of_sample | **FAIL** | `0.9247` | `<= 0.65` |
| `walk_forward_oos_vs_benchmark` | out_of_sample | **FAIL** | `-0.2855` | `>= 0.0` |
| `perturbation_neighbor_coverage` | perturbation | **PASS** | `5` | `>= 1 local neighbor` |
| `perturbation_positive_neighbors` | perturbation | **FAIL** | `0.2` | `>= 0.65` |
| `perturbation_sharpe_cliff` | perturbation | **FAIL** | `1.6842` | `<= 0.5` |
| `perturbation_stability_score` | perturbation | **FAIL** | `0.1` | `>= 0.6` |
| `outlier_dominance_check` | tail_risk | **PASS** | `{"n_extreme": 0, "excess_kurtosis": 0.13, "winsorized_sharpe": 1.123, "impact_pct": -0.03}` | `no extreme outliers with excess kurtosis > 20.0 or >50% winsorized Sharpe drop` |

## 2. Transaction Costs & Net Performance

- **Total Gross Compounded Return:** `26.5459%`
- **Total Net Compounded Return:** `23.0460%`
- **Mean Per-Period Cost Drag:** `0.78 bps`
- **Annualized Cost Drag:** `1.9644%`

## 3. Multiple-Testing Diagnostics (DSR & PSR)

| Metric | Value |
|---|---|
| Observed Net Sharpe (Annualized) | `1.0902` |
| Probabilistic Sharpe Ratio vs Benchmark | `0.8794` |
| Deflated Sharpe Ratio (DSR) | `0.3633` |
| Raw Logged Trials / Effective Trials ($N_{eff}$) | `16` / `15.52` |
| Expected Max Null Sharpe ($SR^*$, Annualized) | `1.4154` |
| Skewness / Excess Kurtosis | `0.035` / `0.129` |
| Autocorrelation Variance Inflation Factor (VIF) | `1.233` |

## 4. Probability of Backtest Overfitting (CSCV PBO)

| Metric | Value |
|---|---|
| Partitions ($S$) / Splits Evaluated | `8` / `70` (exact=True) |
| Probability of Backtest Overfitting (PBO) | `0.5143` |
| Probability of OOS Loss | `0.6714` |
| Median OOS Rank Logit | `-0.1178` |
| Mean IS vs OOS Selected Sharpe | `1.8501` -> `-0.4654` |
| IS-to-OOS Degradation Slope | `-0.3262` |

## 5. Block Bootstrap Uncertainty Bounds

- **Method:** `stationary` (block_length=`10`, resamples=`1000`, seed=`42`)
- **Annualized Sharpe 95% CI:** `[-0.8346, 2.7086]` (SE=`0.9287`)
- **Annualized Mean Return 95% CI:** `[-11.6830%, 38.6996%]`
- **Max Drawdown Observed / 95% Upper Bound:** `13.95%` / `24.44%`
- **Empirical Bootstrap p-value ($SR \le 0$):** `0.1270`

## 6. Walk-Forward Out-of-Sample Evaluation

- **Mode:** `expanding` | train=`120` | purge=`5` | test=`40` | folds=`5` | OOS obs=`200`
- **Mean IS Net Sharpe:** `1.8731`
- **Aggregate OOS Net Sharpe:** `0.1411` (Gross: `0.2732`)
- **Aggregate OOS Benchmark Sharpe:** `0.4266`
- **IS-to-OOS Degradation Ratio:** `0.9247` | **Fold Hit Rate:** `60.0%`

| Fold | Train Window | Test Window | Selected Trial | Train Net SR | Test Net SR | Bench SR |
|---|---|---|---|---|---|---|
| 0 | `2024-01-02` .. `2024-06-17` | `2024-06-25` .. `2024-08-19` | `null_f5_s80` | `2.301` | `-1.186` | `-5.129` |
| 1 | `2024-01-02` .. `2024-08-12` | `2024-08-20` .. `2024-10-14` | `null_f20_s60` | `1.679` | `-0.428` | `0.089` |
| 2 | `2024-01-02` .. `2024-10-07` | `2024-10-15` .. `2024-12-09` | `null_f5_s60` | `1.572` | `0.461` | `5.833` |
| 3 | `2024-01-02` .. `2024-12-02` | `2024-12-10` .. `2025-02-03` | `null_f5_s80` | `1.795` | `2.055` | `1.333` |
| 4 | `2024-01-02` .. `2025-01-27` | `2025-02-04` .. `2025-03-31` | `null_f5_s80` | `2.019` | `0.047` | `-0.079` |

## 7. Parameter Perturbation Stability

- **Center Trial:** `null_f5_s80` (`{'fast': 5, 'slow': 80}`, net SR=`1.0902`)
- **Neighbors Evaluated:** `5` | **Positive Fraction:** `20.0%`
- **Median / Min Neighbor Net Sharpe:** `-0.7459` / `-1.2106`
- **Sharpe Cliff Ratio:** `1.6842` | **Stability Score:** `0.1000`

## 8. Outlier & Tail-Risk Diagnostics

- **Max |Return| / Max Robust |z|:** `0.0295` / `3.41` (`0` extreme obs)
- **Raw vs 1%-Winsorized Annualized Sharpe:** `1.0902` vs `1.1230` (impact=`-3.0%`)

## 9. Registry Integrity & Cherry-Picking Audit

- **Hash-Chain Valid:** `True` | **Audit Passed:** `True`
- **Logged Trials:** `16` (completed=`16`, failed/rejected=`0`)
- **Duplicate Fraction:** `0.00%` (exact=`0`, near=`0`)
- **Grid Coverage:** `16/16` (`100.0%`)

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
