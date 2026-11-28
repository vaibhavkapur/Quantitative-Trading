# Strategy Robustness Evaluation Report — `exp-extreme-outliers`

- **Run ID:** `run-a3ea7501c2ee6945`
- **Candidate Trial:** `outlier_t3`
- **Promotion Verdict:** **`REJECT`**
- **Data Provenance:** `synthetic_fixture`
- **Registry Head Hash:** `68bd275d97c9b918...`
- **Policy Version / Seed:** `v1.0.0` / `seed=42`

> **Summary:** REJECTED: Candidate 'outlier_t3' failed 8 required gate(s): deflated_sharpe_ratio, probability_of_backtest_overfitting, pbo_oos_loss_probability, bootstrap_sharpe_ci_lower, perturbation_positive_neighbors, perturbation_sharpe_cliff, perturbation_stability_score, outlier_dominance_check.

## 1. Promotion Gate Scorecard

| Gate | Category | Result | Observed | Threshold |
|---|---|---|---|---|
| `evidence_completeness` | governance | **PASS** | `complete` | `all 7 diagnostics present` |
| `registry_and_trial_integrity` | governance | **PASS** | `{"registry_valid": true, "duplicate_fraction": 0.0, "grid_coverage": null, "flags": 0}` | `valid hash chain, no sequence gaps, duplicates & grid coverage within policy` |
| `deflated_sharpe_ratio` | multiple_testing | **FAIL** | `0.7281` | `>= 0.95` |
| `psr_vs_benchmark` | multiple_testing | **PASS** | `0.9799` | `>= 0.9` |
| `probability_of_backtest_overfitting` | multiple_testing | **FAIL** | `0.3286` | `<= 0.25` |
| `pbo_oos_loss_probability` | multiple_testing | **FAIL** | `0.3857` | `<= 0.3` |
| `bootstrap_sharpe_ci_lower` | uncertainty | **FAIL** | `-1.0742` | `>= 0.0` |
| `walk_forward_oos_sample_size` | out_of_sample | **PASS** | `200` | `>= 40` |
| `walk_forward_oos_net_sharpe` | out_of_sample | **PASS** | `1.344` | `>= 0.35` |
| `walk_forward_is_to_oos_degradation` | out_of_sample | **PASS** | `-0.3622` | `<= 0.65` |
| `perturbation_neighbor_coverage` | perturbation | **PASS** | `4` | `>= 1 local neighbor` |
| `perturbation_positive_neighbors` | perturbation | **FAIL** | `0.0` | `>= 0.65` |
| `perturbation_sharpe_cliff` | perturbation | **FAIL** | `1.3344` | `<= 0.5` |
| `perturbation_stability_score` | perturbation | **FAIL** | `0.0` | `>= 0.6` |
| `outlier_dominance_check` | tail_risk | **FAIL** | `{"n_extreme": 2, "excess_kurtosis": 155.86, "winsorized_sharpe": -0.192, "impact_pct": 1.185}` | `no extreme outliers with excess kurtosis > 20.0 or >50% winsorized Sharpe drop` |

## 2. Transaction Costs & Net Performance

- **Total Gross Compounded Return:** `46.2679%`
- **Total Net Compounded Return:** `45.7428%`
- **Mean Per-Period Cost Drag:** `0.10 bps`
- **Annualized Cost Drag:** `0.2520%`

## 3. Multiple-Testing Diagnostics (DSR & PSR)

| Metric | Value |
|---|---|
| Observed Net Sharpe (Annualized) | `1.0402` |
| Probabilistic Sharpe Ratio vs Benchmark | `0.9799` |
| Deflated Sharpe Ratio (DSR) | `0.7281` |
| Raw Logged Trials / Effective Trials ($N_{eff}$) | `6` / `5.91` |
| Expected Max Null Sharpe ($SR^*$, Annualized) | `0.7326` |
| Skewness / Excess Kurtosis | `12.053` / `155.862` |
| Autocorrelation Variance Inflation Factor (VIF) | `0.964` |

## 4. Probability of Backtest Overfitting (CSCV PBO)

| Metric | Value |
|---|---|
| Partitions ($S$) / Splits Evaluated | `8` / `70` (exact=True) |
| Probability of Backtest Overfitting (PBO) | `0.3286` |
| Probability of OOS Loss | `0.3857` |
| Median OOS Rank Logit | `0.9163` |
| Mean IS vs OOS Selected Sharpe | `1.0233` -> `0.1682` |
| IS-to-OOS Degradation Slope | `-0.0522` |

## 5. Block Bootstrap Uncertainty Bounds

- **Method:** `stationary` (block_length=`10`, resamples=`1000`, seed=`42`)
- **Annualized Sharpe 95% CI:** `[-1.0742, 1.8221]` (SE=`0.7042`)
- **Annualized Mean Return 95% CI:** `[-7.9591%, 77.4020%]`
- **Max Drawdown Observed / 95% Upper Bound:** `7.74%` / `16.43%`
- **Empirical Bootstrap p-value ($SR \le 0$):** `0.1050`

## 6. Walk-Forward Out-of-Sample Evaluation

- **Mode:** `expanding` | train=`120` | purge=`5` | test=`40` | folds=`5` | OOS obs=`200`
- **Mean IS Net Sharpe:** `0.9866`
- **Aggregate OOS Net Sharpe:** `1.3440` (Gross: `1.3527`)
- **Aggregate OOS Benchmark Sharpe:** `N/A`
- **IS-to-OOS Degradation Ratio:** `-0.3622` | **Fold Hit Rate:** `60.0%`

| Fold | Train Window | Test Window | Selected Trial | Train Net SR | Test Net SR | Bench SR |
|---|---|---|---|---|---|---|
| 0 | `2024-01-02` .. `2024-06-17` | `2024-06-25` .. `2024-08-19` | `outlier_t3` | `0.944` | `2.219` | `N/A` |
| 1 | `2024-01-02` .. `2024-08-12` | `2024-08-20` .. `2024-10-14` | `outlier_t3` | `0.898` | `-2.964` | `N/A` |
| 2 | `2024-01-02` .. `2024-10-07` | `2024-10-15` .. `2024-12-09` | `outlier_t3` | `0.792` | `3.046` | `N/A` |
| 3 | `2024-01-02` .. `2024-12-02` | `2024-12-10` .. `2025-02-03` | `outlier_t3` | `0.891` | `3.431` | `N/A` |
| 4 | `2024-01-02` .. `2025-01-27` | `2025-02-04` .. `2025-03-31` | `outlier_t3` | `1.408` | `-3.361` | `N/A` |

## 7. Parameter Perturbation Stability

- **Center Trial:** `outlier_t3` (`{'threshold': 1.3}`, net SR=`1.0402`)
- **Neighbors Evaluated:** `4` | **Positive Fraction:** `0.0%`
- **Median / Min Neighbor Net Sharpe:** `-0.3478` / `-0.3662`
- **Sharpe Cliff Ratio:** `1.3344` | **Stability Score:** `0.0000`

## 8. Outlier & Tail-Risk Diagnostics

- **Max |Return| / Max Robust |z|:** `0.2500` / `48.36` (`2` extreme obs)
- **Raw vs 1%-Winsorized Annualized Sharpe:** `1.0402` vs `-0.1925` (impact=`118.5%`)

## 9. Registry Integrity & Cherry-Picking Audit

- **Hash-Chain Valid:** `True` | **Audit Passed:** `True`
- **Logged Trials:** `6` (completed=`6`, failed/rejected=`0`)
- **Duplicate Fraction:** `0.00%` (exact=`0`, near=`0`)
- **Grid Coverage:** N/A (no grid manifest supplied)

## 10. Review Notes & Warnings

- [DSR] Heavy tails detected (excess kurtosis=155.86); asymptotic normal PSR/DSR approximation has higher uncertainty.
- [Outliers] Detected 2 extreme outlier observation(s) with robust |z| > 8.0 (max |z|=48.36, max |r|=0.2500).
- [Outliers] Excess kurtosis (155.86) exceeds warning threshold (20.0).
- [Outliers] 1% winsorization changes annualized Sharpe from 1.04 to -0.19 (118.5% sensitivity).

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
