# Strategy Robustness Evaluation Report — `exp-insufficient-sample`

- **Run ID:** `run-47f222cfa9b45dec`
- **Candidate Trial:** `short_t1`
- **Promotion Verdict:** **`REJECT`**
- **Data Provenance:** `synthetic_fixture`
- **Registry Head Hash:** `8ad05e6864665e00...`
- **Policy Version / Seed:** `v1.0.0` / `seed=42`

> **Summary:** REJECTED: Incomplete evidence (dsr_diagnostics, pbo_diagnostics, block_bootstrap, walk_forward_oos). Promotion requires full diagnostic suite.

## 1. Promotion Gate Scorecard

| Gate | Category | Result | Observed | Threshold |
|---|---|---|---|---|
| `evidence_completeness` | governance | **FAIL** | `missing: dsr_diagnostics, pbo_diagnostics, block_bootstrap, walk_forward_oos` | `all 7 diagnostics present` |
| `registry_and_trial_integrity` | governance | **PASS** | `{"registry_valid": true, "duplicate_fraction": 0.0, "grid_coverage": null, "flags": 0}` | `valid hash chain, no sequence gaps, duplicates & grid coverage within policy` |
| `perturbation_neighbor_coverage` | perturbation | **FAIL** | `0` | `>= 1 local neighbor` |
| `perturbation_positive_neighbors` | perturbation | **FAIL** | `0.0` | `>= 0.65` |
| `perturbation_sharpe_cliff` | perturbation | **FAIL** | `1.0` | `<= 0.5` |
| `perturbation_stability_score` | perturbation | **FAIL** | `0.0` | `>= 0.6` |
| `outlier_dominance_check` | tail_risk | **PASS** | `{"n_extreme": 0, "excess_kurtosis": 2.07, "winsorized_sharpe": 6.723, "impact_pct": -0.025}` | `no extreme outliers with excess kurtosis > 20.0 or >50% winsorized Sharpe drop` |

## 2. Transaction Costs & Net Performance

- **Total Gross Compounded Return:** `5.9928%`
- **Total Net Compounded Return:** `5.9928%`
- **Mean Per-Period Cost Drag:** `0.00 bps`
- **Annualized Cost Drag:** `0.0000%`

## 7. Parameter Perturbation Stability

- **Center Trial:** `short_t1` (`{'k': 1}`, net SR=`6.5604`)
- **Neighbors Evaluated:** `0` | **Positive Fraction:** `0.0%`
- **Median / Min Neighbor Net Sharpe:** `0.0000` / `0.0000`
- **Sharpe Cliff Ratio:** `1.0000` | **Stability Score:** `0.0000`

## 8. Outlier & Tail-Risk Diagnostics

- **Max |Return| / Max Robust |z|:** `0.0174` / `3.67` (`0` extreme obs)
- **Raw vs 1%-Winsorized Annualized Sharpe:** `6.5604` vs `6.7231` (impact=`-2.5%`)

## 9. Registry Integrity & Cherry-Picking Audit

- **Hash-Chain Valid:** `True` | **Audit Passed:** `True`
- **Logged Trials:** `4` (completed=`4`, failed/rejected=`0`)
- **Duplicate Fraction:** `0.00%` (exact=`0`, near=`0`)
- **Grid Coverage:** N/A (no grid manifest supplied)

## 10. Review Notes & Warnings

- [Perturbation] No neighboring parameter configurations found within relative_step=0.20 or adjacent grid steps; local robustness cannot be confirmed.

## 11. Methodology Assumptions & Limitations

> Statistical diagnostics (DSR, CSCV PBO, block bootstrap, walk-forward, perturbation) evaluate historical consistency under stated stationarity and dependence assumptions. They do NOT prove future returns. Offline research prototype only.

### Key Recorded Assumptions


### Runtime Environment

- Python `3.12.15` | NumPy `2.5.5` | SciPy `1.18.1` | pandas `2.3.3`
- Policy SHA-256: `43727f3390f894e0a59231cfed79907f5c0c63f3f2d2e4ced816c5cb510eca75`
