# Asset Pricing Run Summary — `reference_hand_calculated`

- **Run ID:** `mfape-321cf5ec1174d659`
- **Schema version:** `1.0.0`
- **Engine version:** `0.1.0`
- **Baseline model:** `CAPM`
- **Split boundaries:** `{"full_panel_end": "2020-06-30", "full_panel_periods": 6, "full_panel_start": "2020-01-31", "test_end": null, "test_periods": 0, "test_start": null, "train_end": "2020-06-30", "train_periods": 6, "train_start": "2020-01-31", "zero_leakage_verified": true}`

## Model Comparison vs Baseline

| Model | Factors | TS Mean R² | TS Mean |α| | GRS p-val | FM CS R² | FM Pricing RMSE | OOS CS R² | OOS CS RMSE | OOS TS R² (α=0) | Δ OOS CS RMSE vs Baseline |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CAPM | MKT_RF | 0.8533 | 0.00250 | N/A | 0.6923 | 0.00224 | N/A | N/A | N/A | N/A |
| TWO_FACTOR | MKT_RF, VAL | 1.0000 | 0.00000 | N/A | 1.0000 | 0.00000 | N/A | N/A | N/A | N/A |

## Fama-MacBeth Cross-Sectional Factor Premia (Training Window)

### CAPM (2020-01-31 to 2020-06-30, T=6, N=4)

| Term | Est (period) | Ann. Est | FM SE | FM t-stat | Shanken SE | Shanken t-stat | HAC SE | HAC t-stat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gamma_0 | -0.00550 | -0.0660 | 0.00696 | -0.79 | 0.01015 | -0.54 | 0.00635 | -0.87 |
| MKT_RF | 0.01500 | 0.1800 | 0.00966 | 1.55 | 0.01177 | 1.27 | 0.00882 | 1.70 |

### TWO_FACTOR (2020-01-31 to 2020-06-30, T=6, N=4)

| Term | Est (period) | Ann. Est | FM SE | FM t-stat | Shanken SE | Shanken t-stat | HAC SE | HAC t-stat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gamma_0 | 0.00000 | 0.0000 | 0.00000 | 0.00 | 0.00000 | 0.00 | 0.00000 | 0.00 |
| MKT_RF | 0.01000 | 0.1200 | 0.00730 | 1.37 | 0.00730 | 1.37 | 0.00667 | 1.50 |
| VAL | 0.00500 | 0.0600 | 0.00632 | 0.79 | 0.00632 | 0.79 | 0.00577 | 0.87 |

