# Asset Pricing Run Summary — `synthetic_ff3_demo`

- **Run ID:** `mfape-f7a36239df1960a9`
- **Schema version:** `1.0.0`
- **Engine version:** `0.1.0`
- **Baseline model:** `CAPM`
- **Split boundaries:** `{"full_panel_end": "2024-12-31", "full_panel_periods": 180, "full_panel_start": "2010-01-31", "test_end": "2024-12-31", "test_periods": 60, "test_start": "2020-01-31", "train_end": "2019-12-31", "train_periods": 120, "train_start": "2010-01-31", "zero_leakage_verified": true}`

## Model Comparison vs Baseline

| Model | Factors | TS Mean R² | TS Mean |α| | GRS p-val | FM CS R² | FM Pricing RMSE | OOS CS R² | OOS CS RMSE | OOS TS R² (α=0) | Δ OOS CS RMSE vs Baseline |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CAPM | MKT_RF | 0.8431 | 0.00443 | 0.0582 | 0.1820 | 0.00505 | -0.8682 | 0.00693 | 0.8742 | 0.00000 |
| FF3 | MKT_RF, SMB, HML | 0.9575 | 0.00055 | 0.9557 | 0.9881 | 0.00061 | -0.4598 | 0.00613 | 0.9712 | -0.00080 |
| FF4_MOM | MKT_RF, SMB, HML, MOM | 0.9579 | 0.00055 | 0.9583 | 0.9881 | 0.00061 | -0.4598 | 0.00613 | 0.9709 | -0.00080 |

## Fama-MacBeth Cross-Sectional Factor Premia (Training Window)

### CAPM (2010-01-31 to 2019-12-31, T=120, N=25)

| Term | Est (period) | Ann. Est | FM SE | FM t-stat | Shanken SE | Shanken t-stat | HAC SE | HAC t-stat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gamma_0 | -0.00111 | -0.0133 | 0.00087 | -1.27 | 0.00090 | -1.22 | 0.00095 | -1.17 |
| MKT_RF | 0.01029 | 0.1235 | 0.00392 | 2.62 | 0.00393 | 2.62 | 0.00414 | 2.48 |

### FF3 (2010-01-31 to 2019-12-31, T=120, N=25)

| Term | Est (period) | Ann. Est | FM SE | FM t-stat | Shanken SE | Shanken t-stat | HAC SE | HAC t-stat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gamma_0 | 0.00009 | 0.0011 | 0.00070 | 0.13 | 0.00081 | 0.12 | 0.00074 | 0.13 |
| MKT_RF | 0.00733 | 0.0880 | 0.00382 | 1.92 | 0.00385 | 1.91 | 0.00407 | 1.80 |
| SMB | 0.00460 | 0.0552 | 0.00151 | 3.06 | 0.00151 | 3.04 | 0.00141 | 3.26 |
| HML | 0.00607 | 0.0728 | 0.00134 | 4.51 | 0.00135 | 4.49 | 0.00127 | 4.77 |

### FF4_MOM (2010-01-31 to 2019-12-31, T=120, N=25)

| Term | Est (period) | Ann. Est | FM SE | FM t-stat | Shanken SE | Shanken t-stat | HAC SE | HAC t-stat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gamma_0 | 0.00010 | 0.0012 | 0.00070 | 0.14 | 0.00081 | 0.12 | 0.00074 | 0.13 |
| MKT_RF | 0.00733 | 0.0880 | 0.00382 | 1.92 | 0.00385 | 1.91 | 0.00408 | 1.80 |
| SMB | 0.00460 | 0.0552 | 0.00151 | 3.06 | 0.00151 | 3.04 | 0.00141 | 3.27 |
| HML | 0.00607 | 0.0728 | 0.00135 | 4.51 | 0.00135 | 4.48 | 0.00127 | 4.78 |
| MOM | -0.00265 | -0.0318 | 0.00429 | -0.62 | 0.00481 | -0.55 | 0.00429 | -0.62 |

