# Scenario Validation Evidence

- **Generated UTC:** `2026-11-27T09:50:48.358315Z`
- **All scenarios passed:** `True`

| Scenario | Passed | Key Evidence |
|---|---|---|
| `non_psd_covariance` | PASS | `reject_mode_error_code=NON_PSD_COVARIANCE`, `original_min_eigenvalue=-0.014051785054163267`, `repair_mode_min_eigenvalue=1e-07`, `repair_frobenius_delta=0.014051885054163266`, `post_repair_feasible=True` |
| `zero_volatility` | PASS | `reject_mode_error_code=ZERO_VOLATILITY`, `excluded_reasons={'strat_vol_carry': 'zero_volatility:0.000e+00'}`, `inverse_vol_zero_vol_weight=0.0`, `constrained_feasible=True` |
| `insufficient_history` | PASS | `reject_mode_error_code=INSUFFICIENT_HISTORY`, `excluded_reasons={'strat_eq_momentum': 'insufficient_history:5<20'}`, `observation_counts={'strat_eq_statarb_us': 60, 'strat_eq_momentum': 5, 'strat_macro_cta': 60, 'strat_vol_carry': 60, 'strat_fi_rv': 60}`, `constrained_feasible=True` |
| `infeasible_caps` | PASS | `default_fallback_status=FALLBACK_NO_CHANGE`, `cash_only_fallback_status=FALLBACK_CASH`, `infeasibility_reasons=['infeasible_caps:sum_upper=0.250000<min_invested=0.500000', 'infeasible_group_caps_total:max_deployable=0.250000<min_invested=0.500000']` |
| `stale_positions` | PASS | `error_code=STALE_DATA`, `position_age_hours=72.0`, `max_allowed_hours=24.0` |
| `turnover_limit` | PASS | `achieved_one_way_turnover=0.05`, `limit=0.05`, `feasible=True` |
| `correlated_drawdowns` | PASS | `correlation_stress_active=True`, `cash_weight=0.5`, `portfolio_vol_annual=0.07367422617943693`, `stressed_vol_annual=0.07731269800532009`, `feasible=True` |
