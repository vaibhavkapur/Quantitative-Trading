# Integration Notes — Volatility Forecasting Engine

## 1. Repository Context (`Trading-Quantitative`)

This project resides at `volatility-forecasting-engine/` at the root of the `Trading-Quantitative` repository alongside sibling research projects, including:

- `cross-asset-factor-investing-engine/`
- `multi-factor-asset-pricing-engine/`
- `optimal-execution-research-engine/`
- `quantitative-strategy-robustness-lab/`
- `statistical-arbitrage-research-platform/`

To keep each research prototype independently installable and avoid fragile cross-project internal imports, `volatility-forecasting-engine` has **zero runtime imports** from sibling project folders. Instead, it exposes clean, versioned data contracts (`docs/contracts.md`) and a typed Python library interface (`volatility_forecasting`) that sibling engines can consume via thin adapters.

## 2. Reusable Adapter Boundaries

### 2.1 Upstream Consumers (e.g., Factor Scaling, Optimal Execution, Stat-Arb)

Downstream projects can consume volatility forecasts in two supported ways:

1. **File/Artifact Adapter (Recommended for decoupled pipelines):**
   Read `reports/forecast_intervals.csv` (keyed by `asset_id`, `timestamp`, `horizon_days`, `model`) containing `pred_var`, `pred_vol`, `regime`, and conformal bounds `lower_vol_*` / `upper_vol_*`.
   - `cross-asset-factor-investing-engine` can use `pred_vol` for ex-ante volatility scaling.
   - `optimal-execution-research-engine` can pass `pred_vol` as the timing-risk $\sigma$ parameter in Almgren-Chriss scheduling.
   - `quantitative-strategy-robustness-lab` can ingest `splits.json` and `forecast_intervals.csv` to stress-test strategy sensitivity across regimes.

2. **In-Memory Library Adapter:**
   ```python
   from volatility_forecasting import (
       ModelSpec,
       build_causal_dataset,
       create_model,
       evaluate_models_chronologically,
   )
   ```
   Callers pass canonical Polars DataFrames conforming to `docs/contracts.md` and receive typed Polars outputs without touching disk.

### 2.2 Optional `arch` Library Adapter

`GARCH11Model` supports `params={"backend": "arch"}` as an optional adapter around Kevin Sheppard's `arch` package (`pip install -e '.[arch]'`).
- **Default (`backend="scipy"`):** Runs deterministically with only core dependencies (`numpy`, `scipy`, `polars`, `scikit-learn`, `pyyaml`).
- **Optional (`backend="arch"`):** Delegates GARCH(1,1) estimation to `arch.arch_model(..., mean="Zero", vol="GARCH", p=1, q=1)`. If `arch` is not installed, the adapter fails fast with `UnsupportedOptionError` and explicit installation instructions rather than falling back silently.

## 3. Dependency Licenses & Provenance

All core runtime dependencies are permissive open-source packages (BSD-3-Clause / MIT):
- `numpy` (BSD-3-Clause)
- `scipy` (BSD-3-Clause)
- `scikit-learn` (BSD-3-Clause)
- `polars` (MIT)
- `pyyaml` (MIT)
- `pytest` (MIT, dev only)
- `arch` (NCSA / University of Illinois open source, optional)

All fixtures in `data/fixtures/` are synthetically generated from deterministic mathematical formulas in `src/volatility_forecasting/fixtures.py`. No real market vendor feeds, customer records, credentials, or proprietary employer/SAAT code exist anywhere in this project.

## 4. Production Out-of-Scope Boundaries

Live market data feeds, real-time order routing, options pricing/hedging execution, and production secret management are explicitly disabled and outside scope for this research prototype.
