# Integration and Dependencies — Multi-Factor Asset Pricing Engine

## 1. Repository Context (`Trading-Quantitative`)

This project resides at `multi-factor-asset-pricing-engine/` at the root of the `Trading-Quantitative` repository alongside independent sibling projects, including:

- `cross-asset-factor-investing-engine/` (factor construction and cross-asset portfolio allocation)
- `multi-factor-asset-pricing-engine/` (**this project** — factor pricing estimation and cross-sectional evaluation)
- `optimal-execution-research-engine/` (execution scheduling and impact simulation)
- `quantitative-strategy-robustness-lab/` (multiple-testing and strategy robustness evaluation)
- `statistical-arbitrage-research-platform/` (pairs and basket mean reversion)
- `volatility-forecasting-engine/` (realized-volatility forecasting)

Each project in `Trading-Quantitative` is a standalone, independently installable package with zero runtime cross-imports; this project (`mfape`) exposes clean programmatic and file-based interfaces that sibling projects can consume via lightweight adapters rather than copying code.

## 2. Reusable Adapter Boundaries

Downstream or upstream projects in `Trading-Quantitative` (for example, `cross-asset-factor-investing-engine` or `quantitative-strategy-robustness-lab`) can integrate at three stable boundaries:

1. **In-Memory DataFrame Adapter (`AssetPricingEngine.run_on_frames`):**
   - Accepts four Polars `DataFrame`s (`asset_returns_df`, `factor_returns_df`, `risk_free_df`, `memberships_df`) plus a typed `EngineConfig` object.
   - Avoids disk serialization when chaining factor construction directly into pricing-error evaluation.
2. **Pure Econometric Domain Functions (`mfape.time_series`, `mfape.fama_macbeth`, `mfape.evaluation`):**
   - Operate strictly on `AlignedPanel` (NumPy arrays) and `FactorModelSpec`.
   - Allow external robustness harnesses (e.g., walk-forward or sub-period stress labs) to invoke `fit_time_series_model`, `estimate_fama_macbeth`, and `evaluate_out_of_sample` directly.
3. **CLI / File Artifact Boundary (`mfape run`, `mfape replay`, `run_result.json`):**
   - Language-agnostic integration via versioned JSON (`schema_version: "1.0.0"`) and CSV (`model_comparison.csv`).

## 3. Dependency Manifest and Licenses

Runtime and development dependencies are pinned in `requirements-lock.txt` and declared in `pyproject.toml`:

| Package | Version (Locked) | Purpose | License |
|---|---|---|---|
| `numpy` | `2.0.2` | Dense matrix linear algebra, SVD rank checks, array operations | BSD-3-Clause |
| `polars` | `1.36.1` | High-speed tabular loading, schema validation, point-in-time joins | MIT |
| `statsmodels` | `0.14.6` | Reference OLS, HC3, and Newey-West HAC covariance estimators | BSD-3-Clause |
| `scipy` | `1.13.1` | F-distribution (GRS test) and Student-t / Normal p-values | BSD-3-Clause |
| `pandas` | `2.3.3` | Transitive dependency of `statsmodels` | BSD-3-Clause |
| `patsy` | `1.0.3` | Transitive dependency of `statsmodels` | BSD-2-Clause |
| `pytest` (dev) | `8.4.2` | Automated unit, integration, and boundary test runner | MIT |

## 4. Compatibility & Security Boundaries

- **Python Compatibility:** Verified on Python 3.9.6 (`macOS arm64`); written with `from __future__ import annotations` for forward compatibility with Python 3.9–3.12+.
- **No Proprietary Dependencies:** Zero imports from private employer repositories, SAAT code, live broker APIs, or production databases.
- **Offline Execution:** After dependency installation, all fixture generation, engine runs, replays, and test suites execute 100% offline with zero network calls.
