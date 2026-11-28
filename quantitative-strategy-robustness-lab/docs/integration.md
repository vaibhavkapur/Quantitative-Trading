# Integration & Adapter Boundaries

## 1. Repository Survey (`Trading-Quantitative`)

Sibling project directories at the root of `Trading-Quantitative`:

| Sibling Project Directory | Status | Integration Role |
|---|---|---|
| `multi-factor-asset-pricing-engine/` | Standalone sibling package | Upstream producer of factor portfolio return series and parameter grids |
| `cross-asset-factor-investing-engine/` | Standalone sibling package | Upstream producer of multi-asset strategy backtest trials |
| `statistical-arbitrage-research-platform/` | Standalone sibling package | Upstream producer of spread/pairs trading trial ledgers and turnover series |
| `volatility-forecasting-engine/` | Standalone sibling package | Upstream producer of volatility-timed strategy returns |
| `optimal-execution-research-engine/` | Standalone sibling package | Upstream calibration source for slippage / market impact parameters in `CostModel` |

To keep every project in `Trading-Quantitative` independently installable, `quantitative-strategy-robustness-lab` does **not** import sibling code directly. Instead, it defines stable, file- and dataclass-based adapter contracts that any sibling project can target without tight coupling.

## 2. Reusable Adapter Interfaces

Upstream strategy engines can integrate in two independent ways:

1. **Filesystem Experiment Bundle (Zero Python Coupling):**
   Write a directory containing:
   - `manifest.json` (`experiment_id`, `dataset_version`, `frequency`, `provenance`)
   - `parameter_grid.json` (matching `ParameterGridSpec`)
   - `registry.jsonl` (appended via `ImmutableExperimentRegistry.append()` or `robustness-lab` helper)
   - `series/<series_id>.csv` (`timestamp,gross_return[,turnover,short_exposure,benchmark_return]`)
   Then invoke:
   ```bash
   robustness-lab evaluate --experiment-dir <dir> --policy config/default_policy.json --output-dir reports/
   ```

2. **Typed Python Library Interface (`robustness_lab`):**
   Import pure domain functions and dataclasses directly:
   - `ImmutableExperimentRegistry` for append-only trial logging during hyperparameter sweeps (wrapping both successes and caught exceptions as `TrialStatus.FAILED`).
   - `evaluate_experiment(...)` for in-memory evaluation without temporary CSV files.
   - Standalone diagnostics (`compute_dsr`, `compute_pbo_cscv`, `run_block_bootstrap`, `run_walk_forward`, `evaluate_parameter_perturbation`, `audit_trial_integrity`, `evaluate_promotion`).

## 3. Dependency & License Provenance

All runtime and test dependencies are open-source scientific Python packages pinned in `requirements.lock`:

| Package | Pinned Version | License | Role |
|---|---|---|---|
| `numpy` | `2.5.5` | BSD-3-Clause | Vectorized array math & deterministic RNG |
| `scipy` | `1.18.1` | BSD-3-Clause | Normal CDF/PPF, higher moments, rank statistics, linear regression |
| `pandas` | `2.3.3` | BSD-3-Clause | CSV ingestion/serialization & business-day calendars |
| `pytest` (dev) | `8.4.2` | MIT | Automated test suite |

## 4. Explicit Out-of-Scope / Boundary Policy

- Zero dependencies on private employer, proprietary broker, or SAAT codebases.
- No live capital execution, brokerage APIs, market-data credentials, or real customer records.
- Any future promotion from paper/research status to live deployment requires independent security, operational, legal, and risk committee sign-off outside this repository.
