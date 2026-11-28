# Optimal Execution Research Engine

Offline quantitative research prototype for comparing equity execution schedules (**TWAP**, **VWAP**, **Participation / POV**, and **Almgren-Chriss analytic & numeric**) under explicit spread, permanent/temporary market impact, slice capacity, zero-liquidity, price-shock, and timing-risk assumptions.

> **Research Prototype Disclaimer:** Execution models are mathematical assumptions. All numbers and reports produced by this project are **measured observations on synthetic offline fixtures** — do not interpret simulated fills as real trading results. Live capital, real customer records, broker connectivity, and production credentials are strictly excluded.

## Key Capabilities

- **Execution Schedules (`src/optimal_execution/schedules.py`):**
  - `twap`: Uniform time-weighted slicing over `[0, N)`.
  - `vwap`: Volume-weighted slicing proportional to expected market volume (with explicit zero-liquidity fallback).
  - `participation` (POV): Target market participation rate schedule with deterministic residual accounting.
  - `almgren_chriss_analytic`: Exact discrete-time Almgren-Chriss (2001) `sinh` trajectory using cancellation- and overflow-safe `asinh`/`expm1` evaluation.
  - `almgren_chriss_numeric`: Independent SciPy `SLSQP` optimizer with analytical gradients supporting nonlinear temporary impact ($\alpha \ne 1$) and optional hard per-slice capacity bounds.
- **Fill Simulator (`src/optimal_execution/simulator.py`):**
  - Enforces hard per-slice capacity limits (`filled_k <= max_participation_rate * volume_k`) and exact order conservation (`filled + residual == parent`).
  - Supports zero-liquidity halt slices, `CapacityPolicy.NONE` vs `CapacityPolicy.SPILLOVER`, exogenous `PriceShock` events, and fixed-seed vectorized Monte Carlo simulation.
- **Implementation Shortfall & Sensitivity (`analytics.py`, `sensitivity.py`):**
  - Exact Perold (1988) decomposition: `Total IS = Spread + Temporary Impact + Permanent Impact + Timing Drift + Price Shocks + Opportunity Cost`.
  - Closed-form expected IS and timing risk ($\sqrt{\operatorname{Var}[IS]}$) cross-checked against Monte Carlo empirical mean, standard deviation, p05/p50/p95, VaR 95%, and CVaR 95%.
  - Sensitivity sweeps across liquidity, spread, volatility, impact, and urgency ($\lambda$) regimes exported via Polars DataFrames.

## Repository Layout

```text
optimal-execution-research-engine/
├── plan.md                          # Original implementation plan & acceptance specification
├── README.md                        # Overview, quickstart, measured baseline & limitations
├── pyproject.toml / setup.py        # Package metadata & CLI entry point
├── requirements.lock                # Pinned Python dependencies
├── config/
│   ├── default_scenario.json        # Baseline 13-slice synthetic benchmark scenario
│   └── sensitivity_grid.json        # 4-regime x 5-lambda sensitivity sweep config
├── data/fixtures/                   # Valid, hand-calculated oracle, boundary & invalid fixtures
├── src/optimal_execution/           # Pure domain core + I/O & CLI boundary
├── scripts/                         # Reproducible demo, sensitivity & acceptance scripts
├── tests/                           # 52 unit, integration, boundary & reproducibility tests
├── docs/                            # Architecture, contracts, integration, runbook & ADR-0001
└── reports/                         # Versioned JSON/CSV outputs & reports/acceptance.md
```

CI workflow: [`../.github/workflows/optimal-execution-research-engine.yml`](../.github/workflows/optimal-execution-research-engine.yml)

## Quickstart & Setup

From `optimal-execution-research-engine/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
python setup.py develop --no-deps   # or: pip install --no-build-isolation --no-deps -e .
```

## Demo, Validation & Replay Commands

```bash
# 1. Run all 52 automated tests (domain, integration, analytic-vs-numeric, CLI)
.venv/bin/pytest -v

# 2. Run the reproducible baseline multi-strategy demo -> writes reports/baseline_demo_*
.venv/bin/python scripts/run_demo.py

# 3. Run liquidity & urgency regime sensitivity analysis -> writes reports/sensitivity_*
.venv/bin/python scripts/run_sensitivity.py

# 4. Verify analytic vs numeric Almgren-Chriss agreement across urgency regimes
.venv/bin/optimal-execution compare \
  --scenario config/default_scenario.json \
  --lambdas 0.0,1e-7,1e-6,1e-5,1e-4 \
  --tolerance 1e-4

# 5. Replay a previous run bit-for-bit from its saved manifest
.venv/bin/optimal-execution replay --manifest reports/baseline_replay_manifest.json

# 6. Regenerate full acceptance evidence (reports/acceptance.md + raw JSON/CSV reports)
.venv/bin/python scripts/generate_acceptance_evidence.py
```

## Example Measured Output (Synthetic Baseline Fixture)

Measured output from `config/default_scenario.json` (`SYN_LGCAP`, buy `100,000` shares @ `$150.00`, 13 x 30-min slices, 1.8% daily volatility, 4.0 bps spread, 25% max participation cap, spillover enabled, `seed=20260105`, 2,000 MC paths, Run ID `run-19f653520aa0`):

| Strategy | Completion | Det. IS (bps) | MC Mean IS (bps) | MC Std (bps) | MC p95 (bps) | MC CVaR 95% (bps) |
|---|---:|---:|---:|---:|---:|---:|
| `twap` | 100.00% | 5.436 | 3.442 | 96.040 | 159.305 | 197.765 |
| `vwap` | 100.00% | 5.965 | 4.372 | 94.107 | 155.730 | 195.961 |
| `participation` (12%) | 100.00% | 6.885 | 5.190 | 66.121 | 111.363 | 139.841 |
| `almgren_chriss_analytic` ($\lambda=10^{-6}$) | 100.00% | 8.400 | 7.384 | 50.913 | 88.570 | 111.288 |
| `almgren_chriss_numeric` ($\lambda=10^{-6}$) | 100.00% | 8.400 | 7.384 | 50.913 | 88.570 | 111.288 |

- **Analytic vs. Numeric Almgren-Chriss Verification:** Maximum relative $L_2$ trade difference across $\lambda \in \{0, 10^{-7}, 10^{-6}, 10^{-5}, 10^{-4}\}$ is **`1.393e-07`** (engineering tolerance target: `1e-4`).
- **Hand-Calculated Reference Oracle (`data/fixtures/reference_deterministic.json`):** Measured TWAP deterministic IS `$1,075.00` (`10.750 bps`) and timing risk `$9,354.14` (`93.541 bps`) match hand-calculated values in `data/fixtures/reference_expected.json` within `< 1e-6` USD.

## Documentation & Acceptance Evidence

- **Acceptance Evidence:** [reports/acceptance.md](reports/acceptance.md) (maps every criterion to measured test/benchmark outputs)
- **Raw Machine-Readable Reports:** [reports/baseline_demo_results.json](reports/baseline_demo_results.json), [reports/baseline_demo_summary.csv](reports/baseline_demo_summary.csv), [reports/sensitivity_grid.csv](reports/sensitivity_grid.csv), [reports/analytic_vs_numeric_report.json](reports/analytic_vs_numeric_report.json)
- **Architecture & Math:** [docs/architecture.md](docs/architecture.md) and [docs/adr/0001-domain-separation-and-almgren-chriss-formulation.md](docs/adr/0001-domain-separation-and-almgren-chriss-formulation.md)
- **Data Contracts & Fixtures:** [docs/contracts.md](docs/contracts.md)
- **Integration & Dependencies:** [docs/integration.md](docs/integration.md)
- **Operator Runbook:** [docs/runbook.md](docs/runbook.md)

## Modeling Assumptions & Known Limitations

1. **Arithmetic Price Diffusion:** Mid-price increments follow arithmetic Brownian motion ($\sigma_\$ = S_0 \sigma_{\text{day}}$) to match Almgren-Chriss (2001) exact closed-form variance. Suitable for intraday/single-day horizons; not calibrated for multi-month geometric compounding.
2. **Deterministic Volume Curve per Regime:** Intraday slice volume is deterministic within each scenario and stress-tested via multiplicative liquidity regimes (`0.35x` to `1.8x`). Stochastic intraday volume uncertainty is not modeled.
3. **Static Open-Loop Scheduling:** Schedules are computed upfront (with optional capacity spillover); closed-loop dynamic re-optimization conditioned on realized intraday price paths or short-term alpha signals is deferred.
4. **No L2/L3 Microstructure Simulation:** Spread and temporary impact are parametric functions of trading rate rather than limit order book queue-position simulations.
5. **Offline Research Only:** No live market data, order management system, FIX gateway, or real-capital execution paths exist.
