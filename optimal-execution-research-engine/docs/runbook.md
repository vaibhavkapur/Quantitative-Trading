# Operator & Developer Runbook

## 1. Clean Local Setup

From `optimal-execution-research-engine/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
python setup.py develop --no-deps   # or: pip install --no-build-isolation --no-deps -e .
```

## 2. Standard Commands

### Run the Full Validation Test Suite
```bash
.venv/bin/pytest -v
```

### Run the Reproducible Baseline Demo
```bash
.venv/bin/python scripts/run_demo.py
```
Outputs written to `reports/`:
- `reports/baseline_demo_results.json` (versioned results + runtime metadata + efficient frontier)
- `reports/baseline_demo_summary.csv` (Polars summary table)
- `reports/baseline_replay_manifest.json` (self-contained replay manifest)

### Run Liquidity & Urgency Sensitivity Analysis
```bash
.venv/bin/python scripts/run_sensitivity.py
```
Outputs written to `reports/`:
- `reports/sensitivity_results.json`
- `reports/sensitivity_grid.csv`

### Compare Analytic vs Numeric Almgren-Chriss Schedules
```bash
.venv/bin/python -m optimal_execution.cli compare \
  --scenario config/default_scenario.json \
  --lambdas 0.0,1e-7,1e-6,1e-5,1e-4 \
  --tolerance 1e-4 \
  --output reports/analytic_vs_numeric_report.json
```

### Regenerate All Acceptance Evidence
```bash
.venv/bin/python scripts/generate_acceptance_evidence.py
```

## 3. Replay & Rerun Workflow

Every `demo` run persists a `baseline_replay_manifest.json` containing the exact scenario payload, schedule requests, seed, and `config_sha256`. To reproduce a run bit-for-bit:

```bash
.venv/bin/python -m optimal_execution.cli replay --manifest reports/baseline_replay_manifest.json
```

Verify that the printed `Run ID` matches `metadata.run_id` in `reports/baseline_demo_results.json`.

## 4. Diagnostics & Structured Logs

- `demo` and `sensitivity` emit structured JSON logs to `stderr` (`timestamp_utc`, `level`, `message`, `run_id`, `strategy`, `completion_rate`, `det_total_is_bps`, `mc_mean_is_bps`, `capacity_capped_slices`).
- Every `SimulationResult.diagnostics` records:
  - `capacity_capped_slices`
  - `zero_liquidity_slices`
  - `shock_events_applied`
  - `spillover_events`
  - `partial_fill`
  - `max_observed_participation`
  - `conservation_error`

## 5. Failure Modes & Recovery Steps

| Symptom / Error | Root Cause | Recovery |
|---|---|---|
| Exit code `1`, `FileNotFoundError` | Missing scenario, grid config, or replay manifest path. | Check relative path from `optimal-execution-research-engine/` or pass an absolute path. |
| Exit code `1`, `ValidationError` | Negative quantity/spread/volatility, mismatched `volume_curve` length, missing required JSON keys, or invalid JSON syntax. | Run `python -m optimal_execution.cli validate --scenario <path>` and fix reported field(s) per `docs/contracts.md`. |
| Exit code `1`, `UnsupportedScenarioError` | Non-`1.0.0` `schema_version`, unknown strategy, or `temporary_impact_alpha != 1.0` with `almgren_chriss_analytic`. | Use `almgren_chriss_numeric` for nonlinear impact ($\alpha \ne 1$) or upgrade/align the scenario `schema_version`. |
| Exit code `1`, `ScheduleError` (`eta_tilde <= 0`) | Permanent impact $\frac{1}{2}\gamma\tau$ equals or exceeds temporary impact $\eta$, breaking convexity. | Increase `temporary_impact_eta`, reduce `permanent_impact_gamma`, or shorten `slice_duration_minutes`. |
| Exit code `2` from `compare` | Analytic and numeric schedules diverged beyond `--tolerance` (expected if `--enforce-capacity` binds). | Inspect `relative_l2_diff` in output JSON; remove `--enforce-capacity` for unconstrained oracle verification. |
