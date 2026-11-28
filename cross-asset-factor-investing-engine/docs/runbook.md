# Operator & Developer Runbook

## 1. Local Setup (Clean Checkout)

From `cross-asset-factor-investing-engine/`:

```bash
# Install pinned Python 3.12 dependencies into .venv using uv
python3 -m uv sync --python 3.12

# Verify installation
.venv/bin/python -c "import cross_asset_factor_engine as cafe; print(cafe.__version__)"
```

---

## 2. Standard Operational Workflows

### Regenerate Deterministic Fixtures
```bash
.venv/bin/python scripts/generate_fixtures.py
```

### Validate a Dataset Against Canonical Contracts
```bash
.venv/bin/cafe validate --dataset data/fixtures/synthetic_multi_year --base-currency USD
```

### Run End-to-End Demo & Archive Versioned Reports
```bash
.venv/bin/python scripts/run_demo.py
```
Or via the CLI with a custom config / output path:
```bash
.venv/bin/cafe run --config config/default.yaml --output-dir reports/runs
```

### Verify Deterministic Replay / Rerun Equivalence
```bash
.venv/bin/cafe verify-replay --config config/default.yaml
```
Exits `0` if two independent runs produce identical `run_id`, strategy metrics, class attribution, weights, and daily returns within `1e-12` tolerance; exits `2` on divergence.

### Run Full Automated Test Suite
```bash
.venv/bin/pytest tests -v
```

---

## 3. Diagnostics & Structured Logs

Every run writes `reports/runs/<run_id>/diagnostics.jsonl` and embeds `diagnostics_counters` inside `summary.json`:

| Counter | Meaning | Healthy Expectation |
|---|---|---|
| `prices_ingested` | Total price rows processed | `n_assets * n_days` |
| `roll_events_processed` | Futures contract rolls accounted for | Matches expected roll calendar |
| `fx_conversions_applied` | Non-base-currency rows FX-converted | Matches non-USD asset rows |
| `rows_blocked_by_timestamp_gate` | Lagged records prevented from leaking before `publication_ts` | `> 0` when publication lags exist |
| `signals_computed_ok` | Signals computed normally (or audited low-vol fallback) | High coverage post warmup |
| `signals_missing_input` | Signals missing required factor inputs | `0` on complete fixtures; explicit count when partial |
| `rebalances_optimized` | Constrained SLSQP solves completed | Matches `n_strategies * n_rebalance_dates` |
| `constraint_checks_passed` | Post-solve constraint audits passed | Equals `rebalances_optimized` |

---

## 4. Failure Modes & Recovery Procedures

All failures emit structured JSON on `stderr` and return exit code `1` (never silent success).

| Error Code | Typical Cause | Recovery Action |
|---|---|---|
| `CONFIGURATION_ERROR` | Missing YAML path, typo in key, `target_volatility > volatility_cap`, or `execution_lag_days < 1` | Inspect JSON `error.details`, fix YAML config, re-run `cafe run` |
| `SCHEMA_VALIDATION_ERROR` | Missing required CSV columns, negative/null price, duplicate `(date, asset_id)` | Run `cafe validate --dataset <dir>` to identify offending table/columns and fix upstream adapter |
| `UNSUPPORTED_ASSET_ERROR` | Asset class not in `{equity, fx, commodity}` or mismatched `carry_convention` | Align `contract_metadata.csv` with `docs/contracts.md` or extend engine conventions with an ADR |
| `CURRENCY_CONVERSION_ERROR` | Missing currency or trading date gap in `fx_rates.csv` | Backfill missing currency rates for all trading dates in `prices.csv` |
| `ROLL_NORMALIZATION_ERROR` | `contract_code` changed without positive `prev_front_price_new_contract` | Populate prior trading day's closing price of the rolled-into contract on the transition row |
| `TIMESTAMP_GATE_ERROR` | `publication_ts < as_of_ts` in source feed | Fix upstream timestamp clock/timezone normalization to UTC before re-ingesting |
| `MISSING_FACTOR_ERROR` | Missing factor input with `strict_missing_factors: true` | Supply missing `factor_inputs.csv` rows or explicitly set `strict_missing_factors: false` to audit missingness without aborting |
| `CONSTRAINT_VIOLATION_ERROR` | Tight turnover cap prevents unwinding an out-of-bounds prior portfolio below exposure/vol caps, or conflicting net/gross bounds | Inspect `error.details.violations`; relax `max_turnover_per_rebalance` or verify initial/previous portfolio feasibility |
