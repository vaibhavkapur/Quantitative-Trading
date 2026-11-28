# Operator & Developer Runbook

## 1. Clean Environment Setup

Prerequisites: `uv` (`>=0.13`) and Python `3.11+` (`3.12` recommended).

```bash
cd statistical-arbitrage-research-platform
uv sync --python 3.12 --extra dev --frozen
```

Verify installation:

```bash
uv run python -c "import statarb; print(statarb.__version__)"
```

## 2. Demo Walkthrough

### Step 1: Generate deterministic synthetic fixtures

```bash
uv run statarb generate-fixtures --output-dir data/fixtures
```

Creates `data/fixtures/reference_pair_*`, `data/fixtures/synthetic_universe_*`, `data/fixtures/invalid/*`, and `data/fixtures/manifest.json`.

### Step 2: Validate configuration and input data contracts

```bash
uv run statarb validate --config config/default.toml --project-root .
```

Expected output: JSON with `"status": "VALID"`, `config_hash`, `prices_hash`, and `borrow_hash`.

### Step 3: Run walk-forward backtests and generate versioned reports

```bash
uv run python scripts/run_demo.py
```

Or invoke a specific configuration via CLI:

```bash
uv run statarb run --config config/default.toml --project-root . --output-dir reports
```

Outputs written to `reports/`:
- `<run_id>_summary.json` and `<run_name>_latest.json`
- `<run_id>_equity_curve.csv`
- `<run_id>_trades.csv`

## 3. Validation & CI Checks

Run linter, full test suite, and determinism verification:

```bash
uv run ruff check src tests scripts
uv run pytest
```

Verify bit-for-bit rerun reproducibility:

```bash
uv run statarb run --config config/default.toml --project-root . --output-dir /tmp/statarb_r1 --quiet
uv run statarb run --config config/default.toml --project-root . --output-dir /tmp/statarb_r2 --quiet
diff -u /tmp/statarb_r1/default_walkforward_demo_latest.json /tmp/statarb_r2/default_walkforward_demo_latest.json
```

An empty `diff` confirms deterministic logical outputs across reruns.

## 4. Diagnostics & Structured Logging

The pipeline emits newline-delimited JSON events to stdout (`pipeline_start`, `pipeline_complete`) containing `ts_utc`, `run_id`, `config_hash`, `sharpe_net`, and `reconciliation_passed`.

Diagnostic counters recorded in `report["diagnostics"]`:

| Counter | Interpretation |
|---|---|
| `folds_executed` / `folds_with_selected_spreads` | Total walk-forward folds and how many had spreads surviving cointegration + FDR filters |
| `spread_entries` | Signal state machine entries across all OOS folds |
| `spread_exits_mean_revert` / `spread_exits_stop_loss` / `spread_exits_time_stop` | Breakdown of exit triggers |
| `borrow_blocked_entries` | Candidate spread entries skipped because a short leg lacked borrow availability |
| `borrow_forced_recalls` | Open spreads unwound due to mid-trade loss of borrow availability |
| `borrow_capped_trades` | Spreads scaled down proportionally to stay within `max_short_notional` |
| `missing_price_skips` / `delist_liquidations` | Missing price bars encountered and delisted positions closed cleanly at last known price |
| `fold_end_flattens` | End-of-fold position flattenings |

## 5. Failure Modes & Recovery Procedures

| Failure Symptom | Root Cause | Recovery Step |
|---|---|---|
| `ERROR [ConfigurationError]: Configuration file not found` | Incorrect `--config` path | Verify path (e.g., `config/default.toml`) and re-run `statarb validate`. |
| `ERROR [ConfigurationError]: Invalid configuration: Extra inputs are not permitted` | Typo or unsupported key in `.toml`/`.json` | Check `docs/contracts.md` or `config/default.toml` for exact field names. |
| `ERROR [MissingDataError]: prices contains N null/non-finite price bars` | Corrupted or gapped price input | Repair upstream data or explicitly set `data.missing_price_policy = "drop_symbol"` if dropping incomplete tickers is acceptable. |
| `ERROR [ContractValidationError]: Corporate action inconsistency` | Unadjusted split passed in `close_adj` | Backward-adjust `close_adj` for `split_factor` before ingestion and re-run `statarb validate`. |
| `ERROR [BorrowUnavailableError]: Borrow unavailable for ...` | Running with `strict_borrow_errors = true` against a hard-to-borrow fixture | Set `execution.strict_borrow_errors = false` to allow graceful blocking/recall, or update borrow schedule. |
| `ERROR [LeakageViolationError]` | Split boundary misconfiguration in custom caller code | Ensure `train_end < test_start` and pass `max_allowed_date = fold.train_end` when invoking domain functions directly. |
| `ERROR [ReconciliationError]` | Trade ledger disagrees with daily positions or equity | Treat as a critical bug; inspect `<run_id>_trades.csv` and `<run_id>_equity_curve.csv` before trusting any reported metric. |
