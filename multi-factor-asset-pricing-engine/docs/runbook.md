# Developer & Operator Runbook

## 1. Local Setup (Clean Checkout)

From `multi-factor-asset-pricing-engine/`:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/python setup.py develop --no-deps
```

Verify installation:

```bash
.venv/bin/mfape validate --config config/default_config.json
```

## 2. Regenerating Deterministic Fixtures

```bash
.venv/bin/python scripts/generate_fixtures.py
```

This deterministically regenerates all synthetic, analytical reference, universe-change, and invalid fixtures under `data/fixtures/`.

## 3. Running the End-to-End Demo

Option A — Python demo script (runs reference oracle, 15-year synthetic FF3 demo, and replay verification):

```bash
.venv/bin/python scripts/run_demo.py
```

Option B — CLI commands directly:

```bash
# Validate configuration
.venv/bin/mfape validate --config config/default_config.json

# Execute full run and write versioned outputs to reports/demo_run/
.venv/bin/mfape run --config config/default_config.json --base-dir .

# Replay and cryptographically verify input fingerprints & run_id
.venv/bin/mfape replay --manifest reports/demo_run/replay_manifest.json --base-dir .
```

## 4. Running the Validation Test Suite

```bash
.venv/bin/pytest -v
```

## 5. Diagnostics and Troubleshooting

All CLI failures exit with code `1` and emit a structured JSON payload to `stderr` containing `error_code`, `error_type`, `message`, and `details`. Run outputs also include a `diagnostics` block inside `run_result.json` with row/period counters and structured event logs.

### Common Failure Codes & Recovery Steps

| Error Code | Root Cause | Recovery Action |
|---|---|---|
| `CONFIGURATION_ERROR` | Missing JSON file, syntax error, missing required field, or `baseline_model_name` not listed in `models` | Inspect `details.missing_fields` or JSON syntax; re-run `mfape validate`. |
| `MISSING_RISK_FREE` | One or more aligned periods lack a finite `rf` observation in `risk_free.csv` | Backfill the dates listed in `details.missing_rf_dates`, or set `"rf_missing_policy": "drop_period"` to drop affected periods explicitly. |
| `PERIOD_ALIGNMENT_ERROR` | Asset and factor date sets differ, or period gaps violate `frequency` cadence | Inspect `details.dates_in_assets_not_factors` / `details.violations`. Reconcile upstream period-end dates, or set `"period_alignment_policy": "inner"` if partial intersection is intended. |
| `UNIVERSE_CHANGE_ERROR` | Asset entered or exited mid-sample under `"universe_policy": "strict_balanced"` | Inspect `details.changes`. Switch to `"pit_active"` to model unbalanced PIT panels or `"min_obs_balanced"` to restrict to always-active core assets. |
| `DATE_LEAKAGE_ERROR` | `split.test_start <= split.train_end` | Ensure `test_start` is strictly later than `train_end` (e.g., `train_end: "2019-12-31"`, `test_start: "2020-01-31"`). |
| `RANK_DEFICIENCY_ERROR` | Collinear factors in time-series regression or collinear/constant betas in Fama-MacBeth | Inspect `details.columns` and `details.singular_values`. Drop redundant factors or verify cross-sectional beta dispersion across test portfolios. |
| `DATA_CONTRACT_ERROR` (during `replay`) | Input file SHA-256 hash differs from `replay_manifest.json` | Restore the original versioned input files, or pass `--skip-fingerprint-check` only when intentionally re-evaluating with updated data. |

## 6. Rerun and Replay Procedure

1. Locate `replay_manifest.json` in the archived report directory (e.g., `reports/demo_run/replay_manifest.json`).
2. Run `.venv/bin/mfape replay --manifest reports/demo_run/replay_manifest.json --base-dir .`.
3. Confirm exit code `0`, `"status": "replayed_ok"`, `"fingerprints_verified": true`, and matching `run_id`.
