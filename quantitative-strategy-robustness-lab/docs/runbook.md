# Operator & Developer Runbook

## 1. Clean Environment Setup

From `quantitative-strategy-robustness-lab/`:

```bash
# Option A: uv (recommended, matches pinned requirements.lock)
python3 -m uv venv .venv --python 3.12
python3 -m uv pip install --python .venv/bin/python -r requirements.lock
python3 -m uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e .

# Option B: standard venv + pip
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install --no-deps --no-build-isolation -e .
```

## 2. Running Validation & Demo Walkthrough

### Full CI Check (Fixtures + Tests + Demo + Replay)
```bash
bash scripts/ci_check.sh
```

### Generate Deterministic Fixtures Only
```bash
PYTHONPATH=src .venv/bin/python scripts/generate_fixtures.py
```

### Run Test Suite
```bash
PYTHONPATH=src .venv/bin/python -m pytest tests -v
```

### Run Reproducible End-to-End Demo
```bash
PYTHONPATH=src .venv/bin/python scripts/run_demo.py
```
Outputs are written to `reports/baseline/<scenario>.json` and `reports/baseline/<scenario>.md`, with an index at `reports/baseline/demo_summary.json`.

### CLI Operations

```bash
# Verify registry hash chain
PYTHONPATH=src .venv/bin/python -m robustness_lab.cli verify-registry \
  --ledger data/fixtures/valid/known_signal_experiment/registry.jsonl

# Evaluate a single experiment with structured JSON logs
PYTHONPATH=src .venv/bin/python -m robustness_lab.cli --verbose evaluate \
  --experiment-dir data/fixtures/valid/known_signal_experiment \
  --policy config/default_policy.json \
  --output-dir reports/manual \
  --report-name known_signal_check

# Deterministic replay against a saved baseline report
PYTHONPATH=src .venv/bin/python -m robustness_lab.cli replay \
  --experiment-dir data/fixtures/valid/known_signal_experiment \
  --policy config/default_policy.json \
  --baseline-report reports/baseline/known_signal.json
```

## 3. Diagnostics & Structured Logging

Pass `--verbose` to `robustness-lab` to emit newline-delimited JSON operational events:
- `evaluation_started`: experiment ID, logged/completed/failed counts, registry validity.
- `diagnostic_skipped_insufficient_sample`: explicit record when a diagnostic cannot run due to sample size (causes `evidence_complete = false` and `REJECT`).
- `report_written`: `run_id`, output JSON and Markdown paths.
- `evaluation_completed`: `run_id`, `candidate_trial_id`, `decision`, `evidence_complete`.

## 4. Failure Modes & Recovery Procedures

| Symptom / Error | Likely Cause | Recovery Action |
|---|---|---|
| `RegistryIntegrityError` / CLI exit `2` on `verify-registry` | Ledger line was manually edited, truncated, or deleted | Do **not** patch hashes manually. Restore the ledger from version control or re-run the full experiment sweep into a new versioned directory and ledger. |
| `ContractValidationError: Missing policy configuration` | Typo in `--policy` path or truncated JSON | Validate JSON against `config/default_policy.json` required sections (`dsr`, `pbo`, `bootstrap`, `walk_forward`, `perturbation`, `integrity`, `promotion`). |
| `decision: REJECT` with `evidence_complete: false` | Sample size too short for DSR/PBO/bootstrap/walk-forward, or missing CSV series | Inspect `verdict.missing_evidence` in the JSON report; extend the backtest history or supply missing return CSVs, then rerun. |
| `cherry_picking_flags` populated | Sequence gaps in `registry.jsonl`, unlogged grid points, or duplicate return series | Ensure the sweep harness logs **every** planned grid configuration (recording exceptions as `FAILED` with `failure_reason`) and deduplicates identical strategies before evaluation. |
| `replay` returns exit `3` (`matched: false`) | Inputs, policy config, or dependency version drifted since baseline | Inspect `diffs` output and compare `runtime.policy_sha256`, `registry_head_hash`, and `runtime.dependencies` between baseline and rerun. |
