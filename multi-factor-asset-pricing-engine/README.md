# Multi-Factor Asset Pricing Engine (`mfape`)

An independently developed educational and research prototype for estimating cross-sectional factor risk premia, fitting time-series CAPM and multi-factor regressions, and evaluating in-sample and out-of-sample (OOS) pricing errors without date leakage.

> **Disclaimer & Scope:** Linear factor models are statistical explanatory approximations—not causal proofs or trading/profitability guarantees. All included fixtures and benchmark results are **100% synthetic and deterministic**; no real transactions, customer records, credentials, or proprietary artifacts are used.

## Features

- **Point-in-time excess-return & factor-alignment pipeline (`mfape.pipeline`, Polars):** Validates schemas, decimal return units, ISO period-end date cadence, duplicate keys, missing risk-free rates, and point-in-time portfolio memberships (tracking universe entries/exits cleanly).
- **Time-series CAPM & multi-factor regressions (`mfape.time_series`, NumPy + statsmodels + SciPy):** Supports `OLS`, `HC3`, and Newey-West `HAC` standard errors, SVD rank/condition-number checks, per-asset alpha/beta uncertainty, residuals, and the Gibbons-Ross-Shanken (1989) joint zero-alpha F-test.
- **Fama-MacBeth cross-sectional regressions (`mfape.fama_macbeth`):** Period-by-period cross-sectional slopes with classical Fama-MacBeth (1973) standard errors, Shanken (1992) errors-in-variables (EIV) standard errors, and Newey-West HAC standard errors, plus cross-sectional $R^2$ and pricing error metrics (MAE, RMSE).
- **Zero-leakage OOS evaluation & CAPM baseline comparison (`mfape.evaluation`):** Enforces strictly chronological splits (`train_end < test_start`), freezes training betas and premia before scoring future periods, and compares every candidate model against the designated `CAPM` baseline.
- **Reproducible manifests & replay (`mfape.engine`, `mfape.cli`):** SHA-256 input file fingerprinting, content-addressed `run_id`, structured diagnostic counters, and `mfape replay` verification.

## Quickstart & Local Setup

From `multi-factor-asset-pricing-engine/`:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/python setup.py develop --no-deps
```

### Run the End-to-End Demo

```bash
# 1. (Optional) Regenerate deterministic synthetic fixtures
.venv/bin/python scripts/generate_fixtures.py

# 2. Run analytical oracle, 15-year synthetic FF3 demo, and replay verification
.venv/bin/python scripts/run_demo.py
```

Or invoke the CLI directly:

```bash
.venv/bin/mfape validate --config config/default_config.json
.venv/bin/mfape run --config config/default_config.json --base-dir .
.venv/bin/mfape replay --manifest reports/demo_run/replay_manifest.json --base-dir .
```

### Run Validation Tests

```bash
.venv/bin/pytest -v
```

## Example Output (Measured on Synthetic FF3 Fixture)

The table below shows actual measured outputs from `scripts/run_demo.py` on the **deterministic synthetic 25-portfolio FF3 fixture** (`seed=42`, train `2010-01-31`–`2019-12-31`, test `2020-01-31`–`2024-12-31`, HAC `maxlags=3`, `run_id: mfape-f7a36239df1960a9`):

| Model | Factors | TS Mean $R^2$ | TS Mean $|\alpha|$ | GRS $p$-val | FM CS $R^2$ | FM RMSE | OOS CS RMSE | OOS TS $R^2$ ($\alpha=0$) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `CAPM` (baseline) | `MKT_RF` | 0.8431 | 0.00443 | 0.0582 | 0.1820 | 0.00505 | 0.00693 | 0.8742 |
| `FF3` | `MKT_RF, SMB, HML` | 0.9575 | 0.00055 | 0.9557 | 0.9881 | 0.00061 | 0.00613 | 0.9712 |
| `FF4_MOM` | `MKT_RF, SMB, HML, MOM` | 0.9579 | 0.00055 | 0.9583 | 0.9881 | 0.00061 | 0.00613 | 0.9709 |

Training-window Fama-MacBeth factor premia for `FF3` (monthly decimal returns; annualized = `×12`):

| Term | Est (monthly) | Annualized | FM SE | FM $t$-stat | Shanken SE | Shanken $t$-stat | HAC SE | HAC $t$-stat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gamma_0` | 0.00009 | 0.0011 | 0.00070 | 0.13 | 0.00081 | 0.12 | 0.00074 | 0.13 |
| `MKT_RF` | 0.00733 | 0.0880 | 0.00382 | 1.92 | 0.00385 | 1.91 | 0.00407 | 1.80 |
| `SMB` | 0.00460 | 0.0552 | 0.00151 | 3.06 | 0.00151 | 3.04 | 0.00141 | 3.26 |
| `HML` | 0.00607 | 0.0728 | 0.00134 | 4.51 | 0.00135 | 4.49 | 0.00127 | 4.77 |

## Project Layout

```text
multi-factor-asset-pricing-engine/
├── src/mfape/                 # Core library & CLI (errors, schemas, io, pipeline, time_series, fama_macbeth, evaluation, engine, cli)
├── tests/                     # Unit, integration, boundary, reproducibility, and CLI tests (22 tests)
├── config/                    # Versioned JSON run configurations (default_config.json, reference_config.json)
├── data/fixtures/             # Deterministic analytical, synthetic FF3, universe-change, and invalid fixtures
├── scripts/                   # generate_fixtures.py and run_demo.py
├── docs/                      # architecture.md, contracts.md, integration.md, runbook.md, adr/0001-*.md
├── reports/                   # acceptance.md and raw versioned JSON/CSV/MD outputs (demo_run/, reference_run/)
├── pyproject.toml             # Build & dependency specification
├── setup.py                   # Editable-install compatibility shim
├── requirements-lock.txt      # Pinned dependency lockfile
├── plan.md                    # Implementation plan and completed milestones
└── README.md                  # This file
```

CI workflow: [`../.github/workflows/multi-factor-asset-pricing-engine.yml`](../.github/workflows/multi-factor-asset-pricing-engine.yml)

## Documentation & Acceptance Evidence

- **Acceptance Evidence:** [reports/acceptance.md](reports/acceptance.md) (maps every criterion to automated test results and raw run artifacts)
- **Raw Demo Outputs:** [reports/demo_run/summary.md](reports/demo_run/summary.md), [reports/demo_run/model_comparison.csv](reports/demo_run/model_comparison.csv), [reports/demo_run/run_result.json](reports/demo_run/run_result.json)
- **Architecture & ADR:** [docs/architecture.md](docs/architecture.md), [docs/adr/0001-architecture-and-stack.md](docs/adr/0001-architecture-and-stack.md)
- **Data Contracts:** [docs/contracts.md](docs/contracts.md)
- **Integration & Licenses:** [docs/integration.md](docs/integration.md)
- **Developer/Operator Runbook:** [docs/runbook.md](docs/runbook.md)

## Assumptions and Known Limitations

- **Static training betas in OOS evaluation:** First-pass betas are estimated strictly over the training window and held constant during the test window to maintain a clean zero-leakage boundary; rolling/conditional betas are not yet implemented.
- **Unbalanced PIT panels:** When `universe_policy="pit_active"` has mid-sample entries/exits, the exact GRS joint test is skipped with an explanatory `grs_note`, and Shanken standard errors use the conservative $\sqrt{1 + c}\,\text{SE}_{\text{FM}}$ scaling.
- **Offline synthetic evaluation only:** Empirical market datasets (e.g., CRSP / Ken French Data Library) and live trading integrations are intentionally excluded from this research prototype.
