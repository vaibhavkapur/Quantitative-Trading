# Data Contracts & Schemas

**Contract Schema Version:** `1.0.0`

## 1. Input Contracts

### 1.1 Versioned Strategy Return Series (`series/<series_id>.csv`)

| Column | Required | Type | Units & Semantics | Validation Rules |
|---|---|---|---|---|
| `timestamp` | Yes | ISO-8601 string (`YYYY-MM-DD` or UTC `YYYY-MM-DDTHH:MM:SSZ`) | Period end timestamp | Strictly increasing, unique, non-empty |
| `gross_return` | Yes | float64 | Simple decimal return (`0.01` = +1%) | Finite (no NaN/Inf), `>= -1.0` |
| `turnover` | No | float64 | One-way portfolio turnover fraction (`0.10` = 10%) | Finite, `>= 0.0` (defaults to `0.0`) |
| `short_exposure` | No | float64 | Gross short leverage fraction (`0.25` = 25% short) | Finite, `>= 0.0` (defaults to `0.0`) |
| `benchmark_return` | No | float64 | Aligned simple benchmark decimal return | Finite, `>= -1.0` |

**Supported frequencies:** `daily` (252 periods/year), `weekly` (52), `monthly` (12). Any other frequency raises `UnsupportedConfigurationError`.

**Alignment invariant:** All completed trials within an experiment evaluation must share identical `timestamps` and `frequency`; mismatches raise `ContractValidationError`.

### 1.2 Cost Model (`CostModel`)

| Field | Type | Units | Default | Constraints |
|---|---|---|---|---|
| `commission_bps` | float | One-way basis points per unit turnover (`1 bps = 1e-4`) | `2.0` | Finite, `>= 0` |
| `slippage_bps` | float | One-way slippage bps per unit turnover | `3.0` | Finite, `>= 0` |
| `borrow_annual_bps` | float | Annualized short borrow bps on `short_exposure` | `50.0` | Finite, `>= 0` |
| `fixed_cost_per_period` | float | Fixed decimal drag per period | `0.0` | Finite, `>= 0` |
| `annualization_factor` | int | Periods per year for borrow proration | `252` | Integer `> 0` |

**Net return formula:**
$$\text{cost}_t = \text{turnover}_t \cdot (\text{commission\_bps} + \text{slippage\_bps}) \cdot 10^{-4} + \text{short\_exposure}_t \cdot \frac{\text{borrow\_annual\_bps} \cdot 10^{-4}}{\text{annualization\_factor}} + \text{fixed\_cost}$$
$$r^{\text{net}}_t = \max(-1.0,\; r^{\text{gross}}_t - \text{cost}_t)$$

### 1.3 Experiment Registry Ledger (`registry.jsonl`)

Each line is a canonical JSON `RegistryEntry`:

```json
{
  "prev_hash": "<64-hex SHA-256 of previous entry or 64 zeros for genesis>",
  "record": {
    "trial_id": "trial_lb20_vt10",
    "sequence_number": 5,
    "experiment_id": "exp-known-signal-v1",
    "strategy_name": "synthetic_momentum_plateau",
    "parameters": {"lookback": 20, "vol_target": 0.1},
    "status": "COMPLETED | FAILED | REJECTED | ABORTED",
    "created_at_utc": "2026-01-15T10:05:00Z",
    "dataset_version": "synth-2026.01",
    "code_version": "0.1.0",
    "cost_model": { "...": "..." },
    "series_id": "trial_lb20_vt10",
    "return_series_sha256": "<64-hex SHA-256 of rounded gross returns>",
    "failure_reason": null,
    "metrics_summary": {},
    "split_boundaries": [],
    "tags": {}
  },
  "entry_hash": "<SHA-256 of canonical JSON of {prev_hash, record}>"
}
```

**Ledger invariants:**
- `trial_id` must be unique across the ledger.
- `sequence_number` must be a strictly increasing integer `>= 1`. Gaps in `1..max_seq` are flagged as potential trial omissions by `integrity.py`.
- `COMPLETED` trials MUST specify `series_id`.
- `FAILED`, `REJECTED`, and `ABORTED` trials MUST specify a non-empty `failure_reason`.

### 1.4 Parameter Grid Manifest (`parameter_grid.json`)

```json
{
  "grid_id": "grid-known-signal-v1",
  "version": "1.0.0",
  "parameters": {
    "lookback": [15, 20, 25],
    "vol_target": [0.08, 0.10, 0.12]
  }
}
```

### 1.5 Policy Configuration (`config/default_policy.json`)

Requires top-level sections: `policy_version`, `seed`, `dsr`, `pbo`, `bootstrap`, `walk_forward`, `perturbation`, `integrity`, `promotion`.

## 2. Error Taxonomy

| Exception Class | Trigger Condition | CLI Exit Code |
|---|---|---|
| `ContractValidationError` | Missing file/columns, NaN/Inf, return `< -1.0`, non-monotonic timestamps, negative costs, zero-vol non-zero return, duplicate `trial_id` | `2` |
| `InsufficientSampleError` | Observations `< min_observations`, PBO block size `< 5` or `< 2` trials, walk-forward window exceeds sample length | `2` (or skips diagnostic & forces `REJECT` in pipeline) |
| `UnsupportedConfigurationError` | Unknown frequency, odd or `< 4` CSCV partitions, unsupported bootstrap or walk-forward mode | `2` |
| `RegistryIntegrityError` | Broken SHA-256 chain, tampered entry, malformed JSONL line, attempting append on corrupted ledger | `2` |

## 3. Output Contract (`reports/*.json`)

Every evaluation produces a versioned JSON payload (`schema_version: "1.0.0"`) containing:
- `run_id`: Deterministic `run-<16-hex>` hash of `(experiment_id, candidate_trial_id, registry_head_hash, policy_sha256, seed)`.
- `runtime`: Python, NumPy, SciPy, pandas versions, OS platform, `policy_sha256`, `seed`, `data_provenance`, UTC timestamp.
- `counters`: Trial counts, observation counts, completed/missing diagnostic counts, fold/split/resample/warning counts.
- `cost_summary`: Gross vs. net compounded return and bps drag.
- `verdict`: `decision` (`PROMOTE` | `HOLD_FOR_REVIEW` | `REJECT`), `evidence_complete`, `missing_evidence`, `gates[]`, `review_notes[]`, `summary`.
- `diagnostics`: Full serialized outputs for `integrity`, `dsr`, `pbo`, `bootstrap`, `walk_forward`, `perturbation`, and `outliers` (each `null` if unavailable, which triggers incomplete-evidence rejection).
- `limitations_notice`: Explicit methodology disclaimer.

## 4. Fixture Catalog (`data/fixtures/`)

All fixtures are generated deterministically by `scripts/generate_fixtures.py` with `provenance="synthetic_fixture"`.

- **Valid fixtures (`data/fixtures/valid/`):**
  - `known_signal_experiment/`: 360-day 3×3 plateau with 8 completed trials + 1 logged `FAILED` trial (100% grid coverage) and synthetic benchmark.
  - `null_overfit_experiment/`: 360-day 4×4 grid of 16 zero-drift Gaussian noise trials.
  - `cherry_picked_experiment/`: 360-day experiment with missing grid configs (5/9), sequence gaps, and exact duplicate returns.
  - `hand_calculated_reference.json`: 12-period analytical oracle verified to `< 1e-10` tolerance.
- **Invalid fixtures (`data/fixtures/invalid/`):**
  - `non_monotonic_timestamps.csv`, `nan_returns.csv`, `below_minus_100pct.csv`, `tampered_registry.jsonl`.
- **Boundary fixtures (`data/fixtures/boundary/`):**
  - `insufficient_sample_experiment/`: 24-observation series triggering insufficient-sample / incomplete-evidence rejection.
  - `extreme_outlier_experiment/`: 360-observation series dominated by two +22%/+25% single-day spikes.
