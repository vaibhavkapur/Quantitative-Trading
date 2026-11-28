# Data Contracts & Schema Specification

**Schema Version:** `1.0.0`

## 1. Input Contract: `ExecutionScenario` (JSON)

### Top-Level Fields

| Field | Type | Units / Semantics | Validation Rules |
|---|---|---|---|
| `schema_version` | `string` | SemVer | Must equal `"1.0.0"`; otherwise raises `UnsupportedScenarioError`. |
| `scenario_id` | `string` | Unique identifier | Non-empty trimmed string. |
| `description` | `string` | Human note | Optional (default `""`). |
| `capacity_policy` | `string` | `"none"` or `"spillover"` | Optional (default `"none"`). Unsupported values raise `UnsupportedScenarioError`. |
| `seed` | `integer` | PRNG seed | Non-negative integer (default `42`). |
| `num_monte_carlo_paths` | `integer` | Count | Integer `>= 1` (default `500`). |
| `order` | `object` | `ParentOrder` | Required. |
| `market` | `object` | `MarketProfile` | Required. |
| `impact` | `object` | `ImpactParams` | Required. |

### `order` (`ParentOrder`)

| Field | Type | Units | Validation Rules |
|---|---|---|---|
| `order_id` | `string` | Identifier | Non-empty string. |
| `symbol` | `string` | Synthetic ticker | Non-empty string (synthetic symbols only, e.g., `SYN_LGCAP`). |
| `side` | `string` | `"buy"` or `"sell"` | Buy sign `+1`, sell sign `-1`. |
| `quantity` | `number` | Shares | Finite float `> 0`. |
| `arrival_price` | `number` | USD / share | Finite float `> 0`. |
| `horizon_slices` | `integer` | Count ($N$) | Integer `>= 1`. |
| `slice_duration_minutes` | `number` | Minutes | Finite float `> 0` (default `30.0`; trading day = `390.0` min). |
| `start_timestamp_utc` | `string` | ISO-8601 UTC | Start timestamp of slice 0 (default `"2026-01-05T14:30:00Z"`). Slice $k$ covers $[t_0 + k\Delta t,\; t_0 + (k+1)\Delta t)$. |

### `market` (`MarketProfile`)

| Field | Type | Units | Validation Rules |
|---|---|---|---|
| `volume_curve` | `array<number>` | Shares / slice | Length must equal `order.horizon_slices`; every entry finite and `>= 0`. |
| `daily_volatility` | `number` | Fraction / $\sqrt{\text{day}}$ | Finite float `>= 0` (e.g., `0.02` = 2% daily). |
| `bid_ask_spread_bps` | `number` | Basis points (full spread) | Finite float `>= 0`. Half-spread $\epsilon = 0.5 \cdot \text{bps} \cdot 10^{-4} \cdot S_0$. |
| `max_participation_rate` | `number` | Fraction of slice volume | Finite float in `(0, 1]` (default `0.25`). |
| `max_slice_quantity` | `number \| null` | Shares / slice | Optional absolute cap `>= 0`. |
| `price_shocks` | `array<PriceShock>` | Exogenous jumps | Optional list of `{slice_index: int in [0, N-1], shock_bps: finite float, label: str}`. |

### `impact` (`ImpactParams`)

| Field | Type | Units | Validation Rules |
|---|---|---|---|
| `permanent_impact_gamma` | `number` | `USD / share^2` | Finite float `>= 0`. |
| `temporary_impact_eta` | `number` | `USD * day / share^(1 + alpha)` | Finite float `>= 0`. |
| `temporary_impact_alpha` | `number` | Dimensionless exponent | Finite float `> 0` (default `1.0`). Analytic Almgren-Chriss requires `alpha == 1.0` and $\tilde{\eta} = \eta - \frac{1}{2}\gamma\tau > 0$. |

## 2. Output Contracts

Every JSON report written by `io.save_json_report` has the top-level envelope:

```json
{
  "metadata": {
    "run_id": "run-<12-hex-sha256>",
    "schema_version": "1.0.0",
    "engine_version": "0.1.0",
    "timestamp_utc": "ISO-8601 UTC",
    "python_version": "3.x.y",
    "platform": "...",
    "package_versions": {"numpy": "...", "scipy": "...", "polars": "..."},
    "scenario_id": "...",
    "seed": 20260105,
    "config_sha256": "<64-hex-sha256>"
  },
  "data": { ... }
}
```

- **Cost units**: All monetary fields end in `_usd` (USD). Basis-point fields end in `_bps` and are normalized by parent order arrival notional ($S_0 \cdot Q$), so `1 bps = 0.01%` of arrival notional.
- **Sign convention**: Positive Implementation Shortfall (`total_is_usd > 0`, `total_is_bps > 0`) always indicates an execution **cost** (worse than arrival price) for both buy and sell orders.

## 3. Error Hierarchy

| Exception | Trigger Condition | CLI Exit Code |
|---|---|---|
| `ValidationError` | Missing required fields, non-finite/out-of-range numbers, negative volume/quantity, mismatched `volume_curve` length, malformed JSON. | `1` |
| `UnsupportedScenarioError` | Unknown `schema_version`, unsupported `strategy` or `capacity_policy`, nonlinear `temporary_impact_alpha != 1.0` passed to analytic Almgren-Chriss. | `1` |
| `ScheduleError` | Violation of Almgren-Chriss convexity condition ($\tilde{\eta} \le 0$) or numeric optimizer convergence failure. | `1` |
| `FileNotFoundError` | Missing scenario, grid config, or replay manifest path. | `1` |

## 4. Fixture Catalog (`data/fixtures/`)

| Fixture File | Category | Purpose |
|---|---|---|
| `reference_deterministic.json` | Valid / Reference | 4 quarter-day slices, 10,000 shares, exact hand-calculated oracle. |
| `reference_expected.json` | Oracle | Independently hand-calculated TWAP & VWAP expected costs and timing risk with tolerances (`1e-6` USD). |
| `zero_liquidity.json` | Boundary | Three zero-volume halt slices verifying zero fills and spillover handling. |
| `partial_fill_capacity.json` | Boundary | Sell order exceeding total market capacity (15,000 vs 6,000 shares), testing residual & opportunity cost. |
| `price_shock.json` | Integration | Mid-horizon `+150 bps` jump and `-40 bps` partial reversion verifying shock accounting. |
| `invalid_negative_quantity.json` | Invalid | Negative `quantity` (`-5000`); must raise `ValidationError`. |
| `invalid_missing_fields.json` | Invalid | Missing `market` and `impact` sections; must raise `ValidationError`. |
| `invalid_unsupported_schema.json` | Invalid / Unsupported | `schema_version: "99.0.0"`; must raise `UnsupportedScenarioError`. |
