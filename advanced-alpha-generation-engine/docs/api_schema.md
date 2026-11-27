# API & Domain Schema Reference

## Canonical Domain Contracts (`src/alpha_engine/contracts/models.py`)

All timestamps MUST be timezone-aware UTC (`pl.Datetime("us", "UTC")` in Polars frames, `datetime(..., tzinfo=timezone.utc)` in Pydantic models). All monetary fields explicitly carry `currency` (default `"USD"`).

### 1. Input Schemas

- **`MarketBar`**: `symbol`, `event_ts_utc`, `available_at_utc` (`>= event_ts_utc`), `open`, `high` (`>= low`), `low`, `close`, `adjusted_close`, `volume`, `currency`, `split_factor`, `dividend_cash`.
- **`FundamentalRecord`**: `symbol`, `period_end_utc`, `event_ts_utc`, `available_at_utc` (`>= period_end_utc` and `>= event_ts_utc`), `earnings_ttm_usd`, `book_value_usd`, `market_cap_usd`, `roe`, `currency`.
- **`CorporateAction`**: `action_id`, `symbol`, `action_type` (`split | reverse_split | dividend | delisting`), `ex_date_utc`, `available_at_utc`, `ratio`, `cash_amount_usd`, `currency`.
- **`UniverseMembership`**: `symbol`, `sector`, `entry_ts_utc`, `exit_ts_utc` (nullable), `exit_reason` (nullable).
- **`DatasetManifest`**: `manifest_version`, `dataset_id`, `generated_at_utc`, `source_type` (`synthetic | permitted_historical`), `entitlements`, `license`, `seed`, `n_symbols`, `n_delisted_symbols`, `n_trading_days`, `file_checksums` (SHA-256), `manifest_digest`.

### 2. Signal & Evaluation Schemas

- **`SignalDefinition`**: `signal_id`, `name`, `version` (SemVer `X.Y.Z`), `family`, `description`, `parameters`, `lookback_bars`, `horizon_bars`, `inputs_required`, `causal_guarantee`, computed `definition_hash` (16-char SHA-256 prefix).
- **`SignalRecord`**: `signal_id`, `signal_version`, `symbol`, `event_ts_utc`, `available_at_utc` (`>= event_ts_utc`), `raw_value`, `neutralized_value`.
- **`EvaluationReport`**: Full walk-forward OOS report including `trial_id`, `dataset_manifest_digest`, `oos_mean_ic`, `oos_std_ic`, `oos_icir`, `oos_ic_tstat`, `oos_ic_pvalue`, `bh_adjusted_pvalue`, `bootstrap_ic_ci_low/high`, `avg_daily_turnover`, `max_abs_sector_exposure`, `max_peer_correlation`, `stability` (`StabilityReport`), `folds` (`list[FoldMetrics]`), `cost_sensitivity` (`list[CostSensitivityPoint]`), `passed_gates`, `gate_failures`, and `negative_result_notes`.
- **`TrialRecord`**: Immutable append-only record written to `trials.jsonl` for every evaluated trial (both passing and failing).
- **`CatalogEntry`**: Versioned entry keyed by `signal_id@version` with `status` (`candidate | promoted | rejected`), `promoted_at_utc`, `promoted_by`, and `evaluation_summary`.
- **`PaperSignalExport`**: `export_id`, `exported_at_utc`, `mode="paper_only"`, `execution_enabled=False`, `signal_id`, `signal_version`, `definition_hash`, `as_of_utc`, `records_count`, `gross_exposure`, `net_exposure`, `provenance`, `weights`, `sha256_digest`.

## Structured Domain Error Catalog (`src/alpha_engine/errors.py`)

All exceptions inherit from `AlphaEngineError` and implement `.to_dict()` returning `{error_code, error_type, message, details}`:

| Error Class | Error Code | Trigger Condition |
|---|---|---|
| `DataValidationError` | `DATA_VALIDATION_ERROR` | Missing columns, naive timestamps, checksum mismatch, insufficient history |
| `LookAheadViolationError` | `LOOK_AHEAD_VIOLATION` | `available_at_utc < event_ts_utc`, future data beyond `as_of_utc`, unlagged filings, overlapping WF folds |
| `SurvivorshipViolationError` | `SURVIVORSHIP_VIOLATION` | Delisted universe members omitted from bars or bars present after `exit_ts_utc` |
| `CorporateActionIntegrityError` | `CORPORATE_ACTION_INTEGRITY_ERROR` | Unadjusted split/reverse-split price cliff detected in `adjusted_close` |
| `MissingSymbolError` | `MISSING_SYMBOL_ERROR` | Active symbol coverage falls below `min_symbol_coverage_ratio` or `min_active_symbols_per_day` |
| `FactorDuplicationError` | `FACTOR_DUPLICATION_ERROR` | Candidate cross-sectional correlation with a promoted peer exceeds `max_peer_factor_correlation` |
| `HighTurnoverError` | `HIGH_TURNOVER_ERROR` | Turnover breaches policy threshold |
| `UnstableCoefficientsError` | `UNSTABLE_COEFFICIENTS_ERROR` | Coefficient sign-flip rate or CV exceeds policy threshold |
| `PromotionGateError` | `PROMOTION_GATE_ERROR` | Attempt to promote or export a signal that failed gates or lacks approval metadata |
| `ExecutionDisabledError` | `EXECUTION_DISABLED_ERROR` | Attempt to enable live order execution in config or exporter |
| `OptionalDependencyError` | `OPTIONAL_DEPENDENCY_ERROR` | Optional native/extra dependency (e.g., `xgboost` / `libomp`) unavailable |
