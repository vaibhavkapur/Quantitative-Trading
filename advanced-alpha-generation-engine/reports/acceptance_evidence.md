# Acceptance Criteria Verification Evidence

Generated at: `2026-11-27T09:52:05.474838+00:00`

## Section 6 Acceptance Checklist

| Criterion | Status | Evidence Artifact |
|---|---|---|
| Every signal has an availability timestamp | PASS | Enforced by `SignalRecord` & `PointInTimeValidator.validate_signal_causality`; verified in `tests/unit/test_contracts_and_pit.py` and `tests/unit/test_factors.py` |
| Research results include out-of-sample IC and costs | PASS | `reports/m3_walk_forward_evaluation.json` (walk-forward OOS Spearman/Pearson IC, ICIR, bootstrap CI, cost sensitivity `[0, 5, 10, 25, 50]` bps) |
| Signal definitions are versioned | PASS | Semantic version + SHA-256 `definition_hash` in `SignalDefinition`; persisted in `reports/catalog/catalog.json` |
| Multiple trials are recorded | PASS | Append-only ledger `reports/catalog/trials.jsonl` with Benjamini-Hochberg FDR adjustment across all 12 evaluated candidates (including negative results) |
| Signals that fail validation cannot be promoted | PASS | Hard-enforced by `AlphaCatalog.promote()` raising `PromotionGateError`; verified in `tests/unit/test_catalog_and_exporter.py` and `reports/scenario_verification_evidence.json` |
| Clean checkout can run setup, tests, and demo with permitted fixtures | PASS | `pyproject.toml`, `uv.lock`, `scripts/setup.sh`, `scripts/demo.py`, `scripts/validate.py`, checksummed synthetic fixtures in `data/fixtures/manifest.json` |
| Architecture, dependency assumptions, known limitations, and runbook committed | PASS | `docs/architecture.md`, `docs/foundation_audit.md`, `docs/api_schema.md`, `docs/assumptions_and_limitations.md`, `docs/runbook.md`, and `docs/adr/ADR-001..003` |
| Relevant CI checks pass and raw evaluation/test evidence retained | PASS | `pytest` passed, `ruff` passed, [validation workflow](../../.github/workflows/advanced-alpha.yml) committed, raw JSON evidence saved under `reports/` |
| Security, data licensing, IP, and deployment approval requirements reviewed | PASS | `docs/foundation_audit.md`, `docs/runbook.md`, `data/fixtures/manifest.json`, `execution_enabled=false` hard guard in `EngineConfig` and `PaperSignalExporter` |

## Scenario Verification Summary (`plan.md` Section 5)

All 7 required failure-path scenarios passed with injected faults (raw evidence in `reports/scenario_verification_evidence.json`):

- **Survivorship bias**: Rejected with `SURVIVORSHIP_VIOLATION`
- **Corporate actions**: Rejected with `CORPORATE_ACTION_INTEGRITY_ERROR`
- **Look-ahead**: Rejected with `LOOK_AHEAD_VIOLATION`
- **Missing symbols**: Rejected with `MISSING_SYMBOL_ERROR`
- **Factor duplication**: Rejected with `PROMOTION_GATE_ERROR` / `FACTOR_DUPLICATION_ERROR`
- **High turnover**: Rejected with `PROMOTION_GATE_ERROR`
- **Unstable coefficients**: Flagged `is_stable=false` and blocked with `PROMOTION_GATE_ERROR`

## Demo Run Summary

- Promoted (paper-only): `['ml_gbdt_baseline@1.0.0', 'mom_20_5@1.0.0', 'stat_equal_composite@1.0.0']`
- Rejected / negative results recorded: `9` candidates
- Paper-only exports generated: `3` (`execution_enabled=false`)
