"""Versioned alpha catalog and append-only experiment trial registry with strict promotion gates."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from alpha_engine.config import RegistryPolicy
from alpha_engine.contracts.models import (
    CatalogEntry,
    EvaluationReport,
    SignalDefinition,
    TrialRecord,
)
from alpha_engine.errors import (
    DataValidationError,
    FactorDuplicationError,
    PromotionGateError,
)
from alpha_engine.evaluation.metrics import (
    benjamini_hochberg_adjust,
    compute_factor_correlation_matrix,
)


class AlphaCatalog:
    """Append-only trial registry and versioned signal catalog with non-bypassable promotion gates."""

    def __init__(self, storage_dir: Path, policy: RegistryPolicy) -> None:
        self.storage_dir = storage_dir
        self.policy = policy
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.trials_path = self.storage_dir / "trials.jsonl"
        self.catalog_path = self.storage_dir / "catalog.json"
        self.corr_path = self.storage_dir / "correlation_matrix.json"

        self._entries: dict[str, CatalogEntry] = {}
        self._reports: dict[str, EvaluationReport] = {}
        self._trials: list[TrialRecord] = []
        self._corr_matrix: dict[str, dict[str, float]] = {}
        self._load_existing()

    @staticmethod
    def _key(signal_id: str, version: str) -> str:
        return f"{signal_id}@{version}"

    def _load_existing(self) -> None:
        if self.trials_path.exists():
            with self.trials_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        self._trials.append(TrialRecord.model_validate(json.loads(line)))
        if self.catalog_path.exists():
            with self.catalog_path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
                for k, v in raw.items():
                    self._entries[k] = CatalogEntry.model_validate(v)
        if self.corr_path.exists():
            with self.corr_path.open("r", encoding="utf-8") as fh:
                self._corr_matrix = json.load(fh)

    def _append_trial(self, trial: TrialRecord) -> None:
        self._trials.append(trial)
        with self.trials_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(trial.model_dump(mode="json"), sort_keys=True) + "\n")

    def _persist_catalog(self) -> None:
        payload = {k: v.model_dump(mode="json") for k, v in sorted(self._entries.items())}
        with self.catalog_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        with self.corr_path.open("w", encoding="utf-8") as fh:
            json.dump(self._corr_matrix, fh, indent=2, sort_keys=True)

    @property
    def trials(self) -> list[TrialRecord]:
        return list(self._trials)

    @property
    def entries(self) -> dict[str, CatalogEntry]:
        return dict(self._entries)

    @property
    def correlation_matrix(self) -> dict[str, dict[str, float]]:
        return {k: dict(v) for k, v in self._corr_matrix.items()}

    def get_entry(self, signal_id: str, version: str) -> CatalogEntry:
        key = self._key(signal_id, version)
        if key not in self._entries:
            raise DataValidationError(
                f"Signal {key} not found in catalog.",
                details={"signal_key": key},
            )
        return self._entries[key]

    def register_batch(
        self,
        candidates: list[tuple[SignalDefinition, EvaluationReport, pl.DataFrame]],
    ) -> list[EvaluationReport]:
        """Register a batch of evaluated candidates with multiple-testing BH adjustment & correlation audit.

        Every trial — including negative and failed results — is permanently recorded.
        """
        if not candidates:
            return []

        # 1. Compute pairwise correlation matrix across candidate OOS neutralized panels
        frames = {defn.signal_id: oos_df for defn, _, oos_df in candidates}
        batch_corr = compute_factor_correlation_matrix(frames)
        for sid, row in batch_corr.items():
            self._corr_matrix.setdefault(sid, {}).update(row)

        # 2. Compute Benjamini-Hochberg adjusted p-values across historical + current batch trials
        historical_pvals = [t.oos_ic_pvalue for t in self._trials]
        batch_pvals = [rep.oos_ic_pvalue for _, rep, _ in candidates]
        all_adj = benjamini_hochberg_adjust(historical_pvals + batch_pvals)
        batch_adj = all_adj[len(historical_pvals) :]

        # Rank candidates by OOS ICIR descending so higher-quality candidates take precedence if two are duplicates
        indexed = list(enumerate(candidates))
        indexed.sort(key=lambda pair: pair[1][1].oos_icir, reverse=True)

        accepted_ids: set[str] = {
            e.signal_definition.signal_id
            for e in self._entries.values()
            if e.status == "promoted"
        }

        updated_reports: dict[int, EvaluationReport] = {}

        for orig_idx, (defn, rep, _) in indexed:
            adj_p = batch_adj[orig_idx]
            sid = defn.signal_id

            # Check correlation against already-promoted or higher-ranked passing batch peers
            corr_row = self._corr_matrix.get(sid, {})
            max_peer_corr = 0.0
            most_corr_peer: str | None = None
            for peer_id, corr_val in corr_row.items():
                if peer_id == sid:
                    continue
                if abs(corr_val) > max_peer_corr:
                    max_peer_corr = abs(corr_val)
                    most_corr_peer = peer_id

            # Specifically check duplication against already-accepted/promoted signals
            dup_peer: str | None = None
            dup_corr = 0.0
            for peer_id in accepted_ids:
                if peer_id == sid:
                    continue
                cval = abs(corr_row.get(peer_id, 0.0))
                if cval > dup_corr:
                    dup_corr = cval
                    dup_peer = peer_id

            # Rebuild gate failures incorporating BH-adjusted p-value and peer duplication
            failures = [f for f in rep.gate_failures if not f.startswith("bh_adjusted_pvalue=") and not f.startswith("max_peer_correlation=")]
            if adj_p > self.policy.max_bh_adjusted_pvalue:
                failures.append(
                    f"bh_adjusted_pvalue={adj_p:.4f} > max {self.policy.max_bh_adjusted_pvalue:.4f}"
                )
            if dup_peer is not None and dup_corr > self.policy.max_peer_factor_correlation:
                failures.append(
                    f"max_peer_correlation={dup_corr:.4f} with '{dup_peer}' > max {self.policy.max_peer_factor_correlation:.4f}"
                )

            passed = len(failures) == 0
            notes = (
                ""
                if passed
                else "Failed validation gates (recorded as negative/rejected trial): "
                + "; ".join(failures)
            )
            final_rep = rep.model_copy(
                update={
                    "bh_adjusted_pvalue": round(adj_p, 6),
                    "max_peer_correlation": round(max_peer_corr, 6),
                    "most_correlated_peer": most_corr_peer,
                    "passed_gates": passed,
                    "gate_failures": failures,
                    "negative_result_notes": notes,
                }
            )
            updated_reports[orig_idx] = final_rep
            self._reports[final_rep.trial_id] = final_rep

            if passed:
                accepted_ids.add(sid)

            trial = TrialRecord(
                trial_id=final_rep.trial_id,
                signal_id=defn.signal_id,
                signal_version=defn.version,
                definition_hash=defn.definition_hash,
                recorded_at_utc=datetime.now(UTC),
                dataset_manifest_digest=final_rep.dataset_manifest_digest,
                oos_mean_ic=final_rep.oos_mean_ic,
                oos_icir=final_rep.oos_icir,
                oos_ic_pvalue=final_rep.oos_ic_pvalue,
                bh_adjusted_pvalue=final_rep.bh_adjusted_pvalue,
                avg_daily_turnover=final_rep.avg_daily_turnover,
                max_abs_sector_exposure=final_rep.max_abs_sector_exposure,
                max_peer_correlation=final_rep.max_peer_correlation,
                net_sharpe_at_base_cost=final_rep.net_sharpe_at_base_cost,
                passed_gates=final_rep.passed_gates,
                gate_failures=final_rep.gate_failures,
                status="candidate" if final_rep.passed_gates else "failed",
                notes=final_rep.negative_result_notes,
            )
            self._append_trial(trial)

            key = self._key(defn.signal_id, defn.version)
            self._entries[key] = CatalogEntry(
                signal_definition=defn,
                latest_trial_id=final_rep.trial_id,
                status="candidate" if final_rep.passed_gates else "rejected",
                promoted_at_utc=None,
                promoted_by=None,
                evaluation_summary={
                    "oos_mean_ic": final_rep.oos_mean_ic,
                    "oos_icir": final_rep.oos_icir,
                    "bh_adjusted_pvalue": final_rep.bh_adjusted_pvalue,
                    "bootstrap_ic_ci": [
                        final_rep.bootstrap_ic_ci_low,
                        final_rep.bootstrap_ic_ci_high,
                    ],
                    "avg_daily_turnover": final_rep.avg_daily_turnover,
                    "max_abs_sector_exposure": final_rep.max_abs_sector_exposure,
                    "max_peer_correlation": final_rep.max_peer_correlation,
                    "most_correlated_peer": final_rep.most_correlated_peer,
                    "net_sharpe_at_base_cost": final_rep.net_sharpe_at_base_cost,
                    "passed_gates": final_rep.passed_gates,
                    "gate_failures": final_rep.gate_failures,
                },
            )

        self._persist_catalog()
        return [updated_reports[i] for i in range(len(candidates))]

    def promote(
        self,
        signal_id: str,
        version: str,
        *,
        approver: str,
        approval_ticket: str,
    ) -> CatalogEntry:
        """Promote a passing versioned candidate after verifying gates, peer correlation, and approval."""
        if not approver.strip() or not approval_ticket.strip():
            raise PromotionGateError(
                "Promotion requires explicit approver and approval_ticket metadata.",
                details={"signal_id": signal_id, "version": version},
            )

        key = self._key(signal_id, version)
        entry = self.get_entry(signal_id, version)

        # Find latest trial record
        trial = next((t for t in reversed(self._trials) if t.trial_id == entry.latest_trial_id), None)
        if trial is None:
            raise PromotionGateError(
                f"Cannot promote {key}: trial record '{entry.latest_trial_id}' missing.",
                details={"signal_key": key},
            )

        if not trial.passed_gates:
            raise PromotionGateError(
                f"Signal {key} failed validation gates and CANNOT be promoted.",
                details={
                    "signal_key": key,
                    "trial_id": trial.trial_id,
                    "gate_failures": trial.gate_failures,
                },
            )

        # Check duplication against already-promoted signals
        corr_row = self._corr_matrix.get(signal_id, {})
        for other_key, other_entry in self._entries.items():
            if other_key == key or other_entry.status != "promoted":
                continue
            other_sid = other_entry.signal_definition.signal_id
            corr_val = abs(corr_row.get(other_sid, 0.0))
            if corr_val > self.policy.max_peer_factor_correlation:
                raise FactorDuplicationError(
                    f"Cannot promote {key}: |corr|={corr_val:.3f} with promoted signal '{other_sid}' "
                    f"exceeds max {self.policy.max_peer_factor_correlation:.3f}.",
                    details={
                        "signal_key": key,
                        "promoted_peer": other_key,
                        "abs_correlation": round(corr_val, 4),
                    },
                )

        promoted = entry.model_copy(
            update={
                "status": "promoted",
                "promoted_at_utc": datetime.now(UTC),
                "promoted_by": f"{approver} ({approval_ticket})",
            }
        )
        self._entries[key] = promoted
        self._persist_catalog()
        return promoted
