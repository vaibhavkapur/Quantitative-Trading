"""Paper-only signal exporter with hard execution-disabled safeguard and SHA-256 provenance."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from alpha_engine.catalog.registry import AlphaCatalog
from alpha_engine.contracts.models import PaperSignalExport
from alpha_engine.errors import (
    DataValidationError,
    ExecutionDisabledError,
    LookAheadViolationError,
    PromotionGateError,
)


class PaperSignalExporter:
    """Exports promoted signals strictly in paper-only mode with full provenance and integrity digest."""

    def __init__(self, catalog: AlphaCatalog) -> None:
        self.catalog = catalog

    def export_paper_signals(
        self,
        *,
        signal_id: str,
        version: str,
        weights_df: pl.DataFrame,
        as_of_utc: datetime,
        dataset_manifest_digest: str,
        output_path: Path | None = None,
        request_live_execution: bool = False,
    ) -> PaperSignalExport:
        if request_live_execution:
            raise ExecutionDisabledError(
                "Order execution is hard-disabled in the MVP alpha research engine. "
                "Only paper-only signal exports are permitted.",
                details={"signal_id": signal_id, "version": version},
            )

        entry = self.catalog.get_entry(signal_id, version)
        if entry.status != "promoted":
            raise PromotionGateError(
                f"Only promoted signals may be exported for paper trading; '{signal_id}@{version}' "
                f"has status '{entry.status}'.",
                details={"signal_id": signal_id, "version": version, "status": entry.status},
            )

        cutoff = as_of_utc.astimezone(UTC)
        if weights_df.height == 0:
            raise DataValidationError("Cannot export empty weights_df.")

        # Enforce strict point-in-time causality on exported records
        future_rows = weights_df.filter(
            (pl.col("available_at_utc") > pl.lit(cutoff))
            | (pl.col("event_ts_utc") > pl.lit(cutoff))
        )
        if future_rows.height > 0:
            raise LookAheadViolationError(
                f"Export weights contain {future_rows.height} rows beyond as_of_utc={cutoff.isoformat()}.",
                details={"as_of_utc": cutoff.isoformat(), "future_rows": future_rows.height},
            )

        # Select latest available cross-section <= as_of_utc (active non-exited weights)
        latest_ts = weights_df.select(pl.col("event_ts_utc").max()).item()
        latest_slice = (
            weights_df.filter(
                (pl.col("event_ts_utc") == pl.lit(latest_ts)) & (pl.col("sector") != "EXITED")
            )
            .sort("symbol")
        )

        records: list[dict] = []
        for row in latest_slice.to_dicts():
            records.append(
                {
                    "symbol": row["symbol"],
                    "sector": row["sector"],
                    "event_ts_utc": row["event_ts_utc"].isoformat(),
                    "available_at_utc": row["available_at_utc"].isoformat(),
                    "target_weight": round(float(row["weight"]), 6),
                }
            )

        gross = sum(abs(r["target_weight"]) for r in records)
        net = sum(r["target_weight"] for r in records)

        provenance = {
            "dataset_manifest_digest": dataset_manifest_digest,
            "latest_trial_id": entry.latest_trial_id,
            "promoted_by": entry.promoted_by,
            "promoted_at_utc": entry.promoted_at_utc.isoformat() if entry.promoted_at_utc else None,
            "causal_guarantee": entry.signal_definition.causal_guarantee,
            "disclaimer": "PAPER_RESEARCH_ONLY_NO_LIVE_EXECUTION_OR_EDGE_PROMISE",
        }

        payload_for_hash = {
            "signal_id": signal_id,
            "signal_version": version,
            "definition_hash": entry.signal_definition.definition_hash,
            "as_of_utc": cutoff.isoformat(),
            "weights": records,
            "provenance": provenance,
        }
        digest = hashlib.sha256(
            json.dumps(payload_for_hash, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        export = PaperSignalExport(
            export_id=f"paper-{signal_id}-{uuid.uuid4().hex[:8]}",
            exported_at_utc=datetime.now(UTC),
            mode="paper_only",
            execution_enabled=False,
            signal_id=signal_id,
            signal_version=version,
            definition_hash=entry.signal_definition.definition_hash,
            as_of_utc=cutoff,
            records_count=len(records),
            gross_exposure=round(gross, 6),
            net_exposure=round(net, 6),
            provenance=provenance,
            weights=records,
            sha256_digest=digest,
        )

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as fh:
                json.dump(export.model_dump(mode="json"), fh, indent=2, sort_keys=True)

        return export
