"""Strategy performance and exposure snapshot validation and publishing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from dcae.domain.models import (
    EngineConfig,
    FundSnapshot,
    format_utc,
    parse_utc,
)
from dcae.errors import PointInTimeViolationError, StaleDataError


@dataclass(frozen=True)
class SnapshotManifest:
    """Published metadata record for a validated FundSnapshot."""

    snapshot_id: str
    schema_version: str
    as_of_utc: str
    published_at_utc: str
    content_hash: str
    strategy_count: int
    eligible_strategy_count: int
    position_age_hours: float
    provenance: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "schema_version": self.schema_version,
            "as_of_utc": self.as_of_utc,
            "published_at_utc": self.published_at_utc,
            "content_hash": self.content_hash,
            "strategy_count": self.strategy_count,
            "eligible_strategy_count": self.eligible_strategy_count,
            "position_age_hours": float(self.position_age_hours),
            "provenance": self.provenance,
        }


class SnapshotPublisher:
    """Validates, enforces point-in-time / freshness contracts, and versions snapshots."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self._store: Dict[str, FundSnapshot] = {}
        self._manifests: Dict[str, SnapshotManifest] = {}

    def validate_and_publish(
        self,
        snapshot: FundSnapshot,
        *,
        enforce_freshness: bool = True,
        enforce_point_in_time: bool = True,
    ) -> SnapshotManifest:
        """Validate schema, point-in-time invariants, and position freshness."""
        snapshot.validate()
        as_of = parse_utc(snapshot.as_of_utc)
        pos_updated = parse_utc(snapshot.positions.updated_at_utc)
        pos_as_of = parse_utc(snapshot.positions.as_of_utc)

        if enforce_point_in_time and pos_updated > as_of:
            raise PointInTimeViolationError(
                "Position snapshot updated_at_utc is after snapshot as_of_utc",
                context={
                    "snapshot_id": snapshot.snapshot_id,
                    "as_of_utc": snapshot.as_of_utc,
                    "positions_updated_at_utc": snapshot.positions.updated_at_utc,
                },
            )

        age_hours = max(
            (as_of - pos_updated).total_seconds() / 3600.0,
            (as_of - pos_as_of).total_seconds() / 3600.0,
        )
        if enforce_freshness and age_hours > self.config.max_position_staleness_hours:
            raise StaleDataError(
                f"Position snapshot is stale ({age_hours:.2f}h > {self.config.max_position_staleness_hours:.2f}h)",
                context={
                    "snapshot_id": snapshot.snapshot_id,
                    "position_age_hours": age_hours,
                    "max_position_staleness_hours": self.config.max_position_staleness_hours,
                    "positions_updated_at_utc": snapshot.positions.updated_at_utc,
                    "as_of_utc": snapshot.as_of_utc,
                },
            )

        content_hash = snapshot.content_hash()
        eligible_count = sum(1 for s in snapshot.strategies if s.eligible)
        manifest = SnapshotManifest(
            snapshot_id=snapshot.snapshot_id,
            schema_version=snapshot.schema_version,
            as_of_utc=format_utc(as_of),
            published_at_utc=format_utc(parse_utc(snapshot.published_at_utc)),
            content_hash=content_hash,
            strategy_count=len(snapshot.strategies),
            eligible_strategy_count=eligible_count,
            position_age_hours=age_hours,
            provenance=snapshot.provenance.to_dict(),
        )
        self._store[snapshot.snapshot_id] = snapshot
        self._manifests[snapshot.snapshot_id] = manifest
        return manifest

    def get(self, snapshot_id: str) -> Optional[FundSnapshot]:
        return self._store.get(snapshot_id)

    def get_manifest(self, snapshot_id: str) -> Optional[SnapshotManifest]:
        return self._manifests.get(snapshot_id)

    def list_manifests(self) -> List[SnapshotManifest]:
        return list(self._manifests.values())

    @staticmethod
    def from_payload(payload: Mapping[str, Any]) -> FundSnapshot:
        return FundSnapshot.from_dict(payload)

    @staticmethod
    def point_in_time_filter(snapshot: FundSnapshot) -> Tuple[Dict[str, List[float]], Dict[str, int]]:
        """Extract strictly point-in-time returns (event_at <= as_of and available_at <= as_of)."""
        as_of = parse_utc(snapshot.as_of_utc)
        series: Dict[str, List[float]] = {}
        dropped_future: Dict[str, int] = {}
        for strat in snapshot.strategies:
            vals: List[float] = []
            dropped = 0
            for obs in strat.returns:
                ev = parse_utc(obs.event_at_utc)
                av = parse_utc(obs.available_at_utc)
                if ev <= as_of and av <= as_of:
                    vals.append(float(obs.period_return))
                else:
                    dropped += 1
            series[strat.strategy_id] = vals
            dropped_future[strat.strategy_id] = dropped
        return series, dropped_future
