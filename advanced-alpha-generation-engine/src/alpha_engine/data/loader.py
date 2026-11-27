"""Point-in-time data loader with manifest checksum verification and PIT as-of joins."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from alpha_engine.config import UniversePolicy
from alpha_engine.contracts.models import DatasetManifest
from alpha_engine.contracts.validation import PointInTimeValidator, ValidationSummary
from alpha_engine.errors import DataValidationError


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ResearchDataset:
    bars: pl.DataFrame
    fundamentals: pl.DataFrame
    corporate_actions: pl.DataFrame
    membership: pl.DataFrame
    manifest: DatasetManifest
    validation: ValidationSummary


class PointInTimeDataLoader:
    """Loads Parquet research fixtures, verifies checksums, and enforces PIT contracts."""

    def __init__(self, fixtures_dir: Path, universe_policy: UniversePolicy) -> None:
        self.fixtures_dir = fixtures_dir
        self.policy = universe_policy
        self.validator = PointInTimeValidator(universe_policy)

    def load(
        self,
        *,
        as_of_utc: datetime | None = None,
        verify_checksums: bool = True,
    ) -> ResearchDataset:
        manifest_path = self.fixtures_dir / "manifest.json"
        if not manifest_path.exists():
            raise DataValidationError(
                f"Dataset manifest not found at {manifest_path}. Run fixture generator first.",
                details={"manifest_path": str(manifest_path)},
            )

        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = DatasetManifest.model_validate(json.load(fh))

        if verify_checksums:
            for fname, expected_sha in manifest.file_checksums.items():
                fpath = self.fixtures_dir / fname
                if not fpath.exists():
                    raise DataValidationError(
                        f"Missing fixture file referenced in manifest: {fname}",
                        details={"file": str(fpath)},
                    )
                actual_sha = _sha256_file(fpath)
                if actual_sha != expected_sha:
                    raise DataValidationError(
                        f"Checksum mismatch for fixture {fname}.",
                        details={"file": fname, "expected": expected_sha, "actual": actual_sha},
                    )

        bars = pl.read_parquet(self.fixtures_dir / "market_bars.parquet")
        fundamentals = pl.read_parquet(self.fixtures_dir / "fundamentals.parquet")
        actions = pl.read_parquet(self.fixtures_dir / "corporate_actions.parquet")
        membership = pl.read_parquet(self.fixtures_dir / "universe_membership.parquet")

        if as_of_utc is not None:
            cutoff = as_of_utc.astimezone(UTC)
            bars = bars.filter(
                (pl.col("available_at_utc") <= pl.lit(cutoff))
                & (pl.col("event_ts_utc") <= pl.lit(cutoff))
            )
            fundamentals = fundamentals.filter(pl.col("available_at_utc") <= pl.lit(cutoff))
            actions = actions.filter(pl.col("available_at_utc") <= pl.lit(cutoff))

        validation = self.validator.validate_all(
            bars,
            fundamentals,
            actions,
            membership,
            as_of_utc=as_of_utc,
        )

        # Attach PIT sector metadata onto bars
        bars_with_sector = bars.join(
            membership.select("symbol", "sector"),
            on="symbol",
            how="left",
        ).sort(["event_ts_utc", "symbol"])

        return ResearchDataset(
            bars=bars_with_sector,
            fundamentals=fundamentals.sort(["available_at_utc", "symbol"]),
            corporate_actions=actions.sort(["ex_date_utc", "symbol"]),
            membership=membership.sort("symbol"),
            manifest=manifest,
            validation=validation,
        )

    @staticmethod
    def attach_pit_fundamentals(
        bars_df: pl.DataFrame,
        fundamentals_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Backward as-of join using available_at_utc so only already-filed fundamentals are visible."""
        left = bars_df.sort(["symbol", "available_at_utc"])
        right = fundamentals_df.rename(
            {
                "available_at_utc": "filing_available_at_utc",
                "event_ts_utc": "filing_event_ts_utc",
                "period_end_utc": "fundamental_period_end_utc",
            }
        ).sort(["symbol", "filing_available_at_utc"])

        joined = left.join_asof(
            right.select(
                "symbol",
                "fundamental_period_end_utc",
                "filing_available_at_utc",
                "earnings_ttm_usd",
                "book_value_usd",
                "market_cap_usd",
                "roe",
            ),
            left_on="available_at_utc",
            right_on="filing_available_at_utc",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        return joined.sort(["event_ts_utc", "symbol"])
