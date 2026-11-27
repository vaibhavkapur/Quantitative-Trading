"""Domain contracts package."""

from __future__ import annotations

from dcae.domain.models import (
    SCHEMA_VERSION,
    AllocationDecision,
    AllocationStatus,
    ConstraintCheckResult,
    ConstraintSpec,
    EngineConfig,
    FeasibilityReport,
    FundSnapshot,
    ObjectiveBreakdown,
    PositionSnapshot,
    ProposalStatus,
    Provenance,
    ReturnObservation,
    StrategyExposure,
    StrategySnapshot,
    canonical_json,
    format_utc,
    parse_utc,
    sha256_hex,
)

__all__ = [
    "SCHEMA_VERSION",
    "AllocationDecision",
    "AllocationStatus",
    "ConstraintCheckResult",
    "ConstraintSpec",
    "EngineConfig",
    "FeasibilityReport",
    "FundSnapshot",
    "ObjectiveBreakdown",
    "PositionSnapshot",
    "ProposalStatus",
    "Provenance",
    "ReturnObservation",
    "StrategyExposure",
    "StrategySnapshot",
    "canonical_json",
    "format_utc",
    "parse_utc",
    "sha256_hex",
]
