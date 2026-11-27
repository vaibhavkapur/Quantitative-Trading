"""Integration tests for clean-room legacy portfolio foundation adapters."""

from __future__ import annotations

import pytest

from alpha_engine.adapters import (
    AIQuantResearchAgentAdapter,
    QuantumOrderBookAdapter,
    QuantumPairsSpreadAdapter,
    QuantumStatArbAdapter,
    ResearchHypothesisPayload,
)
from alpha_engine.config import UniversePolicy
from alpha_engine.contracts.validation import PointInTimeValidator
from alpha_engine.data.loader import ResearchDataset
from alpha_engine.errors import DataValidationError


def test_legacy_signal_adapters_conform_to_pit_contract(
    research_dataset: ResearchDataset,
    universe_policy: UniversePolicy,
) -> None:
    validator = PointInTimeValidator(universe_policy)
    adapters = [
        QuantumStatArbAdapter(window_days=20),
        QuantumPairsSpreadAdapter(window_days=30),
        QuantumOrderBookAdapter(smoothing_days=5),
    ]
    for adapter in adapters:
        assert adapter.provenance.integration_mode == "clean_room_contract_adapter"
        assert adapter.provenance.known_limitations
        sig = adapter.compute(research_dataset.bars)
        assert sig.height > 1000
        validator.validate_signal_causality(sig, research_dataset.bars)


def test_research_agent_adapter_compiles_valid_and_rejects_invalid_proposals(
    research_dataset: ResearchDataset,
    universe_policy: UniversePolicy,
) -> None:
    proposal = ResearchHypothesisPayload(
        proposal_id="PROP-001",
        template="momentum",
        signal_id="agent_mom_30_5",
        version="1.0.0",
        lookback_days=30,
        skip_days=5,
        rationale="Test medium-term causal momentum horizon.",
    )
    factor = AIQuantResearchAgentAdapter.compile_proposal(proposal)
    sig = factor.compute(research_dataset.bars)
    PointInTimeValidator(universe_policy).validate_signal_causality(sig, research_dataset.bars)

    bad = ResearchHypothesisPayload(
        proposal_id="PROP-BAD",
        template="momentum",
        signal_id="agent_bad_mom",
        lookback_days=5,
        skip_days=10,
        rationale="Invalid lookback <= skip.",
    )
    with pytest.raises(DataValidationError, match="lookback_days > skip_days"):
        AIQuantResearchAgentAdapter.compile_proposal(bad)
