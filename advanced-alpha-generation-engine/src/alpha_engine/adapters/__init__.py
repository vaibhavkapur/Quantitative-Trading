"""Clean-room adapter interfaces for external and legacy portfolio foundations."""

from alpha_engine.adapters.base import AdapterProvenance, ExternalSignalAdapter
from alpha_engine.adapters.legacy_foundations import (
    AIQuantResearchAgentAdapter,
    QuantumOrderBookAdapter,
    QuantumPairsSpreadAdapter,
    QuantumStatArbAdapter,
    ResearchHypothesisPayload,
)

__all__ = [
    "AIQuantResearchAgentAdapter",
    "AdapterProvenance",
    "ExternalSignalAdapter",
    "QuantumOrderBookAdapter",
    "QuantumPairsSpreadAdapter",
    "QuantumStatArbAdapter",
    "ResearchHypothesisPayload",
]
