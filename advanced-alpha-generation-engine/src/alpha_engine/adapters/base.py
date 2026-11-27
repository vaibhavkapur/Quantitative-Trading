"""Base external adapter protocol separating provider boundaries from core engine logic."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

from alpha_engine.factors.base import BaseFactor


class AdapterProvenance(BaseModel):
    model_config = ConfigDict(frozen=True)

    adapter_name: str
    upstream_reference: str
    ip_clearance_status: str
    integration_mode: str = "clean_room_contract_adapter"
    pinned_commit: str | None = None
    known_limitations: list[str]


class ExternalSignalAdapter(BaseFactor, ABC):
    """Base contract for integrating external or legacy signal prototypes."""

    @property
    @abstractmethod
    def provenance(self) -> AdapterProvenance:
        """Return provenance, IP status, and limitations of the adapted foundation."""
