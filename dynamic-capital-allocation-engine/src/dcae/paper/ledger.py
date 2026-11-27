"""Paper portfolio ledger with idempotent approval-gated rebalance execution."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

from dcae.domain.models import (
    PositionSnapshot,
    ProposalStatus,
    format_utc,
    parse_utc,
    sha256_hex,
)
from dcae.errors import ValidationError
from dcae.proposals.service import ProposalService


@dataclass(frozen=True)
class PaperExecutionReceipt:
    execution_id: str
    proposal_id: str
    idempotency_key: str
    executed_by: str
    executed_at_utc: str
    total_nav_usd: float
    cash_usd: float
    total_cost_usd: float
    strategy_notional_usd: Dict[str, float]
    idempotent_replay: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "proposal_id": self.proposal_id,
            "idempotency_key": self.idempotency_key,
            "executed_by": self.executed_by,
            "executed_at_utc": self.executed_at_utc,
            "total_nav_usd": float(self.total_nav_usd),
            "cash_usd": float(self.cash_usd),
            "total_cost_usd": float(self.total_cost_usd),
            "strategy_notional_usd": {
                k: float(v) for k, v in sorted(self.strategy_notional_usd.items())
            },
            "idempotent_replay": self.idempotent_replay,
        }


class PaperPortfolioLedger:
    """In-memory / JSON-persisted paper position ledger (never executes live capital)."""

    def __init__(
        self,
        *,
        initial_cash_usd: float,
        strategy_notional_usd: Optional[Dict[str, float]] = None,
        as_of_utc: str,
        currency: str = "USD",
    ) -> None:
        if initial_cash_usd < 0.0:
            raise ValidationError("initial_cash_usd cannot be negative")
        self.currency = currency
        self.cash_usd = float(initial_cash_usd)
        self.strategy_notional_usd: Dict[str, float] = {
            str(k): float(v) for k, v in (strategy_notional_usd or {}).items()
        }
        self.as_of_utc = format_utc(parse_utc(as_of_utc))
        self.updated_at_utc = self.as_of_utc
        self.cumulative_transaction_cost_usd = 0.0
        self.rebalance_count = 0

    @property
    def total_nav_usd(self) -> float:
        return self.cash_usd + sum(self.strategy_notional_usd.values())

    def snapshot(self, *, as_of_utc: Optional[str] = None) -> PositionSnapshot:
        snap = PositionSnapshot(
            as_of_utc=as_of_utc or self.as_of_utc,
            updated_at_utc=self.updated_at_utc,
            currency=self.currency,
            total_nav_usd=self.total_nav_usd,
            cash_usd=self.cash_usd,
            strategy_notional_usd=dict(self.strategy_notional_usd),
        )
        snap.validate()
        return snap

    def mark_to_market(
        self,
        *,
        period_returns: Mapping[str, float],
        as_of_utc: str,
        cash_rate_period: float = 0.0,
    ) -> float:
        """Advance paper positions by realized strategy returns over one period."""
        ts = format_utc(parse_utc(as_of_utc))
        self.cash_usd *= 1.0 + cash_rate_period
        for sid, notional in list(self.strategy_notional_usd.items()):
            r = float(period_returns.get(sid, 0.0))
            self.strategy_notional_usd[sid] = max(0.0, notional * (1.0 + r))
        self.as_of_utc = ts
        self.updated_at_utc = ts
        return self.total_nav_usd

    def apply_approved_proposal(
        self,
        *,
        proposal_service: ProposalService,
        proposal_id: str,
        executed_by: str,
        idempotency_key: Optional[str] = None,
        executed_at_utc: Optional[str] = None,
        mode: str = "paper",
    ) -> PaperExecutionReceipt:
        """Idempotently apply an APPROVED rebalance proposal to the paper ledger."""
        if not executed_by or not executed_by.strip():
            raise ValidationError("executed_by must be non-empty")

        raw = proposal_service.require_execution_authorized(proposal_id, mode=mode)
        idem = idempotency_key or f"exec_{proposal_id}"

        existing = proposal_service.audit_store.load_paper_execution_by_key(idem)
        if existing is not None:
            return PaperExecutionReceipt(
                execution_id=str(existing["execution_id"]),
                proposal_id=str(existing["proposal_id"]),
                idempotency_key=str(existing["idempotency_key"]),
                executed_by=str(existing["executed_by"]),
                executed_at_utc=str(existing["executed_at_utc"]),
                total_nav_usd=float(existing["total_nav_usd"]),
                cash_usd=float(existing["cash_usd"]),
                total_cost_usd=float(existing["total_cost_usd"]),
                strategy_notional_usd=dict(existing["strategy_notional_usd"]),
                idempotent_replay=True,
            )

        ts = (
            format_utc(parse_utc(executed_at_utc))
            if executed_at_utc
            else format_utc(datetime.now(timezone.utc))
        )

        nav_before = self.total_nav_usd
        target_weights: Dict[str, float] = raw["decision"]["target_weights"]
        trades = raw["trades"]

        # Calculate realized transaction costs proportionally to current NAV
        bps_by_sid: Dict[str, float] = {}
        for t in trades:
            d_usd = abs(float(t["delta_notional_usd"]))
            cost_usd = float(t["estimated_cost_usd"])
            bps_by_sid[t["strategy_id"]] = (cost_usd / d_usd * 10000.0) if d_usd > 1e-9 else 0.0

        total_cost = 0.0
        new_notionals: Dict[str, float] = {}
        for sid, tw in target_weights.items():
            cur_n = self.strategy_notional_usd.get(sid, 0.0)
            tgt_n = float(tw) * nav_before
            delta_n = abs(tgt_n - cur_n)
            total_cost += delta_n * (bps_by_sid.get(sid, 0.0) / 10000.0)
            new_notionals[sid] = tgt_n

        # Deduct realized transaction cost from post-trade NAV and scale cash/positions cleanly
        nav_after = max(0.0, nav_before - total_cost)
        scale = (nav_after / nav_before) if nav_before > 0.0 else 0.0
        self.strategy_notional_usd = {
            sid: n * scale for sid, n in new_notionals.items()
        }
        self.cash_usd = max(0.0, nav_after - sum(self.strategy_notional_usd.values()))
        self.as_of_utc = ts
        self.updated_at_utc = ts
        self.cumulative_transaction_cost_usd += total_cost
        self.rebalance_count += 1

        execution_id = f"pexec_{sha256_hex({'proposal_id': proposal_id, 'idem': idem})[:14]}"
        receipt = PaperExecutionReceipt(
            execution_id=execution_id,
            proposal_id=proposal_id,
            idempotency_key=idem,
            executed_by=executed_by,
            executed_at_utc=ts,
            total_nav_usd=self.total_nav_usd,
            cash_usd=self.cash_usd,
            total_cost_usd=total_cost,
            strategy_notional_usd=dict(self.strategy_notional_usd),
            idempotent_replay=False,
        )

        # Persist receipt, transition proposal to PAPER_EXECUTED, append audit event
        proposal_service.audit_store.save_paper_execution(receipt.to_dict())
        raw["status"] = ProposalStatus.PAPER_EXECUTED.value
        raw["updated_at_utc"] = ts
        proposal_service.audit_store.save_proposal(raw)
        proposal_service.audit_store.append_event(
            event_type="PAPER_EXECUTED",
            entity_id=proposal_id,
            actor_id=executed_by,
            occurred_at_utc=ts,
            payload=receipt.to_dict(),
        )
        proposal_service.metrics.paper_executions += 1
        return receipt

    def save_json(self, path: str) -> None:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "currency": self.currency,
            "cash_usd": float(self.cash_usd),
            "strategy_notional_usd": {
                k: float(v) for k, v in sorted(self.strategy_notional_usd.items())
            },
            "as_of_utc": self.as_of_utc,
            "updated_at_utc": self.updated_at_utc,
            "cumulative_transaction_cost_usd": float(self.cumulative_transaction_cost_usd),
            "rebalance_count": self.rebalance_count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PaperPortfolioLedger:
        ledger = cls(
            initial_cash_usd=float(data["cash_usd"]),
            strategy_notional_usd=dict(data.get("strategy_notional_usd", {})),
            as_of_utc=str(data["as_of_utc"]),
            currency=str(data.get("currency", "USD")),
        )
        ledger.updated_at_utc = str(data.get("updated_at_utc", ledger.as_of_utc))
        ledger.cumulative_transaction_cost_usd = float(
            data.get("cumulative_transaction_cost_usd", 0.0)
        )
        ledger.rebalance_count = int(data.get("rebalance_count", 0))
        return ledger

    @classmethod
    def load_json(cls, path: str) -> PaperPortfolioLedger:
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
