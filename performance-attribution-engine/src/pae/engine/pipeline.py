"""Deterministic end-to-end Performance Attribution Pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pae.config import AttributionConfig
from pae.domain.models import AttributionInputBatch, DataQualityException
from pae.domain.validation import validate_input_batch
from pae.engine.brinson import BrinsonReport, compute_brinson_attribution
from pae.engine.contribution import ContributionReport, compute_contributions
from pae.engine.factors import FactorAttributionReport, compute_factor_attribution
from pae.engine.returns import FxConverter, ReturnSummary, compute_returns_and_benchmark
from pae.engine.reconciliation import ReconciliationReport, run_reconciliation
from pae.engine.tca import ExecutionCostReport, compute_execution_cost_attribution


@dataclass(frozen=True)
class ReturnWaterfall:
    """Gross-to-Net and Paper-to-Actual return waterfall avoiding gross/net conflation."""

    theoretical_paper_return_before_is: float
    implicit_execution_cost_drag: float
    cumulative_gross_return: float
    explicit_trade_cost_drag: float
    other_fees_and_compounding_drag: float
    cumulative_net_return: float
    explicit_net_residual: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AttributionReport:
    """Complete, self-contained, content-addressed performance attribution report."""

    report_id: str
    version_id: str
    version_number: int
    supersedes_version_id: Optional[str]
    correction_reason: Optional[str]
    portfolio_id: str
    batch_id: str
    reporting_currency: str
    period_start: str
    period_end: str
    methodology_version: str
    config_version: str
    config_hash: str
    input_hash: str
    report_hash: str
    environment_mode: str
    order_execution_enabled: bool
    data_source: str
    entitlement_tag: str
    as_of_utc: str
    generated_at_utc: str
    disclosures: List[str]
    waterfall: ReturnWaterfall
    returns: ReturnSummary
    contributions: ContributionReport
    brinson: BrinsonReport
    factors: FactorAttributionReport
    execution_costs: ExecutionCostReport
    reconciliation: ReconciliationReport
    revision_diff: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": {
                "report_id": self.report_id,
                "version_id": self.version_id,
                "version_number": self.version_number,
                "supersedes_version_id": self.supersedes_version_id,
                "correction_reason": self.correction_reason,
                "portfolio_id": self.portfolio_id,
                "batch_id": self.batch_id,
                "reporting_currency": self.reporting_currency,
                "period_start": self.period_start,
                "period_end": self.period_end,
                "methodology_version": self.methodology_version,
                "config_version": self.config_version,
                "config_hash": self.config_hash,
                "input_hash": self.input_hash,
                "report_hash": self.report_hash,
                "environment_mode": self.environment_mode,
                "order_execution_enabled": self.order_execution_enabled,
                "data_source": self.data_source,
                "entitlement_tag": self.entitlement_tag,
                "as_of_utc": self.as_of_utc,
                "generated_at_utc": self.generated_at_utc,
                "disclosures": self.disclosures,
            },
            "revision_diff": self.revision_diff,
            "waterfall": self.waterfall.to_dict(),
            "returns": self.returns.to_dict(),
            "contributions": self.contributions.to_dict(),
            "brinson": self.brinson.to_dict(),
            "factors": self.factors.to_dict(),
            "execution_costs": self.execution_costs.to_dict(),
            "reconciliation": self.reconciliation.to_dict(),
        }


class AttributionPipeline:
    """Orchestrates validation, return/benchmark, contribution, Brinson, factor, TCA, and reconciliation."""

    def __init__(self, config: Optional[AttributionConfig] = None) -> None:
        self.config = config or AttributionConfig()

    def run(
        self,
        batch: AttributionInputBatch,
        *,
        report_id: Optional[str] = None,
        version_number: int = 1,
        supersedes_version_id: Optional[str] = None,
        correction_reason: Optional[str] = None,
        prior_report: Optional[Dict[str, Any]] = None,
        generated_at_utc: Optional[str] = None,
    ) -> AttributionReport:
        # 1. Validate canonical inputs
        exceptions: List[DataQualityException] = validate_input_batch(batch)

        # 2. Build FX converter in batch reporting currency
        fx = FxConverter(
            observations=batch.fx_rates,
            reporting_currency=batch.reporting_currency,
            max_staleness_hours=self.config.fx_max_staleness_hours,
        )

        # 3. Compute daily & cumulative returns + aligned benchmark
        returns = compute_returns_and_benchmark(batch, self.config, fx, exceptions)

        # 4. Compute daily & Carino-linked strategy/instrument/sector/currency contributions
        contributions = compute_contributions(batch, self.config, fx, returns, exceptions)

        # 5. Compute optional Brinson sector attribution
        brinson = compute_brinson_attribution(batch, self.config, returns, contributions, exceptions)

        # 6. Compute optional statistical risk-factor attribution
        factors = compute_factor_attribution(batch, self.config, returns, exceptions)

        # 7. Compute execution-cost (Implementation Shortfall) attribution
        execution_costs = compute_execution_cost_attribution(batch, self.config, fx, returns, exceptions)

        # 8. Run reconciliation checks & aggregate exceptions
        reconciliation = run_reconciliation(
            self.config, returns, contributions, brinson, factors, exceptions
        )

        # 9. Build transparent Gross-vs-Net & Paper-to-Actual return waterfall
        total_fee_drag = returns.cumulative_fee_drag
        other_fees_and_comp = total_fee_drag - execution_costs.explicit_trade_cost_return_drag
        waterfall = ReturnWaterfall(
            theoretical_paper_return_before_is=(
                returns.cumulative_gross_return + execution_costs.implicit_cost_return_drag
            ),
            implicit_execution_cost_drag=execution_costs.implicit_cost_return_drag,
            cumulative_gross_return=returns.cumulative_gross_return,
            explicit_trade_cost_drag=execution_costs.explicit_trade_cost_return_drag,
            other_fees_and_compounding_drag=other_fees_and_comp,
            cumulative_net_return=returns.cumulative_net_return,
            explicit_net_residual=reconciliation.linked_net_residual,
        )

        logical_report_id = report_id or (
            f"RPT-{batch.portfolio_id}-{returns.period_start.replace('-', '')}-"
            f"{returns.period_end.replace('-', '')}"
        )
        cfg_hash = self.config.config_hash()
        inp_hash = batch.content_hash()
        gen_ts = generated_at_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Compute revision diff if superseding an earlier report version
        rev_diff: Optional[Dict[str, Any]] = None
        if prior_report is not None:
            prev_ret = prior_report.get("returns", {})
            prev_rec = prior_report.get("reconciliation", {})
            rev_diff = {
                "prior_version_id": prior_report.get("metadata", {}).get("version_id"),
                "correction_reason": correction_reason,
                "delta_cumulative_gross_return": (
                    returns.cumulative_gross_return - float(prev_ret.get("cumulative_gross_return", 0.0))
                ),
                "delta_cumulative_net_return": (
                    returns.cumulative_net_return - float(prev_ret.get("cumulative_net_return", 0.0))
                ),
                "delta_total_fees_base": (
                    returns.total_fees_base - float(prev_ret.get("total_fees_base", 0.0))
                ),
                "delta_linked_net_residual": (
                    reconciliation.linked_net_residual - float(prev_rec.get("linked_net_residual", 0.0))
                ),
            }

        # Deterministic content hash (excludes generated_at_utc so re-runs on identical inputs hash identically)
        deterministic_body = {
            "report_id": logical_report_id,
            "version_number": version_number,
            "supersedes_version_id": supersedes_version_id,
            "correction_reason": correction_reason,
            "config_hash": cfg_hash,
            "input_hash": inp_hash,
            "returns": returns.to_dict(),
            "contributions": contributions.to_dict(),
            "brinson": brinson.to_dict(),
            "factors": factors.to_dict(),
            "execution_costs": execution_costs.to_dict(),
            "reconciliation": reconciliation.to_dict(),
        }
        report_hash = hashlib.sha256(
            json.dumps(deterministic_body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        version_id = f"{logical_report_id}-v{version_number}-{report_hash[:8]}"

        disclosures = [
            f"ENVIRONMENT: {self.config.environment_mode} — not audited live performance.",
            "GROSS VS NET: Gross returns exclude explicit trading/management fees; Net returns reflect all explicit fees and fee corrections. Gross is never presented as Net.",
            self.config.non_causation_disclaimer,
            "ORDER EXECUTION: Permanently disabled in the Performance Attribution Engine.",
        ]

        return AttributionReport(
            report_id=logical_report_id,
            version_id=version_id,
            version_number=version_number,
            supersedes_version_id=supersedes_version_id,
            correction_reason=correction_reason,
            portfolio_id=batch.portfolio_id,
            batch_id=batch.batch_id,
            reporting_currency=batch.reporting_currency,
            period_start=returns.period_start,
            period_end=returns.period_end,
            methodology_version=self.config.methodology_version,
            config_version=self.config.config_version,
            config_hash=cfg_hash,
            input_hash=inp_hash,
            report_hash=report_hash,
            environment_mode=self.config.environment_mode,
            order_execution_enabled=self.config.order_execution_enabled,
            data_source=batch.data_source,
            entitlement_tag=batch.entitlement_tag,
            as_of_utc=batch.as_of_utc,
            generated_at_utc=gen_ts,
            disclosures=disclosures,
            waterfall=waterfall,
            returns=returns,
            contributions=contributions,
            brinson=brinson,
            factors=factors,
            execution_costs=execution_costs,
            reconciliation=reconciliation,
            revision_diff=rev_diff,
        )
