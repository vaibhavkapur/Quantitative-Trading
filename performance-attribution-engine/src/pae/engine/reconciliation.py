"""Reconciliation engine verifying return identities, residuals, provenance, and data quality."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List

from pae.config import AttributionConfig
from pae.domain.models import DataQualityException
from pae.engine.brinson import BrinsonReport
from pae.engine.contribution import ContributionReport
from pae.engine.factors import FactorAttributionReport
from pae.engine.returns import ReturnSummary


@dataclass(frozen=True)
class ReconciliationCheck:
    """Single automated reconciliation invariant check."""

    check_name: str
    passed: bool
    expected: float
    actual: float
    residual: float
    tolerance: float
    details: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReconciliationReport:
    """Comprehensive reconciliation summary with residuals and structured exceptions."""

    status: str  # "RECONCILED", "RECONCILED_WITH_WARNINGS", "DEGRADED_ZERO_NAV", "UNRECONCILED"
    return_tolerance: float
    return_tolerance_bps: float
    max_abs_daily_gross_residual: float
    max_abs_daily_net_residual: float
    max_abs_daily_net_residual_bps: float
    linked_gross_residual: float
    linked_net_residual: float
    linked_net_residual_bps: float
    total_valuation_rounding_delta_base: float
    all_contributions_linked_to_positions: bool
    checks: List[ReconciliationCheck] = field(default_factory=list)
    exceptions: List[DataQualityException] = field(default_factory=list)
    exception_counts_by_severity: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def run_reconciliation(
    config: AttributionConfig,
    returns: ReturnSummary,
    contributions: ContributionReport,
    brinson: BrinsonReport,
    factors: FactorAttributionReport,
    exceptions: List[DataQualityException],
) -> ReconciliationReport:
    """Verify daily & multi-period attribution identities and emit explicit residual exceptions."""
    checks: List[ReconciliationCheck] = []
    tol = config.return_tolerance

    max_gross_res = 0.0
    max_net_res = 0.0
    tot_rounding_delta = 0.0
    daily_breaches = 0

    for dsum in contributions.daily_summaries:
        ag = abs(dsum.gross_residual)
        an = abs(dsum.net_residual)
        max_gross_res = max(max_gross_res, ag)
        max_net_res = max(max_net_res, an)
        tot_rounding_delta += dsum.total_valuation_rounding_delta_base

        if ag > tol or an > tol:
            daily_breaches += 1
            if config.emit_exception_on_residual_breach:
                exceptions.append(
                    DataQualityException(
                        code="RESIDUAL_EXCEEDS_TOLERANCE",
                        severity="ERROR",
                        trade_date=dsum.trade_date,
                        entity_id="PORTFOLIO",
                        message=(
                            f"Reconciliation residual on {dsum.trade_date} exceeds tolerance "
                            f"({an * 10000.0:.4f} bps net, {ag * 10000.0:.4f} bps gross > "
                            f"{config.return_tolerance_bps:.4f} bps)."
                        ),
                        details={
                            "gross_residual": dsum.gross_residual,
                            "net_residual": dsum.net_residual,
                            "net_residual_bps": dsum.net_residual * 10000.0,
                            "tolerance": tol,
                            "valuation_rounding_delta_base": dsum.total_valuation_rounding_delta_base,
                        },
                    )
                )
        elif abs(dsum.total_valuation_rounding_delta_base) > 1e-9 or an > 1e-12:
            exceptions.append(
                DataQualityException(
                    code="ROUNDING_RESIDUAL_WITHIN_TOLERANCE",
                    severity="INFO",
                    trade_date=dsum.trade_date,
                    entity_id="PORTFOLIO",
                    message=(
                        f"Non-zero residual ({dsum.net_residual * 10000.0:+.6f} bps; "
                        f"rounding delta {dsum.total_valuation_rounding_delta_base:+.6f} base) "
                        f"is within tolerance ({config.return_tolerance_bps:.4f} bps) and reported explicitly."
                    ),
                    details={
                        "net_residual": dsum.net_residual,
                        "valuation_rounding_delta_base": dsum.total_valuation_rounding_delta_base,
                    },
                )
            )

    # Check 1: Daily residual tolerance across all days
    checks.append(
        ReconciliationCheck(
            check_name="daily_contribution_residual_within_tolerance",
            passed=(daily_breaches == 0),
            expected=0.0,
            actual=max_net_res,
            residual=max_net_res,
            tolerance=tol,
            details=f"{daily_breaches} day(s) breached tolerance across {len(contributions.daily_summaries)} day(s).",
        )
    )

    # Check 2: Multi-period Carino linked Net identity (sum_linked_net + linked_net_residual == cumulative_net)
    net_identity_diff = returns.cumulative_net_return - (
        contributions.sum_linked_net_contributions + contributions.linked_net_residual
    )
    checks.append(
        ReconciliationCheck(
            check_name="linked_net_return_identity",
            passed=(abs(net_identity_diff) <= 1e-9),
            expected=returns.cumulative_net_return,
            actual=contributions.sum_linked_net_contributions + contributions.linked_net_residual,
            residual=net_identity_diff,
            tolerance=1e-9,
            details="Carino-linked strategy net contributions + cash - fund fees + explicit residual == cumulative net return.",
        )
    )

    # Check 3: Multi-period Carino linked Gross identity
    gross_identity_diff = returns.cumulative_gross_return - (
        contributions.sum_linked_gross_contributions + contributions.linked_gross_residual
    )
    checks.append(
        ReconciliationCheck(
            check_name="linked_gross_return_identity",
            passed=(abs(gross_identity_diff) <= 1e-9),
            expected=returns.cumulative_gross_return,
            actual=contributions.sum_linked_gross_contributions + contributions.linked_gross_residual,
            residual=gross_identity_diff,
            tolerance=1e-9,
            details="Carino-linked strategy gross contributions + cash + explicit residual == cumulative gross return.",
        )
    )

    # Check 4: Strict daily Gross - Fee Drag == Net separation
    max_gn_diff = max(
        (abs((d.gross_return - d.fee_drag_return) - d.net_return) for d in returns.daily_returns),
        default=0.0,
    )
    checks.append(
        ReconciliationCheck(
            check_name="gross_minus_fees_equals_net_daily",
            passed=(max_gn_diff <= 1e-12),
            expected=0.0,
            actual=max_gn_diff,
            residual=max_gn_diff,
            tolerance=1e-12,
            details="Verifies gross return is never conflated with net return.",
        )
    )

    # Check 5: Brinson identity when Brinson is available
    if brinson.status in {"AVAILABLE", "PARTIAL"} and brinson.cumulative_active_gross_return is not None:
        brinson_sum = (
            brinson.total_allocation_effect
            + brinson.total_selection_effect
            + brinson.total_interaction_effect
            + brinson.total_intraday_rebalance_effect
            + brinson.total_residual
        )
        br_diff = brinson.cumulative_active_gross_return - brinson_sum
        checks.append(
            ReconciliationCheck(
                check_name="brinson_active_return_identity",
                passed=(abs(br_diff) <= 1e-9),
                expected=brinson.cumulative_active_gross_return,
                actual=brinson_sum,
                residual=br_diff,
                tolerance=1e-9,
                details="Allocation + Selection + Interaction + Intraday Rebalance + Residual == Active Gross Return.",
            )
        )

    # Check 6: Factor identity when available
    if factors.status == "AVAILABLE":
        checks.append(
            ReconciliationCheck(
                check_name="factor_return_decomposition_identity",
                passed=(abs(factors.residual_check) <= 1e-9),
                expected=factors.portfolio_gross_return_sample,
                actual=factors.explained_plus_specific_return,
                residual=factors.residual_check,
                tolerance=1e-9,
                details="Risk-free + Linked Factor Contributions + Specific Return == Sample Gross Return.",
            )
        )

    # Check 7: Every contribution links to underlying positions
    all_linked = all(len(c.position_ids) > 0 for c in contributions.by_strategy) and all(
        len(c.position_ids) > 0 for c in contributions.by_instrument
    )
    checks.append(
        ReconciliationCheck(
            check_name="every_contribution_links_to_positions",
            passed=all_linked,
            expected=1.0,
            actual=1.0 if all_linked else 0.0,
            residual=0.0 if all_linked else 1.0,
            tolerance=0.0,
            details="Confirms every strategy and instrument contribution links to PositionSnapshot IDs.",
        )
    )

    # Count exceptions by severity
    counts: Dict[str, int] = {"INFO": 0, "WARNING": 0, "ERROR": 0, "CRITICAL": 0}
    for exc in exceptions:
        counts[exc.severity] = counts.get(exc.severity, 0) + 1

    # Determine overall reconciliation status
    has_zero_nav = any(e.code == "ZERO_NAV" for e in exceptions)
    all_checks_passed = all(c.passed for c in checks)
    if has_zero_nav:
        status = "DEGRADED_ZERO_NAV"
    elif not all_checks_passed or counts.get("ERROR", 0) > 0 or counts.get("CRITICAL", 0) > 0:
        status = "UNRECONCILED"
    elif counts.get("WARNING", 0) > 0:
        status = "RECONCILED_WITH_WARNINGS"
    else:
        status = "RECONCILED"

    return ReconciliationReport(
        status=status,
        return_tolerance=tol,
        return_tolerance_bps=config.return_tolerance_bps,
        max_abs_daily_gross_residual=max_gross_res,
        max_abs_daily_net_residual=max_net_res,
        max_abs_daily_net_residual_bps=max_net_res * 10000.0,
        linked_gross_residual=contributions.linked_gross_residual,
        linked_net_residual=contributions.linked_net_residual,
        linked_net_residual_bps=contributions.linked_net_residual * 10000.0,
        total_valuation_rounding_delta_base=tot_rounding_delta,
        all_contributions_linked_to_positions=all_linked,
        checks=checks,
        exceptions=list(exceptions),
        exception_counts_by_severity=counts,
    )
