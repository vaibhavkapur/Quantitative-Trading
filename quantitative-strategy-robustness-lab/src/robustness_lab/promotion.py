"""Promotion gate evaluator for quantitative strategy candidates.

Pure domain module (no file I/O). Enforces explicit hard rules:
  - Always REJECT when required diagnostic evidence is incomplete.
  - Always REJECT when registry integrity or cherry-picking audit fails.
  - Require multiple-testing survival (DSR, PBO), uncertainty bounds (bootstrap),
    out-of-sample net-of-cost walk-forward performance, and perturbation stability.
"""

from __future__ import annotations

from typing import Any, Optional

from .models import (
    BootstrapResult,
    DSRResult,
    IntegrityAuditResult,
    OutlierDiagnostic,
    PBOResult,
    PerturbationResult,
    PromotionDecision,
    PromotionGateCheck,
    PromotionVerdict,
    WalkForwardResult,
)


def evaluate_promotion(
    candidate_trial_id: str,
    policy: dict[str, Any],
    integrity: Optional[IntegrityAuditResult],
    dsr: Optional[DSRResult],
    pbo: Optional[PBOResult],
    bootstrap: Optional[BootstrapResult],
    walk_forward: Optional[WalkForwardResult],
    perturbation: Optional[PerturbationResult],
    outliers: Optional[OutlierDiagnostic],
) -> PromotionVerdict:
    """Evaluate all promotion gates and return an auditable PromotionVerdict."""
    missing: list[str] = []
    required_artifacts = {
        "integrity_audit": integrity,
        "dsr_diagnostics": dsr,
        "pbo_diagnostics": pbo,
        "block_bootstrap": bootstrap,
        "walk_forward_oos": walk_forward,
        "parameter_perturbation": perturbation,
        "outlier_diagnostics": outliers,
    }
    for name, obj in required_artifacts.items():
        if obj is None:
            missing.append(name)

    evidence_complete = len(missing) == 0
    gates: list[PromotionGateCheck] = []
    review_notes: list[str] = []

    # 1. Evidence completeness gate
    gates.append(
        PromotionGateCheck(
            gate_name="evidence_completeness",
            category="governance",
            passed=evidence_complete,
            required=True,
            observed_value="complete" if evidence_complete else f"missing: {', '.join(missing)}",
            threshold="all 7 diagnostics present",
            rationale="Promotion is strictly rejected whenever any required robustness diagnostic is missing.",
        )
    )

    # 2. Registry integrity & cherry-picking gate
    if integrity is not None:
        gates.append(
            PromotionGateCheck(
                gate_name="registry_and_trial_integrity",
                category="governance",
                passed=integrity.passed,
                required=bool(policy["promotion"].get("require_registry_integrity", True)),
                observed_value={
                    "registry_valid": integrity.registry_valid,
                    "duplicate_fraction": round(integrity.duplicate_fraction, 4),
                    "grid_coverage": integrity.grid_coverage_fraction,
                    "flags": len(integrity.cherry_picking_flags),
                },
                threshold="valid hash chain, no sequence gaps, duplicates & grid coverage within policy",
                rationale="All attempted trials (including failures) must be logged without tampering or cherry-picking.",
            )
        )
        for flag in integrity.cherry_picking_flags:
            review_notes.append(f"[Integrity] {flag}")

    # 3. Multiple testing: DSR & PSR
    dsr_cfg = policy["dsr"]
    if dsr is not None:
        gates.append(
            PromotionGateCheck(
                gate_name="deflated_sharpe_ratio",
                category="multiple_testing",
                passed=dsr.deflated_sharpe_ratio >= float(dsr_cfg["min_dsr"]),
                required=True,
                observed_value=round(dsr.deflated_sharpe_ratio, 4),
                threshold=f">= {dsr_cfg['min_dsr']}",
                rationale="Deflated Sharpe adjusts for effective trial count, skewness, kurtosis, and autocorrelation.",
            )
        )
        gates.append(
            PromotionGateCheck(
                gate_name="psr_vs_benchmark",
                category="multiple_testing",
                passed=dsr.psr_vs_benchmark >= float(dsr_cfg["min_psr_vs_benchmark"]),
                required=True,
                observed_value=round(dsr.psr_vs_benchmark, 4),
                threshold=f">= {dsr_cfg['min_psr_vs_benchmark']}",
                rationale="Probabilistic Sharpe Ratio must exceed threshold against configured benchmark SR.",
            )
        )
        for w in dsr.warnings:
            review_notes.append(f"[DSR] {w}")

    # 4. Multiple testing: PBO
    pbo_cfg = policy["pbo"]
    if pbo is not None:
        gates.append(
            PromotionGateCheck(
                gate_name="probability_of_backtest_overfitting",
                category="multiple_testing",
                passed=pbo.pbo <= float(pbo_cfg["max_pbo"]),
                required=True,
                observed_value=round(pbo.pbo, 4),
                threshold=f"<= {pbo_cfg['max_pbo']}",
                rationale="CSCV PBO measures how often the in-sample optimal strategy underperforms OOS median.",
            )
        )
        gates.append(
            PromotionGateCheck(
                gate_name="pbo_oos_loss_probability",
                category="multiple_testing",
                passed=pbo.prob_oos_loss <= float(pbo_cfg["max_oos_loss_prob"]),
                required=True,
                observed_value=round(pbo.prob_oos_loss, 4),
                threshold=f"<= {pbo_cfg['max_oos_loss_prob']}",
                rationale="Probability of negative OOS Sharpe for the IS-selected strategy must remain bounded.",
            )
        )
        for w in pbo.warnings:
            review_notes.append(f"[PBO] {w}")

    # 5. Block bootstrap uncertainty bounds
    boot_cfg = policy["bootstrap"]
    if bootstrap is not None:
        gates.append(
            PromotionGateCheck(
                gate_name="bootstrap_sharpe_ci_lower",
                category="uncertainty",
                passed=bootstrap.sharpe_ci_lower >= float(boot_cfg["min_sharpe_ci_lower"]),
                required=True,
                observed_value=round(bootstrap.sharpe_ci_lower, 4),
                threshold=f">= {boot_cfg['min_sharpe_ci_lower']}",
                rationale="Lower confidence bound of block-bootstrapped Sharpe ratio must stay above floor.",
            )
        )

    # 6. Walk-forward OOS & benchmark comparison
    wf_cfg = policy["walk_forward"]
    min_oos_obs = int(policy.get("min_oos_observations", 40))
    if walk_forward is not None:
        gates.append(
            PromotionGateCheck(
                gate_name="walk_forward_oos_sample_size",
                category="out_of_sample",
                passed=walk_forward.total_oos_observations >= min_oos_obs,
                required=True,
                observed_value=walk_forward.total_oos_observations,
                threshold=f">= {min_oos_obs}",
                rationale="Sufficient stitched OOS observations are required before promotion.",
            )
        )
        gates.append(
            PromotionGateCheck(
                gate_name="walk_forward_oos_net_sharpe",
                category="out_of_sample",
                passed=walk_forward.aggregate_oos_net_sharpe >= float(wf_cfg["min_oos_net_sharpe"]),
                required=True,
                observed_value=round(walk_forward.aggregate_oos_net_sharpe, 4),
                threshold=f">= {wf_cfg['min_oos_net_sharpe']}",
                rationale="Out-of-sample net-of-cost Sharpe ratio must survive transaction and borrow drag.",
            )
        )
        gates.append(
            PromotionGateCheck(
                gate_name="walk_forward_is_to_oos_degradation",
                category="out_of_sample",
                passed=walk_forward.is_to_oos_degradation_ratio <= float(wf_cfg["max_is_to_oos_degradation"]),
                required=True,
                observed_value=round(walk_forward.is_to_oos_degradation_ratio, 4),
                threshold=f"<= {wf_cfg['max_is_to_oos_degradation']}",
                rationale="Severe decay from training Sharpe to OOS Sharpe signals overfitting.",
            )
        )
        if walk_forward.aggregate_oos_benchmark_sharpe is not None:
            excess_sr = walk_forward.aggregate_oos_net_sharpe - walk_forward.aggregate_oos_benchmark_sharpe
            min_exc = float(wf_cfg.get("min_oos_excess_over_benchmark_sharpe", 0.0))
            gates.append(
                PromotionGateCheck(
                    gate_name="walk_forward_oos_vs_benchmark",
                    category="out_of_sample",
                    passed=excess_sr >= min_exc,
                    required=True,
                    observed_value=round(excess_sr, 4),
                    threshold=f">= {min_exc}",
                    rationale="OOS net Sharpe should exceed simple benchmark Sharpe over the identical OOS periods.",
                )
            )

    # 7. Parameter perturbation stability
    pert_cfg = policy["perturbation"]
    if perturbation is not None:
        gates.append(
            PromotionGateCheck(
                gate_name="perturbation_neighbor_coverage",
                category="perturbation",
                passed=perturbation.n_neighbors_evaluated >= 1,
                required=True,
                observed_value=perturbation.n_neighbors_evaluated,
                threshold=">= 1 local neighbor",
                rationale="Isolated parameter combinations with no tested neighborhood cannot be promoted.",
            )
        )
        gates.append(
            PromotionGateCheck(
                gate_name="perturbation_positive_neighbors",
                category="perturbation",
                passed=perturbation.positive_neighbor_fraction >= float(pert_cfg["min_positive_neighbor_fraction"]),
                required=True,
                observed_value=round(perturbation.positive_neighbor_fraction, 4),
                threshold=f">= {pert_cfg['min_positive_neighbor_fraction']}",
                rationale="Majority of nearby parameter configurations must retain positive net Sharpe.",
            )
        )
        gates.append(
            PromotionGateCheck(
                gate_name="perturbation_sharpe_cliff",
                category="perturbation",
                passed=perturbation.sharpe_cliff_ratio <= float(pert_cfg["max_sharpe_cliff_ratio"]),
                required=True,
                observed_value=round(perturbation.sharpe_cliff_ratio, 4),
                threshold=f"<= {pert_cfg['max_sharpe_cliff_ratio']}",
                rationale="Sharp drop-off from center to median neighbor indicates fragile parameter tuning.",
            )
        )
        gates.append(
            PromotionGateCheck(
                gate_name="perturbation_stability_score",
                category="perturbation",
                passed=perturbation.stability_score >= float(pert_cfg["min_stability_score"]),
                required=True,
                observed_value=round(perturbation.stability_score, 4),
                threshold=f">= {pert_cfg['min_stability_score']}",
                rationale="Composite local plateau stability score must meet policy minimum.",
            )
        )
        for w in perturbation.warnings:
            review_notes.append(f"[Perturbation] {w}")

    # 8. Outlier sensitivity & extreme tail-risk gate
    if outliers is not None:
        max_kurt = float(policy["integrity"].get("max_excess_kurtosis_warning", 20.0))
        outlier_ok = (
            outliers.n_extreme_observations == 0
            or (abs(outliers.outlier_impact_pct) <= 0.50 and outliers.excess_kurtosis <= max_kurt)
        )
        gates.append(
            PromotionGateCheck(
                gate_name="outlier_dominance_check",
                category="tail_risk",
                passed=outlier_ok,
                required=True,
                observed_value={
                    "n_extreme": outliers.n_extreme_observations,
                    "excess_kurtosis": round(outliers.excess_kurtosis, 2),
                    "winsorized_sharpe": round(outliers.winsorized_sharpe_annual, 3),
                    "impact_pct": round(outliers.outlier_impact_pct, 3),
                },
                threshold=f"no extreme outliers with excess kurtosis > {max_kurt:.1f} or >50% winsorized Sharpe drop",
                rationale="Performance driven by extreme outliers or severe fat tails undermines asymptotic Sharpe diagnostics.",
            )
        )
        for w in outliers.warnings:
            review_notes.append(f"[Outliers] {w}")

    # Determine final verdict
    if not evidence_complete:
        decision = PromotionDecision.REJECT
        summary = f"REJECTED: Incomplete evidence ({', '.join(missing)}). Promotion requires full diagnostic suite."
    elif integrity is None or not integrity.passed:
        decision = PromotionDecision.REJECT
        summary = "REJECTED: Trial registry integrity or cherry-picking audit failed."
    else:
        failed_required = [g for g in gates if g.required and not g.passed]
        if not failed_required:
            decision = PromotionDecision.PROMOTE
            summary = (
                f"PROMOTED (research/paper stage): Candidate '{candidate_trial_id}' passed all "
                f"{len(gates)} robustness gates under policy v{policy['policy_version']}."
            )
        else:
            # Check if only DSR or PBO narrowly missed within hold margin while OOS, integrity & perturbation passed
            failed_names = {g.gate_name for g in failed_required}
            marginal_only = failed_names.issubset({"deflated_sharpe_ratio", "probability_of_backtest_overfitting"})
            promo_cfg = policy["promotion"]
            dsr_in_hold = dsr is not None and dsr.deflated_sharpe_ratio >= float(promo_cfg.get("hold_margin_dsr", 0.85))
            pbo_in_hold = pbo is not None and pbo.pbo <= float(promo_cfg.get("hold_margin_pbo", 0.40))
            if marginal_only and dsr_in_hold and pbo_in_hold:
                decision = PromotionDecision.HOLD_FOR_REVIEW
                summary = (
                    f"HOLD FOR REVIEW: Candidate '{candidate_trial_id}' passed integrity, OOS, and perturbation gates "
                    f"but fell in the marginal review band on: {', '.join(sorted(failed_names))}."
                )
            else:
                decision = PromotionDecision.REJECT
                summary = (
                    f"REJECTED: Candidate '{candidate_trial_id}' failed {len(failed_required)} required gate(s): "
                    f"{', '.join(g.gate_name for g in failed_required)}."
                )

    return PromotionVerdict(
        decision=decision,
        candidate_trial_id=candidate_trial_id,
        evidence_complete=evidence_complete,
        missing_evidence=tuple(missing),
        gates=tuple(gates),
        summary=summary,
        review_notes=tuple(review_notes),
    )
