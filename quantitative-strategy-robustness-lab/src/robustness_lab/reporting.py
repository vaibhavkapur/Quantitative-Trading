"""Machine-readable JSON and human-readable Markdown robustness report generator."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from typing import Any, Optional

import numpy as np
import pandas as pd
import scipy

from . import __version__
from .models import (
    BootstrapResult,
    DSRResult,
    IntegrityAuditResult,
    OutlierDiagnostic,
    PBOResult,
    PerturbationResult,
    PromotionVerdict,
    WalkForwardResult,
    canonical_json,
    sha256_text,
)


def collect_runtime_metadata(policy: dict[str, Any], seed: int, provenance: str) -> dict[str, Any]:
    """Capture deterministic configuration hashes and environment versions."""
    return {
        "lab_version": __version__,
        "policy_version": policy.get("policy_version", "unknown"),
        "policy_sha256": sha256_text(canonical_json(policy)),
        "seed": seed,
        "data_provenance": provenance,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
        },
    }


def compute_run_id(
    experiment_id: str,
    candidate_trial_id: str,
    registry_head_hash: str,
    policy: dict[str, Any],
    seed: int,
) -> str:
    """Stable deterministic identifier for an evaluation run."""
    payload = {
        "experiment_id": experiment_id,
        "candidate_trial_id": candidate_trial_id,
        "registry_head_hash": registry_head_hash,
        "policy_sha256": sha256_text(canonical_json(policy)),
        "seed": seed,
    }
    return f"run-{sha256_text(canonical_json(payload))[:16]}"


def build_report_payload(
    run_id: str,
    experiment_id: str,
    candidate_trial_id: str,
    registry_head_hash: str,
    policy: dict[str, Any],
    seed: int,
    provenance: str,
    cost_summary: dict[str, Any],
    integrity: Optional[IntegrityAuditResult],
    dsr: Optional[DSRResult],
    pbo: Optional[PBOResult],
    bootstrap: Optional[BootstrapResult],
    walk_forward: Optional[WalkForwardResult],
    perturbation: Optional[PerturbationResult],
    outliers: Optional[OutlierDiagnostic],
    verdict: PromotionVerdict,
    counters: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the canonical machine-readable report dictionary."""
    return {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "experiment_id": experiment_id,
        "candidate_trial_id": candidate_trial_id,
        "registry_head_hash": registry_head_hash,
        "runtime": collect_runtime_metadata(policy, seed, provenance),
        "counters": counters,
        "cost_summary": cost_summary,
        "verdict": {
            "decision": verdict.decision.value,
            "candidate_trial_id": verdict.candidate_trial_id,
            "evidence_complete": verdict.evidence_complete,
            "missing_evidence": list(verdict.missing_evidence),
            "summary": verdict.summary,
            "review_notes": list(verdict.review_notes),
            "gates": [asdict(g) for g in verdict.gates],
        },
        "diagnostics": {
            "integrity": asdict(integrity) if integrity is not None else None,
            "dsr": asdict(dsr) if dsr is not None else None,
            "pbo": asdict(pbo) if pbo is not None else None,
            "bootstrap": asdict(bootstrap) if bootstrap is not None else None,
            "walk_forward": asdict(walk_forward) if walk_forward is not None else None,
            "perturbation": asdict(perturbation) if perturbation is not None else None,
            "outliers": asdict(outliers) if outliers is not None else None,
        },
        "limitations_notice": (
            "Statistical diagnostics (DSR, CSCV PBO, block bootstrap, walk-forward, perturbation) "
            "evaluate historical consistency under stated stationarity and dependence assumptions. "
            "They do NOT prove future returns. Offline research prototype only."
        ),
    }


def render_markdown_report(payload: dict[str, Any]) -> str:
    """Render a reviewed Markdown robustness report from the machine-readable payload."""
    verdict = payload["verdict"]
    diag = payload["diagnostics"]
    runtime = payload["runtime"]
    cost = payload["cost_summary"]

    lines: list[str] = [
        f"# Strategy Robustness Evaluation Report — `{payload['experiment_id']}`",
        "",
        f"- **Run ID:** `{payload['run_id']}`",
        f"- **Candidate Trial:** `{payload['candidate_trial_id']}`",
        f"- **Promotion Verdict:** **`{verdict['decision']}`**",
        f"- **Data Provenance:** `{runtime['data_provenance']}`",
        f"- **Registry Head Hash:** `{payload['registry_head_hash'][:16]}...`",
        f"- **Policy Version / Seed:** `v{runtime['policy_version']}` / `seed={runtime['seed']}`",
        "",
        f"> **Summary:** {verdict['summary']}",
        "",
        "## 1. Promotion Gate Scorecard",
        "",
        "| Gate | Category | Result | Observed | Threshold |",
        "|---|---|---|---|---|",
    ]

    for g in verdict["gates"]:
        status = "PASS" if g["passed"] else "FAIL"
        obs = g["observed_value"]
        obs_str = json.dumps(obs) if isinstance(obs, (dict, list)) else str(obs)
        lines.append(f"| `{g['gate_name']}` | {g['category']} | **{status}** | `{obs_str}` | `{g['threshold']}` |")

    lines.extend([
        "",
        "## 2. Transaction Costs & Net Performance",
        "",
        f"- **Total Gross Compounded Return:** `{cost.get('total_gross_compounded', 0.0):.4%}`",
        f"- **Total Net Compounded Return:** `{cost.get('total_net_compounded', 0.0):.4%}`",
        f"- **Mean Per-Period Cost Drag:** `{cost.get('mean_period_cost_bps', 0.0):.2f} bps`",
        f"- **Annualized Cost Drag:** `{cost.get('annualized_cost_drag', 0.0):.4%}`",
        "",
    ])

    dsr = diag.get("dsr")
    if dsr is not None:
        lines.extend([
            "## 3. Multiple-Testing Diagnostics (DSR & PSR)",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Observed Net Sharpe (Annualized) | `{dsr['observed_sharpe_annual']:.4f}` |",
            f"| Probabilistic Sharpe Ratio vs Benchmark | `{dsr['psr_vs_benchmark']:.4f}` |",
            f"| Deflated Sharpe Ratio (DSR) | `{dsr['deflated_sharpe_ratio']:.4f}` |",
            f"| Raw Logged Trials / Effective Trials ($N_{{eff}}$) | `{dsr['raw_trial_count']}` / `{dsr['effective_trial_count']:.2f}` |",
            f"| Expected Max Null Sharpe ($SR^*$, Annualized) | `{dsr['expected_max_sharpe_annual']:.4f}` |",
            f"| Skewness / Excess Kurtosis | `{dsr['skewness']:.3f}` / `{dsr['excess_kurtosis']:.3f}` |",
            f"| Autocorrelation Variance Inflation Factor (VIF) | `{dsr['autocorr_variance_inflation']:.3f}` |",
            "",
        ])

    pbo = diag.get("pbo")
    if pbo is not None:
        lines.extend([
            "## 4. Probability of Backtest Overfitting (CSCV PBO)",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Partitions ($S$) / Splits Evaluated | `{pbo['n_partitions']}` / `{pbo['n_combinations_evaluated']}` (exact={pbo['exact_enumeration']}) |",
            f"| Probability of Backtest Overfitting (PBO) | `{pbo['pbo']:.4f}` |",
            f"| Probability of OOS Loss | `{pbo['prob_oos_loss']:.4f}` |",
            f"| Median OOS Rank Logit | `{pbo['median_logit']:.4f}` |",
            f"| Mean IS vs OOS Selected Sharpe | `{pbo['mean_is_sharpe_selected']:.4f}` -> `{pbo['mean_oos_sharpe_selected']:.4f}` |",
            f"| IS-to-OOS Degradation Slope | `{pbo['is_to_oos_regression_slope']:.4f}` |",
            "",
        ])

    boot = diag.get("bootstrap")
    if boot is not None:
        lines.extend([
            "## 5. Block Bootstrap Uncertainty Bounds",
            "",
            f"- **Method:** `{boot['method']}` (block_length=`{boot['block_length']}`, resamples=`{boot['n_resamples']}`, seed=`{boot['seed']}`)",
            f"- **Annualized Sharpe {int(boot['confidence_level']*100)}% CI:** `[{boot['sharpe_ci_lower']:.4f}, {boot['sharpe_ci_upper']:.4f}]` (SE=`{boot['sharpe_std_error']:.4f}`)",
            f"- **Annualized Mean Return {int(boot['confidence_level']*100)}% CI:** `[{boot['mean_return_ci_lower']:.4%}, {boot['mean_return_ci_upper']:.4%}]`",
            f"- **Max Drawdown Observed / {int(boot['confidence_level']*100)}% Upper Bound:** `{boot['max_drawdown_observed']:.2%}` / `{boot['max_drawdown_ci_upper']:.2%}`",
            f"- **Empirical Bootstrap p-value ($SR \\le 0$):** `{boot['p_value_vs_zero']:.4f}`",
            "",
        ])

    wf = diag.get("walk_forward")
    if wf is not None:
        bench_sr_val = wf["aggregate_oos_benchmark_sharpe"]
        bench_sr_str = f"{bench_sr_val:.4f}" if bench_sr_val is not None else "N/A"
        lines.extend([
            "## 6. Walk-Forward Out-of-Sample Evaluation",
            "",
            f"- **Mode:** `{wf['mode']}` | train=`{wf['train_window']}` | purge=`{wf['purge_gap']}` | test=`{wf['test_window']}` | folds=`{wf['n_folds']}` | OOS obs=`{wf['total_oos_observations']}`",
            f"- **Mean IS Net Sharpe:** `{wf['mean_is_net_sharpe']:.4f}`",
            f"- **Aggregate OOS Net Sharpe:** `{wf['aggregate_oos_net_sharpe']:.4f}` (Gross: `{wf['aggregate_oos_gross_sharpe']:.4f}`)",
            f"- **Aggregate OOS Benchmark Sharpe:** `{bench_sr_str}`",
            f"- **IS-to-OOS Degradation Ratio:** `{wf['is_to_oos_degradation_ratio']:.4f}` | **Fold Hit Rate:** `{wf['oos_hit_rate']:.1%}`",
            "",
            "| Fold | Train Window | Test Window | Selected Trial | Train Net SR | Test Net SR | Bench SR |",
            "|---|---|---|---|---|---|---|",
        ])
        for f in wf["folds"]:
            b_sr = f"{f['test_benchmark_sharpe']:.3f}" if f["test_benchmark_sharpe"] is not None else "N/A"
            lines.append(
                f"| {f['fold_index']} | `{f['train_start']}` .. `{f['train_end']}` | "
                f"`{f['test_start']}` .. `{f['test_end']}` | `{f['selected_trial_id']}` | "
                f"`{f['train_net_sharpe']:.3f}` | `{f['test_net_sharpe']:.3f}` | `{b_sr}` |"
            )
        lines.append("")

    pert = diag.get("perturbation")
    if pert is not None:
        lines.extend([
            "## 7. Parameter Perturbation Stability",
            "",
            f"- **Center Trial:** `{pert['center_trial_id']}` (`{pert['center_parameters']}`, net SR=`{pert['center_net_sharpe']:.4f}`)",
            f"- **Neighbors Evaluated:** `{pert['n_neighbors_evaluated']}` | **Positive Fraction:** `{pert['positive_neighbor_fraction']:.1%}`",
            f"- **Median / Min Neighbor Net Sharpe:** `{pert['median_neighbor_sharpe']:.4f}` / `{pert['min_neighbor_sharpe']:.4f}`",
            f"- **Sharpe Cliff Ratio:** `{pert['sharpe_cliff_ratio']:.4f}` | **Stability Score:** `{pert['stability_score']:.4f}`",
            "",
        ])

    outliers = diag.get("outliers")
    if outliers is not None:
        lines.extend([
            "## 8. Outlier & Tail-Risk Diagnostics",
            "",
            f"- **Max |Return| / Max Robust |z|:** `{outliers['max_abs_return']:.4f}` / `{outliers['max_abs_zscore']:.2f}` (`{outliers['n_extreme_observations']}` extreme obs)",
            f"- **Raw vs 1%-Winsorized Annualized Sharpe:** `{outliers['raw_sharpe_annual']:.4f}` vs `{outliers['winsorized_sharpe_annual']:.4f}` (impact=`{outliers['outlier_impact_pct']*100:.1f}%`)",
            "",
        ])

    integrity = diag.get("integrity")
    if integrity is not None:
        lines.extend([
            "## 9. Registry Integrity & Cherry-Picking Audit",
            "",
            f"- **Hash-Chain Valid:** `{integrity['registry_valid']}` | **Audit Passed:** `{integrity['passed']}`",
            f"- **Logged Trials:** `{integrity['total_logged_trials']}` (completed=`{integrity['completed_trials']}`, failed/rejected=`{integrity['failed_or_rejected_trials']}`)",
            f"- **Duplicate Fraction:** `{integrity['duplicate_fraction']:.2%}` (exact=`{len(integrity['exact_duplicate_pairs'])}`, near=`{len(integrity['near_duplicate_pairs'])}`)",
            f"- **Grid Coverage:** `{integrity['covered_grid_trials']}/{integrity['expected_grid_trials']}` (`{integrity['grid_coverage_fraction']:.1%}`)"
            if integrity["grid_coverage_fraction"] is not None
            else "- **Grid Coverage:** N/A (no grid manifest supplied)",
            "",
        ])

    lines.extend(["## 10. Review Notes & Warnings", ""])
    if verdict["review_notes"]:
        for note in verdict["review_notes"]:
            lines.append(f"- {note}")
    else:
        lines.append("- None.")
    lines.append("")

    lines.extend([
        "## 11. Methodology Assumptions & Limitations",
        "",
        f"> {payload['limitations_notice']}",
        "",
        "### Key Recorded Assumptions",
        "",
    ])
    for section_key in ("dsr", "pbo", "bootstrap", "walk_forward"):
        sec = diag.get(section_key)
        if sec and sec.get("assumptions"):
            for item in sec["assumptions"]:
                lines.append(f"- **[{section_key.upper()}]** {item}")

    lines.extend([
        "",
        "### Runtime Environment",
        "",
        f"- Python `{runtime['python_version']}` | NumPy `{runtime['dependencies']['numpy']}` | "
        f"SciPy `{runtime['dependencies']['scipy']}` | pandas `{runtime['dependencies']['pandas']}`",
        f"- Policy SHA-256: `{runtime['policy_sha256']}`",
        "",
    ])
    return "\n".join(lines)


def write_report_artifacts(
    payload: dict[str, Any],
    output_dir: Path,
    basename: Optional[str] = None,
) -> tuple[Path, Path]:
    """Write machine-readable JSON and Markdown robustness reports to output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = basename or payload["run_id"]
    json_path = out / f"{stem}.json"
    md_path = out / f"{stem}.md"

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown_report(payload), encoding="utf-8")
    return json_path, md_path
